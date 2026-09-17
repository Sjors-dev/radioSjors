"""Local text-to-speech for the DJ patter.

Piper is the intended engine: no per-use cost, small enough for a Celeron.
Rendering is slow on that CPU, which is exactly why the planner works an hour
ahead -- the buffer hides the latency completely.

Nothing here is allowed to be fatal.  If Piper is missing, broken (the target
CPU has no AVX, and some onnxruntime builds fault on it), or simply too slow,
render() returns None and the queue quietly plays music without patter.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from .config import Config

log = logging.getLogger("tts")

# piper's CLI flag spelling has changed between the standalone binary and the
# pip package, so try the known variants once and remember which one works.
_OUTPUT_FLAGS = ("-f", "--output_file", "--output-file")
_LENGTH_FLAGS = ("--length_scale", "--length-scale")


class TTS:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.engine = str(cfg.get("tts.engine", "piper_cli")).lower()
        self.voice_model = str(cfg.get("tts.voice_model", "") or "")
        self.length_scale = float(cfg.get("tts.length_scale", 1.0))
        # A beat between sentences. Does more for how natural a line sounds
        # than slowing the whole voice down, which just sounds sedated.
        self.sentence_silence = float(cfg.get("tts.sentence_silence", 0.0))
        self.timeout = int(cfg.get("tts.timeout_seconds", 180))
        self.loudnorm = bool(cfg.get("tts.loudnorm", True))
        self.out_dir = cfg.path("queue")
        self._output_flag: str | None = None
        self._length_flag: str | None = None
        self._ok = False
        # Circuit breaker: once the engine is clearly broken, stop paying for a
        # doomed subprocess per patter line (a whole block's worth, every hour,
        # on two cores) and just re-probe occasionally.
        self._failures = 0
        self._muted_until = 0.0

    # -- availability -------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self.engine != "none"

    def _base_command(self) -> list[str] | None:
        if self.engine == "piper_cli":
            exe = shutil.which("piper")
            return [exe] if exe else None
        if self.engine == "piper_python":
            import sys
            return [sys.executable, "-m", "piper"]
        if self.engine == "espeak":
            exe = shutil.which("espeak-ng") or shutil.which("espeak")
            return [exe] if exe else None
        return None

    def check(self) -> tuple[bool, str]:
        """Probe the configured engine. Returns (ok, human readable detail)."""
        if not self.enabled:
            return True, "tts disabled (engine: none) - music only, no patter"

        base = self._base_command()
        if base is None:
            return False, f"engine {self.engine!r} not found on PATH"

        if self.engine.startswith("piper"):
            if not self.voice_model:
                return False, "tts.voice_model is not set"
            if not Path(self.voice_model).exists():
                return False, f"voice model missing: {self.voice_model}"
            config_json = Path(str(self.voice_model) + ".json")
            if not config_json.exists():
                return False, f"voice config missing: {config_json}"

        with tempfile.TemporaryDirectory(prefix="airadio-tts-") as tmp:
            probe = Path(tmp) / "probe.wav"
            produced = self._synthesize("Radio check, one two.", probe)
            if produced is None or not produced.exists() or produced.stat().st_size < 1000:
                return False, f"{self.engine} produced no audio"

        self._ok = True
        self._failures = 0
        self._muted_until = 0.0
        return True, f"{self.engine} ok"

    # -- rendering ----------------------------------------------------------

    def render(self, text: str, name_hint: str = "patter") -> Path | None:
        """Render one patter line to a wav file. Returns None on any failure."""
        text = (text or "").strip()
        if not text or not self.enabled:
            return None

        if time.time() < self._muted_until:
            log.debug("tts still muted after %d failures, skipping patter",
                      self._failures)
            return None

        # Every setting that changes how the line sounds is in the key, so
        # tuning the voice never replays a stale render.
        digest = hashlib.sha1(
            f"{self.engine}|{self.voice_model}|{self.length_scale}"
            f"|{self.sentence_silence}|{text}".encode("utf-8")
        ).hexdigest()[:12]
        target = self.out_dir / f"{name_hint}-{digest}.wav"
        if target.exists() and target.stat().st_size > 1000:
            log.debug("patter cache hit: %s", target.name)
            return target

        with tempfile.TemporaryDirectory(prefix="airadio-tts-") as tmp:
            raw = Path(tmp) / "raw.wav"
            produced = self._synthesize(text, raw)
            if produced is None or not produced.exists():
                self._failures += 1
                if self._failures >= 3:
                    self._muted_until = time.time() + 600
                    log.warning("tts has failed %d times in a row; muting patter for "
                                "10 minutes. Music is unaffected. Run "
                                "'main.py doctor' to see why.", self._failures)
                else:
                    log.warning("tts render failed for: %.60s", text)
                return None
            self._failures = 0
            if not self._postprocess(produced, target):
                try:
                    shutil.copyfile(produced, target)
                except Exception as exc:
                    log.warning("could not store patter audio: %s", exc)
                    return None

        if not target.exists() or target.stat().st_size < 1000:
            return None
        log.info("rendered patter (%d chars) -> %s", len(text), target.name)
        return target

    # -- engines ------------------------------------------------------------

    def _synthesize(self, text: str, out_path: Path) -> Path | None:
        base = self._base_command()
        if base is None:
            return None

        if self.engine == "espeak":
            command = base + ["-w", str(out_path), "-s", "150", "--stdin"]
            return self._run(command, text, out_path)

        output_flags = ([self._output_flag] if self._output_flag else list(_OUTPUT_FLAGS))
        if abs(self.length_scale - 1.0) <= 0.001:
            # Default speed: the flag is never added, so trying its spellings
            # would just run the same command three times.
            length_flags: list[str | None] = [None]
        elif self._length_flag:
            length_flags = [self._length_flag]
        else:
            length_flags = list(_LENGTH_FLAGS) + [None]

        for output_flag in output_flags:
            for length_flag in length_flags:
                command = base + ["-m", self.voice_model, output_flag, str(out_path)]
                if length_flag and abs(self.length_scale - 1.0) > 0.001:
                    command += [length_flag, str(self.length_scale)]
                    # Match the flag style piper accepted for length.
                    silence_flag = ("--sentence_silence" if "_" in length_flag
                                    else "--sentence-silence")
                elif self._length_flag:
                    silence_flag = ("--sentence_silence"
                                    if "_" in self._length_flag
                                    else "--sentence-silence")
                else:
                    silence_flag = None

                if silence_flag and self.sentence_silence > 0:
                    command += [silence_flag, str(self.sentence_silence)]

                result = self._run(command, text, out_path)
                if result is not None:
                    self._output_flag = output_flag
                    self._length_flag = length_flag
                    return result

                # A pause between sentences is a nicety, not worth failing for.
                if silence_flag and self.sentence_silence > 0:
                    retry = [part for part in command
                             if part not in (silence_flag, str(self.sentence_silence))]
                    result = self._run(retry, text, out_path)
                    if result is not None:
                        self._output_flag = output_flag
                        self._length_flag = length_flag
                        self.sentence_silence = 0.0
                        log.info("piper rejected %s, continuing without pauses",
                                 silence_flag)
                        return result
        return None

    def _run(self, command: list[str], text: str, out_path: Path) -> Path | None:
        try:
            completed = subprocess.run(
                command,
                input=text.encode("utf-8"),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=self.timeout,
            )
        except subprocess.TimeoutExpired:
            log.warning("tts timed out after %ss", self.timeout)
            return None
        except FileNotFoundError:
            return None
        except Exception as exc:
            log.warning("tts subprocess error: %s", exc)
            return None

        if completed.returncode != 0:
            log.debug("tts command failed (%s): %s", completed.returncode,
                      completed.stderr.decode("utf-8", "replace")[:300])
            return None
        if not out_path.exists() or out_path.stat().st_size < 1000:
            return None
        return out_path

    def _postprocess(self, source: Path, target: Path) -> bool:
        """Normalise loudness and resample so patter sits level with music."""
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            return False
        filters = "loudnorm=I=-16:TP=-1.5:LRA=11" if self.loudnorm else "anull"
        command = [ffmpeg, "-y", "-loglevel", "error", "-i", str(source),
                   "-af", filters, "-ar", "44100", "-ac", "2", str(target)]
        try:
            completed = subprocess.run(command, stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE, timeout=self.timeout)
        except Exception as exc:
            log.warning("ffmpeg postprocess failed: %s", exc)
            return False
        if completed.returncode != 0:
            log.warning("ffmpeg postprocess failed: %s",
                        completed.stderr.decode("utf-8", "replace")[:300])
            return False
        return target.exists() and target.stat().st_size > 1000

    # -- housekeeping -------------------------------------------------------

    def prune(self, keep_hours: float = 48.0) -> int:
        """Delete rendered patter older than keep_hours that is no longer queued."""
        cutoff = time.time() - keep_hours * 3600
        removed = 0
        for path in self.out_dir.glob("*.wav"):
            try:
                if path.stat().st_mtime < cutoff:
                    path.unlink()
                    removed += 1
            except Exception:
                pass
        if removed:
            log.info("pruned %d old patter files", removed)
        return removed
