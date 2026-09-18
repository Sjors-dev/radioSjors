"""Local text-to-speech for the DJ patter.

Piper is the default engine: no per-use cost, small enough for a Celeron.
Rendering is slow on that CPU, which is exactly why the planner works an hour
ahead -- the buffer hides the latency completely. tts.engine: edge_tts trades
that guarantee for noticeably more expressive voices (Microsoft's free Edge
neural voices, no key, no bill), reached over the network through an
unofficial endpoint -- so every host configured for it also needs a piper
voice_model, which _synthesize_edge()'s caller falls back to per line
whenever edge-tts is offline, rate-limited, or blocked.

The station has two hosts, so this module keeps a voice per host and can render
a back-and-forth exchange into a single audio file.  One file, not one per
line: the stream feeder treats a queue item as an atomic thing to push, and an
exchange split across four queue items would let a song land in the middle of
a conversation.

Nothing here is allowed to be fatal.  If Piper is missing, broken (the target
CPU has no AVX, and some onnxruntime builds fault on it), or simply too slow,
render() returns None and the queue quietly plays music without patter.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

from .config import Config

log = logging.getLogger("tts")

# piper's CLI flag spelling has changed between the standalone binary and the
# pip package, so try the known variants once and remember which one works.
_OUTPUT_FLAGS = ("-f", "--output_file", "--output-file")
_LENGTH_FLAGS = ("--length_scale", "--length-scale")

# The co-host voice install.sh downloads. Used to find it next to the main
# voice when the config does not spell out a path.
COHOST_VOICE_FILE = "en_GB-jenny_dioco-medium.onnx"


def _number(value, default: float) -> float:
    """A host may leave a setting out, or leave it blank in YAML (which reads
    back as None). Both mean 'use the station default'."""
    try:
        return float(default if value is None else value)
    except (TypeError, ValueError):
        return float(default)


class Voice:
    """One host's speaking settings.

    length_scale and sentence_silence control pacing. noise_scale and
    noise_w are Piper's own VITS-inherited expressiveness knobs: noise_scale
    varies the acoustic generation itself (higher = more vocal variation,
    can start sounding rough if pushed too far), noise_w varies phoneme
    duration (higher = less metronomic, more natural-sounding rhythm). Both
    default to exactly Piper's own stock values, so a host that does not set
    them sounds identical to before this existed.

    edge_voice/edge_rate/edge_pitch only matter when tts.engine is
    edge_tts -- model stays populated in that mode too, since it is used as
    this host's piper fallback when edge-tts is unreachable.
    """

    def __init__(self, name: str, model: str, length_scale: float,
                 sentence_silence: float, noise_scale: float = 0.667,
                 noise_w: float = 0.8, edge_voice: str = "",
                 edge_rate: str = "+0%", edge_pitch: str = "+0Hz"):
        self.name = name
        self.model = model
        self.length_scale = length_scale
        self.sentence_silence = sentence_silence
        self.noise_scale = noise_scale
        self.noise_w = noise_w
        self.edge_voice = edge_voice
        self.edge_rate = edge_rate
        self.edge_pitch = edge_pitch

    @property
    def available(self) -> bool:
        return bool(self.model) and Path(self.model).exists()

    def key(self) -> str:
        return (f"{self.model}|{self.length_scale}|{self.sentence_silence}|"
               f"{self.noise_scale}|{self.noise_w}|{self.edge_voice}|"
               f"{self.edge_rate}|{self.edge_pitch}")


class TTS:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.engine = str(cfg.get("tts.engine", "piper_cli")).lower()
        self.voice_model = str(cfg.get("tts.voice_model", "") or "")
        self.length_scale = float(cfg.get("tts.length_scale", 1.0))
        # A beat between sentences. Does more for how natural a line sounds
        # than slowing the whole voice down, which just sounds sedated.
        self.sentence_silence = float(cfg.get("tts.sentence_silence", 0.0))
        # Piper's own expressiveness knobs. See Voice's docstring; these
        # defaults are exactly Piper's own stock values.
        self.noise_scale = float(cfg.get("tts.noise_scale", 0.667))
        self.noise_w = float(cfg.get("tts.noise_w", 0.8))
        # Only read when engine: edge_tts. edge_voice is a voice id from
        # `edge-tts --list-voices` (e.g. en-US-AriaNeural); rate/pitch are
        # its own nudge syntax, "+0%"/"+0Hz" for stock delivery.
        self.edge_voice = str(cfg.get("tts.edge_voice", "") or "")
        self.edge_rate = str(cfg.get("tts.edge_rate", "") or "") or "+0%"
        self.edge_pitch = str(cfg.get("tts.edge_pitch", "") or "") or "+0Hz"
        # Which piper flavour to fall back to when edge_tts is the primary
        # engine and a render fails -- offline, rate-limited, or Microsoft
        # changes the unofficial endpoint. Not an official product, so this
        # fallback is what keeps patter alive rather than just going silent.
        self.fallback_engine = str(cfg.get("tts.fallback_engine", "piper_cli")).lower()
        self.timeout = int(cfg.get("tts.timeout_seconds", 180))
        self.loudnorm = bool(cfg.get("tts.loudnorm", True))
        # Beat between one host finishing and the other starting. Real people
        # overlap; a synthetic pair needs a gap or it sounds like one rambling
        # voice that changed timbre mid-thought.
        self.turn_gap = float(cfg.get("dj.turn_gap_seconds", 0.35))
        self.out_dir = cfg.path("queue")
        self._output_flag: str | None = None
        self._length_flag: str | None = None
        self._silence_supported = True
        self._noise_supported = True
        self._ok = False
        # Circuit breaker: once the engine is clearly broken, stop paying for a
        # doomed subprocess per patter line (a whole block's worth, every hour,
        # on two cores) and just re-probe occasionally.
        self._failures = 0
        self._muted_until = 0.0

        self.voices = self._load_voices()

    # -- voices -------------------------------------------------------------

    def _load_voices(self) -> dict[str, Voice]:
        """Build a voice per configured host.

        The first host inherits the tts.* settings when it does not override
        them, so an existing single-voice setup keeps sounding exactly the same
        after a second host is added.
        """
        hosts = self.cfg.get("dj.hosts", []) or []
        voices: dict[str, Voice] = {}
        for index, host in enumerate(hosts):
            if not isinstance(host, dict):
                continue
            name = str(host.get("name") or "").strip()
            if not name:
                continue
            model = str(host.get("voice_model") or "").strip()
            if not model:
                model = self.voice_model if index == 0 else self._sibling_voice()
            # No shared catalog to guess a co-host edge voice from (unlike
            # the piper sibling file), so only the first host inherits the
            # station default -- a second host needs its own edge_voice set.
            edge_voice = str(host.get("edge_voice") or "").strip()
            if not edge_voice and index == 0:
                edge_voice = self.edge_voice
            voices[name] = Voice(
                name=name,
                model=model,
                length_scale=_number(host.get("length_scale"), self.length_scale),
                sentence_silence=_number(host.get("sentence_silence"),
                                         self.sentence_silence),
                noise_scale=_number(host.get("noise_scale"), self.noise_scale),
                noise_w=_number(host.get("noise_w"), self.noise_w),
                edge_voice=edge_voice,
                edge_rate=str(host.get("edge_rate") or "").strip() or self.edge_rate,
                edge_pitch=str(host.get("edge_pitch") or "").strip() or self.edge_pitch,
            )
        if not voices:
            voices["DJ"] = Voice("DJ", self.voice_model, self.length_scale,
                                 self.sentence_silence, self.noise_scale,
                                 self.noise_w)
        return voices

    def _sibling_voice(self) -> str:
        """Guess the co-host model from where the main voice lives.

        install.sh puts both voices in one directory, so this saves the
        listener from writing out an absolute path they did not choose.
        """
        if not self.voice_model:
            return ""
        candidate = Path(self.voice_model).parent / COHOST_VOICE_FILE
        return str(candidate)

    @property
    def hosts(self) -> list[str]:
        """Host names who can actually speak, in whatever engine is active."""
        if not self.enabled:
            return []
        if self.engine == "edge_tts":
            # Either a real edge voice, or a piper model to fall back to.
            return [name for name, voice in self.voices.items()
                   if voice.edge_voice or voice.available]
        if not self.engine.startswith("piper"):
            # espeak ignores the model entirely, so every host can speak --
            # they will just all sound the same.
            return list(self.voices)
        return [name for name, voice in self.voices.items() if voice.available]

    def voice_for(self, host: str | None) -> Voice:
        if host and host in self.voices:
            return self.voices[host]
        available = self.hosts
        if available:
            return self.voices[available[0]]
        return next(iter(self.voices.values()))

    # -- availability -------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self.engine != "none"

    def _base_command(self) -> list[str] | None:
        return self._command_for(self.engine)

    def _command_for(self, engine: str) -> list[str] | None:
        if engine == "piper_cli":
            exe = shutil.which("piper")
            return [exe] if exe else None
        if engine == "piper_python":
            return [sys.executable, "-m", "piper"]
        if engine == "espeak":
            exe = shutil.which("espeak-ng") or shutil.which("espeak")
            return [exe] if exe else None
        if engine == "edge_tts":
            exe = self._find_edge_tts()
            return [exe] if exe else None
        return None

    @staticmethod
    def _find_edge_tts() -> str | None:
        """pip installs edge-tts's console script next to whatever python
        installed it. shutil.which alone misses it whenever that venv's bin/
        is not on PATH -- true for a bare `.venv/bin/python ...` invocation,
        and for a systemd unit that runs the interpreter directly, which are
        exactly the two ways this project is normally run."""
        exe = shutil.which("edge-tts")
        if exe:
            return exe
        name = "edge-tts.exe" if sys.platform == "win32" else "edge-tts"
        candidate = Path(sys.executable).parent / name
        return str(candidate) if candidate.exists() else None

    def check(self) -> tuple[bool, str]:
        """Probe the configured engine. Returns (ok, human readable detail)."""
        if not self.enabled:
            return True, "tts disabled (engine: none) - music only, no patter"

        if self.engine == "edge_tts":
            return self._check_edge()

        base = self._base_command()
        if base is None:
            return False, f"engine {self.engine!r} not found on PATH"

        primary = self.voice_for(None)
        if self.engine.startswith("piper"):
            if not primary.model:
                return False, "tts.voice_model is not set"
            if not Path(primary.model).exists():
                return False, f"voice model missing: {primary.model}"
            config_json = Path(str(primary.model) + ".json")
            if not config_json.exists():
                return False, f"voice config missing: {config_json}"

        with tempfile.TemporaryDirectory(prefix="airadio-tts-") as tmp:
            probe = Path(tmp) / "probe.wav"
            produced = self._synthesize("Radio check, one two.", probe, primary)
            if produced is None or not produced.exists() or produced.stat().st_size < 1000:
                return False, f"{self.engine} produced no audio"

        self._ok = True
        self._failures = 0
        self._muted_until = 0.0

        speaking = self.hosts
        silent = [name for name in self.voices if name not in speaking]
        detail = f"{self.engine} ok, voices: {', '.join(speaking) or 'none'}"
        if silent:
            detail += (f" (no model on disk for {', '.join(silent)}, so they "
                       "will not speak)")
        return True, detail

    def _check_edge(self) -> tuple[bool, str]:
        """edge_tts probe: try the real thing first, then confirm the piper
        fallback actually works so a blocked/offline endpoint is a warning,
        not a silent outage."""
        primary = self.voice_for(None)
        edge_exe = self._find_edge_tts()
        ffmpeg_exe = shutil.which("ffmpeg")

        with tempfile.TemporaryDirectory(prefix="airadio-tts-") as tmp:
            edge_ok = False
            if edge_exe and ffmpeg_exe and primary.edge_voice:
                produced = self._synthesize_edge(
                    "Radio check, one two.", Path(tmp) / "probe-edge.wav", primary)
                edge_ok = bool(produced and produced.exists()
                              and produced.stat().st_size > 1000)

            if not edge_ok:
                reasons = []
                if not edge_exe:
                    reasons.append("edge-tts not installed (checked PATH and "
                                   f"next to {sys.executable})")
                if not ffmpeg_exe:
                    reasons.append("ffmpeg not found (needed to convert edge-tts's mp3)")
                if not primary.edge_voice:
                    reasons.append("tts.edge_voice is not set")
                if not reasons:
                    reasons.append("edge-tts request failed -- run 'edge-tts "
                                   f"--voice {primary.edge_voice} --text hi "
                                   "--write-media test.mp3' by hand to see the "
                                   "raw error (offline, or the unofficial "
                                   "endpoint is blocked, are the usual causes)")
                reason_text = "; ".join(reasons)

                fallback = self._synthesize_piper(
                    "Radio check, one two.", Path(tmp) / "probe-fallback.wav",
                    primary, self.fallback_engine)
                fallback_ok = bool(fallback and fallback.exists()
                                   and fallback.stat().st_size > 1000)
                if not fallback_ok:
                    return False, (f"edge-tts unavailable ({reason_text}) and "
                                   f"the piper fallback ({self.fallback_engine}) "
                                   "also failed")
                self._ok = True
                self._failures = 0
                self._muted_until = 0.0
                return True, (f"edge-tts is not working right now ({reason_text}), "
                              f"but the {self.fallback_engine} fallback is -- "
                              "patter will render with piper until edge-tts "
                              "comes back")

        self._ok = True
        self._failures = 0
        self._muted_until = 0.0
        speaking = self.hosts
        silent = [name for name in self.voices if name not in speaking]
        detail = f"edge-tts ok, voices: {', '.join(speaking) or 'none'}"
        if silent:
            detail += (f" (no edge_voice or piper fallback for "
                       f"{', '.join(silent)}, so they will not speak)")
        return True, detail

    def check_exchange(self) -> tuple[bool, str]:
        """Can two hosts actually be stitched into one file?"""
        if len(self.hosts) < 2:
            return False, "only one voice available, so no two-host segments"
        if not shutil.which("ffmpeg") and not self._can_join_natively():
            return False, "ffmpeg missing and voices differ in format"
        return True, f"two-host segments ok ({' and '.join(self.hosts[:2])})"

    def _can_join_natively(self) -> bool:
        """Do the available voices produce wavs the wave module can splice?"""
        params = set()
        for name in self.hosts[:2]:
            with tempfile.TemporaryDirectory(prefix="airadio-tts-") as tmp:
                out = Path(tmp) / "probe.wav"
                if self._synthesize("Testing.", out, self.voices[name]) is None:
                    return False
                try:
                    with wave.open(str(out), "rb") as handle:
                        params.add((handle.getnchannels(), handle.getsampwidth(),
                                    handle.getframerate()))
                except Exception:
                    return False
        return len(params) == 1

    # -- rendering ----------------------------------------------------------

    def render(self, text: str, name_hint: str = "patter",
               host: str | None = None) -> Path | None:
        """Render one patter line to a wav file. Returns None on any failure."""
        text = (text or "").strip()
        if not text or not self.enabled:
            return None
        if self._muted():
            return None

        voice = self.voice_for(host)
        target = self._target(name_hint, f"{voice.key()}|{text}")
        cached = self._cached(target)
        if cached:
            return cached

        with tempfile.TemporaryDirectory(prefix="airadio-tts-") as tmp:
            raw = Path(tmp) / "raw.wav"
            produced = self._synthesize(text, raw, voice)
            if produced is None or not produced.exists():
                self._note_failure(text)
                return None
            self._failures = 0
            if not self._postprocess([produced], target):
                try:
                    shutil.copyfile(produced, target)
                except Exception as exc:
                    log.warning("could not store patter audio: %s", exc)
                    return None

        if not target.exists() or target.stat().st_size < 1000:
            return None
        log.info("rendered patter (%d chars, %s) -> %s",
                 len(text), voice.name, target.name)
        return target

    def render_exchange(self, lines: list[dict],
                        name_hint: str = "banter") -> Path | None:
        """Render an alternating two-host conversation into one wav.

        `lines` is [{"host": "Ray", "text": "..."}, ...]. A line whose host has
        no voice on disk is spoken by whoever is available, so a missing
        co-host model costs the illusion, not the segment.
        """
        spoken = [(str(line.get("host") or ""), str(line.get("text") or "").strip())
                  for line in lines if isinstance(line, dict)]
        spoken = [(host, text) for host, text in spoken if text]
        if not spoken or not self.enabled:
            return None
        if self._muted():
            return None

        voices = [self.voice_for(host) for host, _ in spoken]
        signature = json.dumps(
            [[voice.key(), text] for voice, (_, text) in zip(voices, spoken)],
            ensure_ascii=False)
        target = self._target(name_hint, signature)
        cached = self._cached(target)
        if cached:
            return cached

        with tempfile.TemporaryDirectory(prefix="airadio-tts-") as tmp:
            parts: list[Path] = []
            for index, (voice, (_, text)) in enumerate(zip(voices, spoken)):
                piece = Path(tmp) / f"line{index:02d}.wav"
                produced = self._synthesize(text, piece, voice)
                if produced is None:
                    self._note_failure(text)
                    return None
                parts.append(produced)
            self._failures = 0

            gap = self.turn_gap if len(parts) > 1 else 0.0
            joined = (self._postprocess(parts, target, gap=gap)
                      or self._join_wavs(parts, target, gap=gap))
            if not joined:
                log.warning("could not stitch a %d line exchange; dropping it",
                            len(parts))
                return None

        if not target.exists() or target.stat().st_size < 1000:
            return None
        log.info("rendered exchange (%d lines: %s) -> %s", len(spoken),
                 ", ".join(voice.name for voice in voices), target.name)
        return target

    # -- render helpers -----------------------------------------------------

    def _target(self, name_hint: str, signature: str) -> Path:
        """Cache path. Every setting that changes the sound is in the key, so
        tuning a voice never replays a stale render."""
        digest = hashlib.sha1(
            f"{self.engine}|{self.turn_gap}|{signature}".encode("utf-8")
        ).hexdigest()[:12]
        return self.out_dir / f"{name_hint}-{digest}.wav"

    @staticmethod
    def _cached(target: Path) -> Path | None:
        if target.exists() and target.stat().st_size > 1000:
            log.debug("patter cache hit: %s", target.name)
            return target
        return None

    def _muted(self) -> bool:
        if time.time() < self._muted_until:
            log.debug("tts still muted after %d failures, skipping patter",
                      self._failures)
            return True
        return False

    def _note_failure(self, text: str) -> None:
        self._failures += 1
        if self._failures >= 3:
            self._muted_until = time.time() + 600
            log.warning("tts has failed %d times in a row; muting patter for "
                        "10 minutes. Music is unaffected. Run "
                        "'main.py doctor' to see why.", self._failures)
        else:
            log.warning("tts render failed for: %.60s", text)

    # -- engines ------------------------------------------------------------

    def _synthesize(self, text: str, out_path: Path, voice: Voice) -> Path | None:
        if self.engine == "edge_tts":
            produced = self._synthesize_edge(text, out_path, voice)
            if produced is not None:
                return produced
            log.debug("edge-tts unavailable for this line, falling back to %s",
                      self.fallback_engine)
            return self._synthesize_piper(text, out_path, voice, self.fallback_engine)

        if self.engine == "espeak":
            return self._synthesize_espeak(text, out_path)

        return self._synthesize_piper(text, out_path, voice, self.engine)

    def _synthesize_espeak(self, text: str, out_path: Path) -> Path | None:
        base = self._command_for("espeak")
        if base is None:
            return None
        command = base + ["-w", str(out_path), "-s", "150", "--stdin"]
        return self._run(command, text, out_path)

    def _synthesize_edge(self, text: str, out_path: Path, voice: Voice) -> Path | None:
        """Microsoft Edge's free neural voices, via the edge-tts package.

        Unofficial endpoint, so this is expected to fail sometimes -- offline,
        rate-limited, or Microsoft changes something. Every failure here is
        silent by design; the caller falls back to piper.
        """
        if not voice.edge_voice:
            return None
        exe = self._find_edge_tts()
        if not exe:
            log.debug("edge-tts not found on PATH or next to %s, using the "
                      "piper fallback", sys.executable)
            return None
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            # edge-tts only speaks mp3; with no way to get a wav out of that,
            # do not even try.
            log.debug("ffmpeg not found, cannot convert edge-tts's mp3 -- "
                      "using the piper fallback")
            return None

        with tempfile.TemporaryDirectory(prefix="airadio-edge-") as tmp:
            mp3_path = Path(tmp) / "raw.mp3"
            command = [exe, "--voice", voice.edge_voice]
            if voice.edge_rate and voice.edge_rate != "+0%":
                command += ["--rate", voice.edge_rate]
            if voice.edge_pitch and voice.edge_pitch != "+0Hz":
                command += ["--pitch", voice.edge_pitch]
            command += ["--text", text, "--write-media", str(mp3_path)]
            try:
                completed = subprocess.run(command, stdout=subprocess.PIPE,
                                           stderr=subprocess.PIPE, timeout=self.timeout)
            except subprocess.TimeoutExpired:
                log.warning("edge-tts timed out after %ss", self.timeout)
                return None
            except FileNotFoundError:
                return None
            except Exception as exc:
                log.warning("edge-tts subprocess error: %s", exc)
                return None
            if (completed.returncode != 0 or not mp3_path.exists()
                    or mp3_path.stat().st_size < 500):
                log.debug("edge-tts failed (%s): %s", completed.returncode,
                          completed.stderr.decode("utf-8", "replace")[:300])
                return None

            convert = [ffmpeg, "-y", "-loglevel", "error", "-i", str(mp3_path),
                      str(out_path)]
            try:
                result = subprocess.run(convert, stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE, timeout=self.timeout)
            except Exception as exc:
                log.warning("could not convert edge-tts audio: %s", exc)
                return None
            if result.returncode != 0:
                return None

        if not out_path.exists() or out_path.stat().st_size < 1000:
            return None
        return out_path

    def _synthesize_piper(self, text: str, out_path: Path, voice: Voice,
                          engine: str) -> Path | None:
        base = self._command_for(engine)
        if base is None:
            return None
        if not voice.model:
            return None

        output_flags = ([self._output_flag] if self._output_flag else list(_OUTPUT_FLAGS))
        if abs(voice.length_scale - 1.0) <= 0.001:
            # Default speed: the flag is never added, so trying its spellings
            # would just run the same command three times.
            length_flags: list[str | None] = [None]
        elif self._length_flag:
            length_flags = [self._length_flag]
        else:
            length_flags = list(_LENGTH_FLAGS) + [None]

        for output_flag in output_flags:
            for length_flag in length_flags:
                command = base + ["-m", voice.model, output_flag, str(out_path)]
                if length_flag and abs(voice.length_scale - 1.0) > 0.001:
                    command += [length_flag, str(voice.length_scale)]
                    underscore = "_" in length_flag
                elif self._length_flag:
                    underscore = "_" in self._length_flag
                else:
                    # Not known yet -- piper's own CLI uses underscores, so
                    # guess that until a probe below proves otherwise.
                    underscore = True

                # Match whatever spelling style length_scale is using (or
                # would use), so one working style is used consistently
                # rather than mixing --length_scale with --noise-scale.
                silence_flag = "--sentence_silence" if underscore else "--sentence-silence"
                noise_scale_flag = "--noise_scale" if underscore else "--noise-scale"
                noise_w_flag = "--noise_w" if underscore else "--noise-w"

                pause = voice.sentence_silence if self._silence_supported else 0.0
                if pause > 0:
                    command += [silence_flag, str(pause)]

                extras: list[str] = []
                if self._noise_supported:
                    if abs(voice.noise_scale - 0.667) > 0.001:
                        extras += [noise_scale_flag, str(voice.noise_scale)]
                    if abs(voice.noise_w - 0.8) > 0.001:
                        extras += [noise_w_flag, str(voice.noise_w)]

                result = self._run(command + extras, text, out_path)
                if result is not None:
                    self._output_flag = output_flag
                    self._length_flag = length_flag
                    return result

                # Piper rejected something. Peel off the newest, least-tested
                # addition first (noise, then the pause) rather than giving
                # up on the whole line -- a flatter-sounding voice beats no
                # patter at all.
                if extras:
                    result = self._run(command, text, out_path)
                    if result is not None:
                        self._output_flag = output_flag
                        self._length_flag = length_flag
                        self._noise_supported = False
                        log.info("piper rejected %s/%s, continuing without "
                                 "the expressiveness tuning", noise_scale_flag,
                                 noise_w_flag)
                        return result

                if pause > 0:
                    retry = [part for part in command
                             if part not in (silence_flag, str(pause))]
                    result = self._run(retry, text, out_path)
                    if result is not None:
                        self._output_flag = output_flag
                        self._length_flag = length_flag
                        self._silence_supported = False
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

    def _postprocess(self, sources: list[Path], target: Path,
                     gap: float = 0.0) -> bool:
        """Join, normalise loudness and resample so patter sits level with music.

        One ffmpeg pass does the lot. Two voices can differ in sample rate, so
        every input is resampled before the concat rather than after.
        """
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg or not sources:
            return False

        command = [ffmpeg, "-y", "-loglevel", "error"]
        for source in sources:
            command += ["-i", str(source)]

        chain = []
        labels = []
        for index in range(len(sources)):
            steps = ["aresample=44100",
                     "aformat=sample_fmts=s16:channel_layouts=stereo"]
            # Pad every turn but the last, so the gap lands between speakers
            # and not as trailing silence before the next song.
            if gap > 0 and index < len(sources) - 1:
                steps.append(f"apad=pad_dur={gap}")
            chain.append(f"[{index}:a]" + ",".join(steps) + f"[a{index}]")
            labels.append(f"[a{index}]")

        joined = "[out]"
        if len(sources) > 1:
            chain.append("".join(labels) + f"concat=n={len(sources)}:v=0:a=1[cat]")
            tail = "[cat]"
        else:
            tail = labels[0]
        filters = "loudnorm=I=-16:TP=-1.5:LRA=11" if self.loudnorm else "anull"
        chain.append(f"{tail}{filters}{joined}")

        command += ["-filter_complex", ";".join(chain), "-map", joined,
                    "-ar", "44100", "-ac", "2", str(target)]
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

    def _join_wavs(self, sources: list[Path], target: Path,
                   gap: float = 0.0) -> bool:
        """Splice wavs with the stdlib, for boxes without ffmpeg.

        Only works when every part shares a format, which is true whenever both
        voices came out of the same Piper build at the same sample rate.
        """
        try:
            with wave.open(str(sources[0]), "rb") as first:
                params = first.getparams()
                frames = [first.readframes(first.getnframes())]
            silence = b"\x00" * int(params.framerate * max(0.0, gap)
                                    * params.sampwidth * params.nchannels)
            for source in sources[1:]:
                with wave.open(str(source), "rb") as handle:
                    if handle.getparams()[:3] != params[:3]:
                        log.info("voices differ in wav format, need ffmpeg to join")
                        return False
                    if silence:
                        frames.append(silence)
                    frames.append(handle.readframes(handle.getnframes()))
            with wave.open(str(target), "wb") as out:
                out.setparams(params)
                for chunk in frames:
                    out.writeframes(chunk)
        except Exception as exc:
            log.warning("could not splice patter audio: %s", exc)
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
