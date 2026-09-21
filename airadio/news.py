"""News for the DJ, from RSS feeds.

RSS is free and needs no API key and no account -- the same reason weather
uses Open-Meteo: the station's hard rule is that it never generates a bill.

Nothing here is allowed to be fatal. If a feed is unreachable, slow, or
shaped oddly, headlines() returns whatever it already has cached (possibly
nothing), and the DJ simply skips the news that hour.
"""

from __future__ import annotations

import logging
import time
from xml.etree import ElementTree

import requests

from .config import Config

log = logging.getLogger("news")

# BBC's RSS needs no key and no account. Political and science, since
# that is what was asked for -- swap or add feeds in config freely, any
# well-formed RSS 2.0 <item><title> feed works.
DEFAULT_FEEDS = [
    "http://feeds.bbci.co.uk/news/politics/rss.xml",
    "http://feeds.bbci.co.uk/news/science_and_environment/rss.xml",
]

_HEADERS = {"User-Agent": "airadio/1.0 (+private single-listener station)"}


class News:
    """Cached recent headlines from a handful of RSS feeds."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.enabled = bool(cfg.get("news.enabled", False))
        self.feeds = [str(url) for url in
                     (cfg.get("news.feeds") or DEFAULT_FEEDS) if url]
        self.per_feed = int(cfg.get("news.headlines_per_feed", 3))
        self.cache_minutes = float(cfg.get("news.cache_minutes", 30))
        self.timeout = float(cfg.get("news.timeout_seconds", 10))
        self._session = requests.Session()
        self._cache: list[dict] = []
        self._cached_at = 0.0
        # Repeated failures usually mean one feed's own outage, not this
        # station's fault. Back off rather than spending several seconds of
        # every planning pass on a doomed request.
        self._failures = 0
        self._quiet_until = 0.0

    # -- fetching -------------------------------------------------------

    def headlines(self, force: bool = False) -> list[dict]:
        """Recent headlines across every configured feed. [] if unavailable."""
        if not self.enabled or not self.feeds:
            return []

        age = time.time() - self._cached_at
        if self._cache and not force and age < self.cache_minutes * 60:
            return self._cache
        if not force and time.time() < self._quiet_until:
            # Serve what is cached rather than nothing: a half-hour-old
            # headline is still true, and better than silence.
            return self._cache

        fetched = self._fetch_all()
        if not fetched:
            self._failures += 1
            self._quiet_until = time.time() + min(1800, 120 * self._failures)
            return self._cache
        self._failures = 0
        self._quiet_until = 0.0
        self._cache = fetched
        self._cached_at = time.time()
        return fetched

    def _fetch_all(self) -> list[dict]:
        items: list[dict] = []
        for url in self.feeds:
            items.extend(self._fetch_one(url))
        return items

    def _fetch_one(self, url: str) -> list[dict]:
        try:
            response = self._session.get(url, timeout=self.timeout,
                                         headers=_HEADERS)
            response.raise_for_status()
            root = ElementTree.fromstring(response.content)
        except Exception as exc:
            log.info("news feed unavailable (%s): %s; skipping it", url, exc)
            return []

        items: list[dict] = []
        for item in root.iter("item"):
            title = _clean(item.findtext("title") or "")
            if not title:
                continue
            items.append({"title": title, "source": url})
            if len(items) >= self.per_feed:
                break
        if not items:
            log.info("news feed had no usable items: %s", url)
        return items

    # -- prompt material --------------------------------------------------

    def briefing(self) -> str:
        """Plain facts for the planner prompt. Empty string if unavailable.

        Deliberately a bare list of real headlines and nothing else, the same
        discipline as weather.briefing(): the model picks and phrases, this
        only says what is actually true right now. The "use nothing else"
        instruction lives in the prompt template, not here, same as weather.
        """
        headlines = self.headlines()
        if not headlines:
            return ""
        return "\n".join(f"- {item['title']}" for item in headlines)


def _clean(title: str) -> str:
    """RSS titles are written for skimming a web page, not for speech."""
    title = " ".join(title.split())
    # "Some story - BBC News" -- the outlet suffix reads badly out loud and
    # is not part of the actual headline.
    for suffix in (" - BBC News", " | BBC News", " - BBC Sport"):
        if title.endswith(suffix):
            title = title[:-len(suffix)]
            break
    return title.strip()
