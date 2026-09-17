"""Weather for the DJ, from Open-Meteo.

Open-Meteo is free and needs no API key and no account, which is the only
reason weather is in this project at all -- the station's hard rule is that it
never generates a bill.

Nothing here is allowed to be fatal.  If the network is down, the cache is
stale or the response is shaped oddly, `current()` returns None and the DJ
simply does not mention the weather that hour.
"""

from __future__ import annotations

import logging
import time

import requests

from .config import Config

log = logging.getLogger("weather")

ENDPOINT = "https://api.open-meteo.com/v1/forecast"

# WMO weather interpretation codes, written the way a person says them out
# loud rather than the way a meteorologist writes them down.
CONDITIONS = {
    0: "clear",
    1: "mostly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "foggy",
    48: "freezing fog",
    51: "light drizzle",
    53: "drizzle",
    55: "heavy drizzle",
    56: "freezing drizzle",
    57: "freezing drizzle",
    61: "light rain",
    63: "rain",
    65: "heavy rain",
    66: "freezing rain",
    67: "freezing rain",
    71: "light snow",
    73: "snow",
    75: "heavy snow",
    77: "snow grains",
    80: "light showers",
    81: "showers",
    82: "heavy showers",
    85: "snow showers",
    86: "heavy snow showers",
    95: "thunderstorms",
    96: "thunderstorms with hail",
    99: "thunderstorms with hail",
}


class Weather:
    """Cached current conditions and today's outlook for one fixed place."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.enabled = bool(cfg.get("weather.enabled", True))
        self.place = str(cfg.get("weather.place_name", "Eindhoven"))
        self.latitude = float(cfg.get("weather.latitude", 51.4416))
        self.longitude = float(cfg.get("weather.longitude", 5.4697))
        self.timezone = str(cfg.get("weather.timezone", "Europe/Amsterdam"))
        self.cache_minutes = float(cfg.get("weather.cache_minutes", 30))
        self.timeout = float(cfg.get("weather.timeout_seconds", 10))
        self._session = requests.Session()
        self._cache: dict | None = None
        self._cached_at = 0.0
        # Repeated failures usually mean the whole network is down, which the
        # rest of the loop is already complaining about. Back off rather than
        # spending ten seconds of every planning pass on a doomed request.
        self._failures = 0
        self._quiet_until = 0.0

    # -- fetching -----------------------------------------------------------

    def current(self, force: bool = False) -> dict | None:
        """Conditions now plus today's range. None if never fetched."""
        if not self.enabled:
            return None

        age = time.time() - self._cached_at
        if self._cache and not force and age < self.cache_minutes * 60:
            return self._cache
        if not force and time.time() < self._quiet_until:
            # Serve a stale reading rather than nothing: an hour-old
            # temperature is still roughly true, and better than silence.
            return self._cache

        data = self._fetch()
        if data is None:
            self._failures += 1
            self._quiet_until = time.time() + min(1800, 120 * self._failures)
            return self._cache
        self._failures = 0
        self._quiet_until = 0.0
        self._cache = data
        self._cached_at = time.time()
        return data

    def _fetch(self) -> dict | None:
        params = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "timezone": self.timezone,
            "forecast_days": 1,
            "current": "temperature_2m,apparent_temperature,weather_code,"
                       "wind_speed_10m,is_day",
            "daily": "weather_code,temperature_2m_max,temperature_2m_min,"
                     "precipitation_probability_max",
        }
        try:
            response = self._session.get(ENDPOINT, params=params,
                                         timeout=self.timeout)
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            log.info("weather unavailable (%s); the DJ will skip it", exc)
            return None

        try:
            now = payload.get("current") or {}
            daily = payload.get("daily") or {}

            def first(key):
                values = daily.get(key) or []
                return values[0] if values else None

            reading = {
                "place": self.place,
                "temp_c": _round(now.get("temperature_2m")),
                "feels_c": _round(now.get("apparent_temperature")),
                "condition": CONDITIONS.get(_int(now.get("weather_code")), ""),
                "wind_kmh": _round(now.get("wind_speed_10m")),
                "is_day": bool(now.get("is_day", 1)),
                "high_c": _round(first("temperature_2m_max")),
                "low_c": _round(first("temperature_2m_min")),
                "today": CONDITIONS.get(_int(first("weather_code")), ""),
                "rain_chance": _int(first("precipitation_probability_max")),
                "fetched_at": time.time(),
            }
        except Exception as exc:
            log.info("weather response was shaped oddly (%s); skipping", exc)
            return None

        if reading["temp_c"] is None:
            return None
        log.info("weather for %s: %s, %s degrees (high %s, low %s)",
                 reading["place"], reading["condition"] or "unknown",
                 reading["temp_c"], reading["high_c"], reading["low_c"])
        return reading

    # -- prompt material ----------------------------------------------------

    def briefing(self, include_outlook: bool = True) -> str:
        """Plain facts for the planner prompt. Empty string if unavailable.

        Deliberately facts and not prose: the DJ writes the line, this only
        says what is true. Numbers are digits here because the prompt tells the
        model to speak them as words.
        """
        reading = self.current()
        if not reading:
            return ""

        parts = [f"Weather in {reading['place']} right now: "
                 f"{reading['temp_c']} degrees Celsius"]
        feels = reading["feels_c"]
        if feels is not None and abs(feels - reading["temp_c"]) >= 3:
            parts.append(f", feels like {feels}")
        if reading["condition"]:
            parts.append(f", {reading['condition']}")
        if reading["wind_kmh"] and reading["wind_kmh"] >= 25:
            parts.append(f", windy at {reading['wind_kmh']} kilometres an hour")
        line = "".join(parts) + "."

        if not include_outlook:
            return line

        outlook = []
        if reading["high_c"] is not None and reading["low_c"] is not None:
            outlook.append(f"today a high of {reading['high_c']} and a low of "
                           f"{reading['low_c']}")
        if reading["today"] and reading["today"] != reading["condition"]:
            outlook.append(f"mostly {reading['today']}")
        if reading["rain_chance"] is not None and reading["rain_chance"] >= 30:
            outlook.append(f"{reading['rain_chance']} percent chance of rain")
        if outlook:
            line += " Later " + ", ".join(outlook) + "."
        return line


def _round(value) -> float | None:
    try:
        return round(float(value))
    except (TypeError, ValueError):
        return None


def _int(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
