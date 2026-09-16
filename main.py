#!/usr/bin/env python3
"""Orchestrator entry point.

    python main.py run      # the brain loop (planner, TTS, downloads, feeder)
    python main.py bot      # the Discord bot
    python main.py doctor   # check everything is wired up

Same thing as the installed `airadio` console script.
"""

import sys

from airadio.cli import main

if __name__ == "__main__":
    sys.exit(main())
