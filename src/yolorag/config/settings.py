from __future__ import annotations

import os

from dotenv import load_dotenv


load_dotenv()


def getenv(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value
