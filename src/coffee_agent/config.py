"""Where the demo database lives, and lazy construction of the shared toolbox."""

from __future__ import annotations

import os
from pathlib import Path

from .db import ReadOnlyDatabase, build_database
from .retrieval import KnowledgeBase
from .tools import Toolbox


def default_db_path() -> Path:
    if os.environ.get("COFFEE_DB"):
        return Path(os.environ["COFFEE_DB"])
    base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(base) / "coffee-data-agent" / "marketplace.db"


def open_toolbox(db_path: Path | None = None, *, create: bool = True) -> Toolbox:
    path = db_path or default_db_path()
    if create and not path.exists():
        build_database(path)
    return Toolbox(ReadOnlyDatabase(path), KnowledgeBase.from_directory())
