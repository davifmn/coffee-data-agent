from pathlib import Path

import pytest

from coffee_agent.db import ReadOnlyDatabase, build_database
from coffee_agent.retrieval import KnowledgeBase
from coffee_agent.tools import Toolbox


@pytest.fixture(scope="session")
def db_path(tmp_path_factory) -> Path:
    return build_database(tmp_path_factory.mktemp("db") / "marketplace.db")


@pytest.fixture
def db(db_path):
    database = ReadOnlyDatabase(db_path)
    yield database
    database.close()


@pytest.fixture(scope="session")
def kb():
    return KnowledgeBase.from_directory()


@pytest.fixture
def toolbox(db, kb):
    return Toolbox(db, kb)
