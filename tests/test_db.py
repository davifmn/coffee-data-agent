import hashlib
import sqlite3

import pytest

from coffee_agent.db import (
    FREE_SHIPPING_THRESHOLD_BRL,
    QueryError,
    ReadOnlyDatabase,
    build_database,
)


def _fingerprint(path) -> str:
    conn = sqlite3.connect(path)
    try:
        digest = hashlib.sha256()
        for line in conn.iterdump():
            digest.update(line.encode())
        return digest.hexdigest()
    finally:
        conn.close()


def test_build_is_deterministic(tmp_path, db_path):
    again = build_database(tmp_path / "again.db")
    assert _fingerprint(again) == _fingerprint(db_path)


def test_different_seed_gives_different_data(tmp_path, db_path):
    other = build_database(tmp_path / "other.db", seed=7)
    assert _fingerprint(other) != _fingerprint(db_path)


def test_data_looks_like_a_marketplace(db):
    counts = dict(
        db.run(
            """
            SELECT 'suppliers', COUNT(*) FROM suppliers UNION ALL
            SELECT 'orders', COUNT(*) FROM orders UNION ALL
            SELECT 'reviews', COUNT(*) FROM reviews
            """
        ).rows
    )
    assert counts["suppliers"] == 12
    assert counts["orders"] == 3600
    assert counts["reviews"] > 1000
    (min_score,) = db.run("SELECT MIN(sca_score) FROM coffees").rows[0]
    assert min_score >= 80  # every listed coffee is specialty grade


def test_free_shipping_rule_is_consistent(db):
    (violations,) = db.run(
        """
        SELECT COUNT(*) FROM (
            SELECT o.id, o.shipping_brl, SUM(i.quantity_kg * i.unit_price_brl) AS subtotal
            FROM orders o JOIN order_items i ON i.order_id = o.id GROUP BY o.id
        ) WHERE (subtotal >= ? AND shipping_brl > 0) OR (subtotal < ? AND shipping_brl = 0)
        """.replace("?", str(FREE_SHIPPING_THRESHOLD_BRL))
    ).rows[0]
    assert violations == 0


def test_some_suppliers_are_below_the_rating_policy(db):
    rows = db.run(
        """
        SELECT s.id, AVG(r.rating) AS avg_rating
        FROM suppliers s JOIN coffees c ON c.supplier_id = s.id JOIN reviews r ON r.coffee_id = c.id
        GROUP BY s.id HAVING avg_rating < 4.0
        """
    ).rows
    assert {row[0] for row in rows} == {4, 9}


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM orders",
        "UPDATE coffees SET price_brl_per_kg = 0",
        "INSERT INTO suppliers VALUES (99, 'x', 'y', 'MG', 0, '2024-01-01')",
        "DROP TABLE reviews",
        "CREATE TABLE t (x)",
        "PRAGMA journal_mode = WAL",
        "ATTACH DATABASE ':memory:' AS other",
    ],
)
def test_non_select_statements_are_rejected(db, sql):
    with pytest.raises(QueryError, match="only SELECT"):
        db.run(sql)


@pytest.mark.parametrize(
    "sql",
    [
        # Writes smuggled behind a SELECT-looking start must still be stopped.
        "WITH x AS (SELECT 1) DELETE FROM orders",
        "SELECT 1; DROP TABLE orders",
        "SELECT load_extension('/tmp/evil')",
        "SELECT * FROM pragma_table_info('orders')",
    ],
)
def test_disguised_writes_and_escapes_are_blocked(db, sql):
    with pytest.raises(QueryError):
        db.run(sql)
    # The data is untouched.
    assert db.run("SELECT COUNT(*) FROM orders").rows[0][0] == 3600


def test_file_is_opened_read_only_even_without_the_authorizer(db_path):
    database = ReadOnlyDatabase(db_path)
    database._conn.set_authorizer(None)
    with pytest.raises(sqlite3.OperationalError):
        database._conn.execute("DELETE FROM orders")
    database.close()


def test_long_queries_time_out(db_path):
    database = ReadOnlyDatabase(db_path, timeout_s=0.2)
    endless = "WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM n) SELECT MAX(x) FROM n"
    with pytest.raises(QueryError, match="time limit"):
        database.run(endless)
    database.close()


def test_rows_are_capped(db):
    result = db.run("SELECT id FROM orders ORDER BY id", max_rows=5)
    assert [r[0] for r in result.rows] == [1, 2, 3, 4, 5]
    assert result.truncated


def test_sql_errors_are_reported(db):
    with pytest.raises(QueryError, match="no such column"):
        db.run("SELECT nope FROM orders")


def test_schema_description_mentions_every_table(db):
    text = db.describe_schema()
    for table in ["suppliers", "coffees", "customers", "orders", "order_items", "reviews"]:
        assert f"CREATE TABLE {table}" in text
    assert "orders=3600" in text
