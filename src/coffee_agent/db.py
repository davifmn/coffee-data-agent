"""Synthetic marketplace database and a locked-down, read-only query runner.

The data describes "Grão Fino", a fictional Brazilian specialty-coffee marketplace:
partner farms (suppliers) list coffees, customers place orders through the app,
the website or Instagram, and leave reviews. Everything is generated from a fixed
seed, so every run (and every test) sees exactly the same numbers.

Model-written SQL is untrusted input, so ``ReadOnlyDatabase`` stacks several
independent defences instead of trusting any single one:

1. the file is opened with ``mode=ro`` (the OS-level handle cannot write);
2. ``PRAGMA query_only`` makes SQLite refuse any write;
3. an authorizer callback allows only reads (SELECT, column reads, functions,
   recursive CTEs) and denies ATTACH, PRAGMA, writes and ``load_extension``;
4. ``sqlite3`` refuses to run more than one statement per call;
5. a progress handler aborts queries that run longer than the time budget,
   and results are capped at ``max_rows``.
"""

from __future__ import annotations

import random
import sqlite3
import time
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

SCHEMA = """
CREATE TABLE suppliers (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    region TEXT NOT NULL,          -- Brazilian coffee region, e.g. 'Cerrado Mineiro'
    state TEXT NOT NULL,           -- two-letter state code, e.g. 'MG'
    certified_organic INTEGER NOT NULL CHECK (certified_organic IN (0, 1)),
    joined_on TEXT NOT NULL        -- ISO date the farm joined the marketplace
);

CREATE TABLE coffees (
    id INTEGER PRIMARY KEY,
    supplier_id INTEGER NOT NULL REFERENCES suppliers(id),
    name TEXT NOT NULL,
    variety TEXT NOT NULL,         -- e.g. 'Bourbon Amarelo', 'Catuaí Vermelho', 'Geisha'
    process TEXT NOT NULL CHECK (process IN
        ('natural', 'pulped natural', 'washed', 'honey', 'anaerobic')),
    altitude_m INTEGER NOT NULL,
    sca_score REAL NOT NULL,       -- SCA cupping score, 80-100 for specialty coffee
    roast_level TEXT NOT NULL CHECK (roast_level IN ('light', 'medium', 'medium-dark')),
    price_brl_per_kg REAL NOT NULL,
    active INTEGER NOT NULL CHECK (active IN (0, 1))
);

CREATE TABLE customers (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    city TEXT NOT NULL,
    state TEXT NOT NULL,
    signed_up_on TEXT NOT NULL
);

CREATE TABLE orders (
    id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(id),
    ordered_on TEXT NOT NULL,      -- ISO date
    channel TEXT NOT NULL CHECK (channel IN ('app', 'web', 'instagram')),
    status TEXT NOT NULL CHECK (status IN ('delivered', 'shipped', 'cancelled')),
    shipping_brl REAL NOT NULL     -- 0 when the order qualified for free shipping
);

CREATE TABLE order_items (
    id INTEGER PRIMARY KEY,
    order_id INTEGER NOT NULL REFERENCES orders(id),
    coffee_id INTEGER NOT NULL REFERENCES coffees(id),
    quantity_kg REAL NOT NULL,     -- bags are sold in 0.25 kg steps
    unit_price_brl REAL NOT NULL   -- price per kg at the time of the order
);

CREATE TABLE reviews (
    id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(id),
    coffee_id INTEGER NOT NULL REFERENCES coffees(id),
    rating INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
    reviewed_on TEXT NOT NULL
);

CREATE INDEX idx_orders_date ON orders(ordered_on);
CREATE INDEX idx_items_order ON order_items(order_id);
CREATE INDEX idx_items_coffee ON order_items(coffee_id);
CREATE INDEX idx_reviews_coffee ON reviews(coffee_id);
"""

FREE_SHIPPING_THRESHOLD_BRL = 150.0  # also stated in knowledge/marketplace_policies.md

REGIONS = [
    # (region, state, min altitude, max altitude)
    ("Cerrado Mineiro", "MG", 850, 1250),
    ("Sul de Minas", "MG", 850, 1350),
    ("Mantiqueira de Minas", "MG", 1000, 1600),
    ("Matas de Minas", "MG", 700, 1200),
    ("Alta Mogiana", "SP", 900, 1300),
    ("Chapada Diamantina", "BA", 1000, 1500),
    ("Caparaó", "MG", 900, 1500),
    ("Montanhas do Espírito Santo", "ES", 800, 1300),
    ("Norte Pioneiro do Paraná", "PR", 600, 1000),
]

FARM_PREFIXES = ["Fazenda", "Sítio", "Fazenda", "Estância"]
FARM_NAMES = [
    "Pedra Alta",
    "Boa Vista do Alto",
    "Três Córregos",
    "Serra Clara",
    "Vale do Sabiá",
    "Recanto das Araras",
    "Morro Verde",
    "Santa Luzia do Mato",
    "Água Limpa",
    "Colina Dourada",
    "Cachoeira Fina",
    "Alto Ipê",
]
VARIETIES = [
    "Bourbon Amarelo",
    "Bourbon Vermelho",
    "Catuaí Amarelo",
    "Catuaí Vermelho",
    "Mundo Novo",
    "Arara",
    "Topázio",
    "Catucaí",
    "Acaiá",
    "Yellow Icatu",
    "Geisha",
]
PROCESSES = ["natural", "pulped natural", "washed", "honey", "anaerobic"]
PROCESS_WEIGHTS = [40, 25, 12, 13, 10]
FIRST_NAMES = [
    "Ana",
    "Bruno",
    "Camila",
    "Daniel",
    "Eduarda",
    "Felipe",
    "Gabriela",
    "Henrique",
    "Isabela",
    "João",
    "Larissa",
    "Marcos",
    "Natália",
    "Otávio",
    "Paula",
    "Rafael",
    "Sofia",
    "Thiago",
    "Vitória",
    "William",
]
LAST_NAMES = [
    "Almeida",
    "Barbosa",
    "Carvalho",
    "Dias",
    "Fernandes",
    "Gomes",
    "Lima",
    "Martins",
    "Oliveira",
    "Pereira",
    "Ribeiro",
    "Rocha",
    "Santos",
    "Silva",
    "Souza",
    "Teixeira",
]
CITIES = [
    ("Belo Horizonte", "MG", 30),
    ("São Paulo", "SP", 25),
    ("Rio de Janeiro", "RJ", 12),
    ("Curitiba", "PR", 7),
    ("Porto Alegre", "RS", 5),
    ("Brasília", "DF", 6),
    ("Salvador", "BA", 4),
    ("Florianópolis", "SC", 4),
    ("Recife", "PE", 3),
    ("Juiz de Fora", "MG", 4),
]

START = date(2024, 1, 1)
END = date(2026, 10, 31)


def _random_date(rng: random.Random, start: date, end: date) -> date:
    return start + timedelta(days=rng.randrange((end - start).days + 1))


def _seasonal_order_date(rng: random.Random) -> date:
    """Order dates with growth over time and peaks in May (Mother's Day) and December."""
    while True:
        day = _random_date(rng, START, END)
        growth = 0.55 + 0.45 * (day - START).days / (END - START).days
        season = {5: 1.35, 11: 1.2, 12: 1.6}.get(day.month, 1.0)
        if rng.random() < growth * season / 1.6:
            return day


def build_database(path: str | Path, *, seed: int = 42) -> Path:
    """Create (or replace) the demo database at ``path`` and return its path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".building")
    tmp.unlink(missing_ok=True)

    rng = random.Random(seed)
    conn = sqlite3.connect(tmp)
    try:
        conn.executescript(SCHEMA)
        _populate(conn, rng)
        conn.commit()
    finally:
        conn.close()
    tmp.replace(path)
    return path


def _populate(conn: sqlite3.Connection, rng: random.Random) -> None:
    # Suppliers: one farm per name, spread over the regions.
    suppliers = []
    for i, farm in enumerate(FARM_NAMES, start=1):
        region, state, lo, hi = REGIONS[(i - 1) % len(REGIONS)]
        joined = _random_date(rng, date(2023, 6, 1), date(2025, 6, 30))
        name = f"{rng.choice(FARM_PREFIXES)} {farm}"
        suppliers.append((i, name, region, state, int(rng.random() < 0.3), joined.isoformat()))
    conn.executemany("INSERT INTO suppliers VALUES (?, ?, ?, ?, ?, ?)", suppliers)

    # Two suppliers have consistently weaker coffees; the review generator uses this so
    # that questions about the partner rating policy have a non-trivial answer.
    weak_suppliers = {4, 9}

    coffees = []
    coffee_quality: dict[int, float] = {}
    coffee_id = 0
    for sup_id, _, region, _, _, _ in suppliers:
        _, _, lo, hi = next(r for r in REGIONS if r[0] == region)
        for _ in range(rng.randint(3, 6)):
            coffee_id += 1
            variety = rng.choice(VARIETIES)
            process = rng.choices(PROCESSES, PROCESS_WEIGHTS)[0]
            altitude = rng.randrange(lo, hi, 10)
            score = 80 + rng.random() * 4 + (altitude - 600) / 1000 * 3
            score += {"anaerobic": 1.8, "washed": 0.6, "honey": 0.5}.get(process, 0)
            score += 2.5 if variety == "Geisha" else 0
            score = round(min(score, 93.5), 2)
            roast = rng.choices(["light", "medium", "medium-dark"], [35, 50, 15])[0]
            price = round(95 + (score - 80) ** 1.6 * 9 + rng.uniform(-8, 8), 2)
            # Listing names carry the farm, like a real marketplace ("Geisha Natural · Pedra Alta").
            farm = next(sup[1] for sup in suppliers if sup[0] == sup_id).split(" ", 1)[1]
            name = f"{variety} {process.title()} · {farm}"
            lot = 2
            while any(c[2] == name for c in coffees):
                name = f"{variety} {process.title()} · {farm} (lot {lot})"
                lot += 1
            active = int(rng.random() > 0.08)
            coffees.append(
                (coffee_id, sup_id, name, variety, process, altitude, score, roast, price, active)
            )
            quality = 3.9 + (score - 80) / 10 * 1.4
            coffee_quality[coffee_id] = quality - (0.9 if sup_id in weak_suppliers else 0)
    conn.executemany("INSERT INTO coffees VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", coffees)

    customers = []
    for cust_id in range(1, 451):
        city, state, _ = rng.choices(CITIES, [c[2] for c in CITIES])[0]
        name = f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
        signed = _random_date(rng, date(2023, 10, 1), date(2026, 9, 30))
        customers.append((cust_id, name, city, state, signed.isoformat()))
    conn.executemany("INSERT INTO customers VALUES (?, ?, ?, ?, ?)", customers)

    prices = {c[0]: c[8] for c in coffees}
    active_ids = [c[0] for c in coffees if c[9]]
    orders, items, reviews = [], [], []
    item_id = review_id = 0
    for order_id in range(1, 3601):
        day = _seasonal_order_date(rng)
        # Instagram grows fast in 2026, after the marketplace started paid campaigns there.
        insta = 0.12 if day.year < 2026 else 0.30
        channel = rng.choices(["app", "web", "instagram"], [0.55, 0.45 - insta, insta])[0]
        status = rng.choices(["delivered", "shipped", "cancelled"], [86, 6, 8])[0]
        if day > END - timedelta(days=10) and status == "delivered":
            status = "shipped"
        customer = rng.randint(1, len(customers))

        subtotal = 0.0
        for coffee in rng.sample(active_ids, rng.choices([1, 2, 3, 4], [45, 32, 15, 8])[0]):
            item_id += 1
            qty = rng.choices([0.25, 0.5, 1.0, 2.0], [40, 35, 20, 5])[0]
            # Prices drift up ~6% a year.
            unit = round(prices[coffee] * (1 + 0.06 * (day.year - 2024)), 2)
            subtotal += qty * unit
            items.append((item_id, order_id, coffee, qty, unit))
            if status == "delivered" and rng.random() < 0.45:
                review_id += 1
                mean = coffee_quality[coffee]
                rating = max(1, min(5, round(rng.gauss(mean, 0.7))))
                reviewed = day + timedelta(days=rng.randint(5, 30))
                reviews.append((review_id, customer, coffee, rating, reviewed.isoformat()))
        shipping = (
            0.0 if subtotal >= FREE_SHIPPING_THRESHOLD_BRL else rng.choice([18.9, 24.9, 29.9])
        )
        orders.append((order_id, customer, day.isoformat(), channel, status, shipping))

    conn.executemany("INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?)", orders)
    conn.executemany("INSERT INTO order_items VALUES (?, ?, ?, ?, ?)", items)
    conn.executemany("INSERT INTO reviews VALUES (?, ?, ?, ?, ?)", reviews)


class QueryError(Exception):
    """The query was rejected or failed. The message is safe to show to the model."""


@dataclass(frozen=True)
class QueryResult:
    columns: list[str]
    rows: list[tuple]
    truncated: bool
    elapsed_ms: float


_ALLOWED_ACTIONS = {
    sqlite3.SQLITE_SELECT,
    sqlite3.SQLITE_READ,
    sqlite3.SQLITE_FUNCTION,
    sqlite3.SQLITE_RECURSIVE,
}
_DENIED_FUNCTIONS = {"load_extension", "readfile", "writefile", "edit", "fts3_tokenizer"}


def _authorizer(action, arg1, arg2, db_name, trigger):
    if action not in _ALLOWED_ACTIONS:
        return sqlite3.SQLITE_DENY
    if action == sqlite3.SQLITE_FUNCTION and (arg2 or "").lower() in _DENIED_FUNCTIONS:
        return sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_OK


class ReadOnlyDatabase:
    def __init__(self, path: str | Path, *, timeout_s: float = 3.0) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"database not found: {self.path} (run `coffee-agent init-db`)")
        self.timeout_s = timeout_s
        self._conn = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True, check_same_thread=False)
        self._conn.execute("PRAGMA query_only = ON")
        self._conn.set_authorizer(_authorizer)
        self._schema_text: str | None = None

    def close(self) -> None:
        self._conn.close()

    def run(self, sql: str, *, max_rows: int = 200) -> QueryResult:
        sql = sql.strip().rstrip(";").strip()
        if not sql:
            raise QueryError("empty query")
        first_word = sql.split(None, 1)[0].lower()
        if first_word not in {"select", "with"}:
            raise QueryError("only SELECT queries (optionally starting with WITH) are allowed")

        deadline = time.monotonic() + self.timeout_s
        self._conn.set_progress_handler(lambda: int(time.monotonic() > deadline), 10_000)
        started = time.perf_counter()
        try:
            cursor = self._conn.execute(sql)
            rows = cursor.fetchmany(max_rows + 1)
        except sqlite3.DatabaseError as exc:
            message = str(exc)
            if "interrupted" in message:
                message = f"query exceeded the {self.timeout_s:g}s time limit"
            elif "not authorized" in message or "prohibited" in message:
                message = "query uses an operation that is not allowed (read-only access)"
            raise QueryError(message) from None
        except sqlite3.ProgrammingError as exc:
            raise QueryError(str(exc)) from None
        finally:
            self._conn.set_progress_handler(None, 0)
        elapsed = (time.perf_counter() - started) * 1000
        columns = [d[0] for d in cursor.description or []]
        truncated = len(rows) > max_rows
        return QueryResult(columns, rows[:max_rows], truncated, round(elapsed, 2))

    def describe_schema(self) -> str:
        """Schema plus row counts and the allowed values of categorical columns."""
        if self._schema_text is not None:
            return self._schema_text
        # These reads pass through the same authorizer as model queries.
        tables = [
            row[0]
            for row in self._conn.execute(
                "SELECT sql FROM sqlite_master WHERE type = 'table' ORDER BY name"
            )
        ]
        counts = {
            name: self._conn.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0]
            for (name,) in self._conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
            ).fetchall()
        }
        date_range = self._conn.execute(
            "SELECT MIN(ordered_on), MAX(ordered_on) FROM orders"
        ).fetchone()
        lines = ["-- SQLite database of the Grão Fino specialty-coffee marketplace", ""]
        lines += [ddl.strip() + ";" for ddl in tables]
        lines += ["", "-- Row counts: " + ", ".join(f"{t}={n}" for t, n in counts.items())]
        lines += [f"-- Orders range from {date_range[0]} to {date_range[1]}."]
        lines += ["-- Revenue of an item = quantity_kg * unit_price_brl (shipping excluded)."]
        self._schema_text = "\n".join(lines)
        return self._schema_text
