"""Small BM25 retriever over the markdown knowledge base.

Documents are split on ``##`` headings so every hit is a self-contained section that
can be cited as ``file.md › Heading``. BM25 is a good fit here: the corpus is small
and domain-specific (process names, varieties, regions), so exact term matching beats
general-purpose embeddings and needs no external service or model download.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

_STOPWORD_TEXT = """
a an and are as at be by can do does for from has have how i in is it its of on or our
should so that the their them then there these this to was what when where which who why
will with you your
"""
STOPWORDS = frozenset(_STOPWORD_TEXT.split())


def tokenize(text: str) -> list[str]:
    # Fold accents so "Caparaó" matches "caparao", then crude plural stemming.
    folded = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
    tokens = []
    for token in re.findall(r"[a-z0-9]+", folded):
        if token in STOPWORDS:
            continue
        if len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
            token = token[:-1]
        tokens.append(token)
    return tokens


@dataclass(frozen=True)
class Chunk:
    source: str  # file name, e.g. "processing.md"
    heading: str
    text: str

    @property
    def citation(self) -> str:
        return f"{self.source} › {self.heading}"


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


def chunk_markdown(source: str, markdown: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    title = source
    heading, lines = None, []

    def flush() -> None:
        body = "\n".join(lines).strip()
        if body:
            chunks.append(Chunk(source, heading or title, body))

    for line in markdown.splitlines():
        if line.startswith("# "):
            title = line[2:].strip()
        elif line.startswith("## "):
            flush()
            heading, lines = line[3:].strip(), []
        else:
            lines.append(line)
    flush()
    return chunks


class KnowledgeBase:
    def __init__(self, chunks: list[Chunk], *, k1: float = 1.5, b: float = 0.75) -> None:
        if not chunks:
            raise ValueError("knowledge base is empty")
        self.chunks = chunks
        self.k1, self.b = k1, b
        # Index the heading too: it is often the most specific text in the section.
        self._docs = [tokenize(f"{c.heading} {c.heading} {c.text}") for c in chunks]
        self._tf = [Counter(doc) for doc in self._docs]
        self._avgdl = sum(len(d) for d in self._docs) / len(self._docs)
        df = Counter(term for doc in self._docs for term in set(doc))
        n = len(self._docs)
        self._idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    @classmethod
    def from_directory(cls, directory: str | Path | None = None) -> KnowledgeBase:
        if directory is None:
            root = resources.files("coffee_agent") / "knowledge"
            files = sorted(
                (f.name, f.read_text(encoding="utf-8"))
                for f in root.iterdir()
                if f.name.endswith(".md")
            )
        else:
            files = sorted(
                (p.name, p.read_text(encoding="utf-8")) for p in Path(directory).glob("*.md")
            )
        chunks = [chunk for name, text in files for chunk in chunk_markdown(name, text)]
        return cls(chunks)

    def search(self, query: str, k: int = 4) -> list[Hit]:
        terms = tokenize(query)
        if not terms:
            return []
        scores = []
        for idx, tf in enumerate(self._tf):
            dl = len(self._docs[idx])
            score = 0.0
            for term in terms:
                freq = tf.get(term)
                if not freq:
                    continue
                norm = (
                    freq
                    * (self.k1 + 1)
                    / (freq + self.k1 * (1 - self.b + self.b * dl / self._avgdl))
                )
                score += self._idf[term] * norm
            if score > 0:
                scores.append((score, idx))
        scores.sort(key=lambda pair: (-pair[0], pair[1]))
        return [Hit(self.chunks[idx], round(score, 3)) for score, idx in scores[:k]]
