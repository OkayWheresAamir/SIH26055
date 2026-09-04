"""BM25 over the passages, with a tokenizer that knows the vocabulary.

BM25 is a lexical ranking function: a passage scores highly when it contains the
query's rarer terms, more often, in fewer words. Three things make it fit this
corpus better than a semantic index would:

  * The vocabulary is fixed technical jargon. `PRI`, `dBm`, `alert-confirm` mean one
    thing here and appear verbatim in the papers that discuss them.
  * It cannot hallucinate a match. A hit is a hit because the words are on the page,
    which is exactly the standard the provenance rules ask for.
  * The score is interpretable, so "no strong match" can be a calibrated statement
    rather than a vibe.

Where lexical search normally loses -- the user asks for "how long to look at a
frequency" and the paper says "dwell time" -- a curated alias map closes the gap.
Aliases are applied at *query* time only: the index stays a literal record of what
the documents say, and the alias list can change without a rebuild.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from docsearch.corpus import Corpus, Passage

K1 = 1.5   # term-frequency saturation: how fast repeats stop helping
B = 0.75   # length normalisation: how much a long passage is penalised

# Words that carry no discriminative weight in a corpus that is entirely about
# radar scheduling. Deliberately short -- an over-eager stoplist silently deletes
# query terms, and "on"/"in"/"of" already cost almost nothing under IDF.
STOPWORDS = frozenset(
    """
    a an the and or but if then than that this these those of in on at to for from by
    with without is are was were be been being it its as we our you your they their
    there here what which who whom when where how why can could should would may might
    will shall do does did done have has had not no nor so such very more most much
    each other some any all both few own same s t don just also into over under about
    """.split()
)

# One thing said two ways. Applied to the query, in both directions, so it does not
# matter which side of the pair you happen to type. Every entry here is a phrasing
# gap seen in this project's own documents.
ALIASES: list[tuple[str, str]] = [
    ("poi", "probability of intercept"),
    ("pot", "probability of intercept"),
    ("dwell", "look duration"),
    ("dwell", "revisit"),
    ("sweep period", "scan period"),
    ("sweep period", "revisit interval"),
    ("pri", "pulse repetition interval"),
    ("prf", "pulse repetition frequency"),
    ("rf", "radio frequency"),
    ("es", "electronic support"),
    ("ew", "electronic warfare"),
    ("esm", "electronic support measures"),
    ("emitter", "radar"),
    ("emitter", "transmitter"),
    ("intercept", "detection"),
    ("interception", "intercept"),
    ("scan", "search"),
    ("agile", "frequency-agile"),
    ("rpca", "robust principal component analysis"),
    ("snr", "signal to noise ratio"),
    ("pd", "probability of detection"),
    ("pfa", "probability of false alarm"),
    ("roc", "receiver operating characteristic"),
    ("deinterleaving", "pulse separation"),
    ("band", "channel"),
    ("receiver", "esm receiver"),
    ("threshold", "detection threshold"),
    ("superheterodyne", "shr"),   # the papers use "SHR"; "superhet" appears nowhere
    ("rl", "reinforcement learning"),
    ("reward", "return"),
    ("policy", "scheduler"),
    ("schedule", "scheduling"),
]

# A number bound to its unit is one fact, not two: "100 ms" and "-90 dBm" survive
# tokenisation whole so they can be searched for as written.
# Longest-first, and case-insensitive: the corpus writes dBm, GHz, ms, µs. Ordering
# matters because a plain `db` alternative would otherwise win against `dBm` and
# leave the 'm' stranded.
_UNIT = r"(?:dbm|dbi|dbw|db|khz|mhz|ghz|hz|ms|us|µs|ns|s|%)"
_TOKEN_RE = re.compile(
    rf"""
      (?P<measure>-?\d+(?:\.\d+)?\s*{_UNIT}(?![a-z0-9]))   # 100 ms, -90 dBm, 2.5 GHz
    | (?P<symbol>[a-z]+_[a-z0-9]+)                         # P_d, T_rcv
    | (?P<word>[a-z][a-z0-9]*(?:-[a-z][a-z0-9]*)+)         # alert-confirm
    | (?P<plain>[a-z][a-z0-9]*)
    | (?P<number>-?\d+(?:\.\d+)?)
    """,
    re.VERBOSE | re.IGNORECASE,
)


def _singular(token: str) -> str:
    """Crude, deliberate de-pluralisation.

    Only the trailing bare 's', and only where it is unambiguous: 'dwells' becomes
    'dwell', but 'bias', 'analysis' and 'status' keep their ending. A real stemmer
    would fold 'scanning' onto 'scan' and also fold jargon onto nonsense, which
    costs more than it buys on a corpus this size.
    """
    if len(token) >= 4 and token.endswith("s") and not token.endswith(("ss", "us", "is", "as")):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    """Text to comparable terms, keeping the domain's spellings intact."""
    tokens: list[str] = []
    for m in _TOKEN_RE.finditer(text):
        kind = m.lastgroup
        raw = m.group().lower()
        if kind == "measure":
            tokens.append(re.sub(r"\s+", "", raw))          # "100 ms" -> "100ms"
        elif kind == "symbol":
            tokens.append(raw)
            tokens.extend(p for p in raw.split("_") if p)   # also findable apart
        elif kind == "word":
            tokens.append(raw)                              # keep the compound
            tokens.extend(_singular(p) for p in raw.split("-") if len(p) > 1)
        elif kind == "plain":
            tokens.append(_singular(raw))
        else:
            tokens.append(raw)
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


def expand_query(query: str) -> list[str]:
    """Query terms plus their aliases, deduplicated, order preserved."""
    lowered = query.lower()
    terms = tokenize(query)

    extra: list[str] = []
    for a, b in ALIASES:
        if a in lowered or all(t in terms for t in tokenize(a)):
            extra.extend(tokenize(b))
        if b in lowered or all(t in terms for t in tokenize(b)):
            extra.extend(tokenize(a))

    seen, out = set(), []
    for t in [*terms, *extra]:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


@dataclass
class Postings:
    docs: np.ndarray   # (n,) int32 — passage indices containing the term
    tf: np.ndarray     # (n,) float32 — how many times, in each


class Bm25Index:
    """Postings lists plus BM25 scoring.

    Stored as one postings list per term rather than a term-document matrix: the
    matrix would be ~20k terms x ~4k passages of mostly zeros, and the postings form
    scores a query in time proportional to the query, not the corpus.
    """

    def __init__(self, passages: list[Passage]) -> None:
        self.passages = passages
        self.n_docs = len(passages)
        self.postings: dict[str, Postings] = {}
        self.doc_len = np.zeros(self.n_docs, dtype=np.float32)

        staging: dict[str, dict[int, int]] = {}
        for i, p in enumerate(passages):
            terms = tokenize(p.text)
            self.doc_len[i] = len(terms)
            counts: dict[str, int] = {}
            for t in terms:
                counts[t] = counts.get(t, 0) + 1
            for t, c in counts.items():
                staging.setdefault(t, {})[i] = c

        for term, docmap in staging.items():
            docs = np.fromiter(docmap.keys(), dtype=np.int32, count=len(docmap))
            tf = np.fromiter(docmap.values(), dtype=np.float32, count=len(docmap))
            order = np.argsort(docs)
            self.postings[term] = Postings(docs=docs[order], tf=tf[order])

        self.avg_len = float(self.doc_len.mean()) if self.n_docs else 0.0
        # Guard the length-normalisation denominator for empty passages.
        self._norm = K1 * (1 - B + B * (self.doc_len / max(self.avg_len, 1e-9)))

    def idf(self, term: str) -> float:
        """Rarity weight. A term in every passage is worth ~nothing; a term in one
        is worth a lot. The +0.5 smoothing is the standard BM25 form."""
        df = len(self.postings[term].docs) if term in self.postings else 0
        return float(np.log(1.0 + (self.n_docs - df + 0.5) / (df + 0.5)))

    def score(self, terms: list[str]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Accumulate BM25 over the query's terms.

        Returns the per-passage total and, per term, its own contribution — which is
        what lets a result say *which* words it matched on.
        """
        total = np.zeros(self.n_docs, dtype=np.float32)
        per_term: dict[str, np.ndarray] = {}
        for term in terms:
            post = self.postings.get(term)
            if post is None:
                continue
            contrib = self.idf(term) * (post.tf * (K1 + 1)) / (post.tf + self._norm[post.docs])
            total[post.docs] += contrib
            per_term[term] = post.docs[contrib > 0]
        return total, per_term

    @classmethod
    def from_corpus(cls, corpus: Corpus) -> "Bm25Index":
        return cls(corpus.passages)
