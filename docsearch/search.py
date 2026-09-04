"""The retrieval interface: query in, cited passages out.

Two things here are deliberate and worth stating.

**Scores are returned, never hidden.** A rank tells you an ordering; a score tells
you whether the ordering means anything. Callers get the raw BM25 value.

**The tool is allowed to say it found nothing.** Any ranker will return a top-k for
any query, including a query its corpus cannot answer -- the worst failure mode in
a repository whose rules turn on citing real sources. A result therefore has to
clear two conditions to be called `strong`: a score floor, and at least two distinct
terms from the typed query actually appearing in the passage. The second condition
is the one that matters, and `_classify` records the measurement that showed why.
Both come from `docsearch.evaluate`; until it has been run, confidence reports
`uncalibrated` rather than guessing.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from docsearch.corpus import CACHE_DIR, REPO_ROOT, Corpus, Passage, load_or_build
from docsearch.index import Bm25Index, expand_query, tokenize

CALIBRATION_PATH = CACHE_DIR / "calibration.json"


@dataclass(frozen=True)
class Hit:
    """A passage, why it surfaced, and how strongly."""

    passage: Passage
    score: float
    rank: int
    matched_terms: tuple[str, ...]
    term_coverage: int = 0   # distinct terms from the *typed* query found in the text

    @property
    def citation(self) -> str:
        return self.passage.citation

    def snippet(self, width: int = 320) -> str:
        """The window of the passage densest in matched terms."""
        return _snippet(self.passage.text, self.matched_terms, width)


@dataclass(frozen=True)
class SearchResult:
    """Hits plus an honest statement about whether to trust them."""

    query: str
    terms: tuple[str, ...]
    hits: tuple[Hit, ...]
    best_score: float
    top_coverage: int        # distinct typed-query terms present in the best hit
    confidence: str          # strong | weak | none | uncalibrated
    threshold: float | None
    n_searched: int

    def __bool__(self) -> bool:
        return self.confidence in {"strong", "weak", "uncalibrated"} and bool(self.hits)

    def __iter__(self):
        return iter(self.hits)

    def __len__(self) -> int:
        return len(self.hits)


# ------------------------------------------------------------------- snippet --


def _snippet(text: str, terms: tuple[str, ...], width: int) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= width or not terms:
        return text[:width]

    lowered = text.lower()
    hits = [m.start() for t in terms for m in re.finditer(re.escape(t), lowered)]
    if not hits:
        return text[:width] + "…"

    hits.sort()
    # Densest window: the start position with the most matches within `width`.
    best_i, best_n = hits[0], 0
    for h in hits:
        n = sum(1 for x in hits if h <= x < h + width)
        if n > best_n:
            best_i, best_n = h, n

    start = max(0, best_i - width // 4)
    if start > 0:
        space = text.find(" ", start)
        start = space + 1 if 0 <= space < start + 40 else start
    end = min(len(text), start + width)
    return ("… " if start > 0 else "") + text[start:end].strip() + ("…" if end < len(text) else "")


# --------------------------------------------------------------- calibration --


def load_calibration(path: Path = CALIBRATION_PATH) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


MIN_STRONG_COVERAGE = 2


def _classify(best: float, coverage: int, calib: dict | None) -> tuple[str, float | None]:
    """Two conditions, and the coverage one is what actually does the work.

    Score alone cannot separate answerable from unanswerable queries here. Measured
    on the gold set: "how do I renew a UK passport" scores 16.86 — above four real
    questions — because "UK" is rare in this corpus and matches an author's
    affiliation in the TSRD paper. One rare incidental word is enough to fake a high
    BM25 score.

    Requiring **two distinct terms from the typed query** to appear in the passage
    kills that failure outright: across the gold set every unanswerable query matched
    at most one, and every well-answered one matched two or more. Score then acts
    only as a floor.
    """
    if calib is None:
        return "uncalibrated", None
    strong_at = float(calib["strong_at"])
    weak_at = float(calib.get("weak_at", strong_at * 0.6))
    if coverage >= MIN_STRONG_COVERAGE and best >= strong_at:
        return "strong", strong_at
    if coverage >= 1 and best >= weak_at:
        return "weak", strong_at
    return "none", strong_at


# ------------------------------------------------------------------ searcher --


class Searcher:
    """Holds the corpus and index so repeated queries do not rebuild them."""

    def __init__(self, corpus: Corpus | None = None, calibration: dict | None = ...) -> None:
        self.corpus = corpus if corpus is not None else load_or_build()
        self.index = Bm25Index.from_corpus(self.corpus)
        self.calibration = load_calibration() if calibration is ... else calibration

    def search(
        self,
        query: str,
        k: int = 8,
        *,
        collection: str | None = "technical",
        authority: str | None = None,
        source_type: str | None = None,
        path_contains: str | None = None,
    ) -> SearchResult:
        """Rank passages against `query`.

        `collection` defaults to `technical`: the pitch-deck material under
        `docs/reference/PPT/` is excluded unless asked for, so it cannot contaminate
        a question about the design. Pass `collection=None` to search everything.

        Filters are applied *before* ranking, so `k` results are always `k` results
        from the allowed set rather than whatever survives a post-hoc cull.
        """
        base_terms = set(tokenize(query))          # what the user actually typed
        terms = expand_query(query)                # plus aliases, for ranking
        scores, per_term = self.index.score(terms)

        allowed = self._mask(collection, authority, source_type, path_contains)
        n_searched = int(allowed.sum())
        scores = np.where(allowed, scores, -np.inf)

        order = np.argsort(-scores)[: max(k, 1)]
        order = [i for i in order if np.isfinite(scores[i]) and scores[i] > 0]

        hits = []
        for rank, i in enumerate(order, start=1):
            passage = self.corpus.passages[i]
            matched = tuple(t for t, docs in per_term.items() if i in docs)
            hits.append(
                Hit(
                    passage=passage,
                    score=float(scores[i]),
                    rank=rank,
                    matched_terms=matched,
                    term_coverage=len(base_terms & set(tokenize(passage.text))),
                )
            )

        best = hits[0].score if hits else 0.0
        coverage = hits[0].term_coverage if hits else 0
        confidence, threshold = _classify(best, coverage, self.calibration)
        return SearchResult(
            query=query,
            terms=tuple(terms),
            hits=tuple(hits),
            best_score=best,
            top_coverage=coverage,
            confidence=confidence,
            threshold=threshold,
            n_searched=n_searched,
        )

    def _mask(
        self,
        collection: str | None,
        authority: str | None,
        source_type: str | None,
        path_contains: str | None,
    ) -> np.ndarray:
        mask = np.ones(len(self.corpus.passages), dtype=bool)
        for i, p in enumerate(self.corpus.passages):
            if collection and p.collection != collection:
                mask[i] = False
            elif authority and p.authority != authority:
                mask[i] = False
            elif source_type and p.source_type != source_type:
                mask[i] = False
            elif path_contains and path_contains not in p.source_path:
                mask[i] = False
        return mask


_SEARCHER: Searcher | None = None


def search(query: str, k: int = 8, **kw) -> SearchResult:
    """Module-level convenience; builds the index once per process."""
    global _SEARCHER
    if _SEARCHER is None:
        _SEARCHER = Searcher()
    return _SEARCHER.search(query, k, **kw)


# ----------------------------------------------------------------------- cli --


def _render(result: SearchResult, width: int, show_terms: bool) -> str:
    lines = []
    if result.confidence == "none":
        lines.append(
            f"no strong match over {result.n_searched} passages — best score "
            f"{result.best_score:.2f} (cut {result.threshold:.2f}), and only "
            f"{result.top_coverage} of your terms appear in it"
        )
        lines.append(
            "the corpus does not appear to answer this. Nothing is shown rather than "
            "ranking the least-bad page."
        )
        return "\n".join(lines)

    if not result.hits:
        lines.append(f"no matches at all over {result.n_searched} passages")
        return "\n".join(lines)

    if result.confidence == "weak":
        banner = (
            f"  (weak — best {result.best_score:.2f} is below the "
            f"{result.threshold:.2f} confidence cut; treat as a lead, not an answer)"
        )
    elif result.confidence == "uncalibrated":
        banner = "  (uncalibrated — run python -m docsearch.evaluate to set the confidence cut)"
    else:
        banner = ""
    lines.append(f"{len(result.hits)} hits over {result.n_searched} passages{banner}")
    if show_terms:
        lines.append(f"terms: {', '.join(result.terms)}")
    lines.append("")

    for h in result.hits:
        p = h.passage
        lines.append(f"{h.rank:2d}. {h.score:7.2f}  {p.citation}")
        meta = f"[{p.authority} · {p.collection} · {p.source_type}]"
        if p.section:
            meta += f" § {p.section}"
        lines.append(f"           {meta}")
        if h.matched_terms:
            lines.append(f"           matched: {', '.join(sorted(h.matched_terms))}")
        lines.append(f"           {h.snippet(width)}")
        lines.append("")
    return "\n".join(lines)


def _main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        prog="python -m docsearch",
        description="Search docs/ and get back citable passages.",
    )
    ap.add_argument("query", nargs="+", help="what to look for")
    ap.add_argument("-k", type=int, default=8, help="how many hits (default 8)")
    ap.add_argument(
        "--collection",
        default="technical",
        choices=["technical", "presentation"],
        help="which corpus (default technical — the pitch decks are excluded)",
    )
    ap.add_argument("--all", action="store_true", help="search every collection")
    ap.add_argument(
        "--authority",
        choices=["project", "protocol", "reference", "teammate-work"],
        help="restrict by how much weight the source carries",
    )
    ap.add_argument(
        "--primary",
        action="store_true",
        help="external sources only — skip our own summaries of them "
             "(shorthand for --authority reference)",
    )
    ap.add_argument("--type", dest="source_type", choices=["pdf", "markdown", "html", "python"])
    ap.add_argument("--path", dest="path_contains", help="restrict to paths containing this")
    ap.add_argument("--width", type=int, default=320, help="snippet width")
    ap.add_argument("--terms", action="store_true", help="show the expanded query terms")
    args = ap.parse_args(argv)

    result = Searcher().search(
        " ".join(args.query),
        k=args.k,
        collection=None if args.all else args.collection,
        authority="reference" if args.primary else args.authority,
        source_type=args.source_type,
        path_contains=args.path_contains,
    )
    print(_render(result, args.width, args.terms))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
