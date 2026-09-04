"""Measure the retriever, and calibrate what it is allowed to call a match.

Three things get measured, and each answers a question that would otherwise be
settled by impression:

  recall@k / MRR      Does the right document come back, and how near the top?
                      This is the number that decides whether embeddings are worth
                      adding — the gate agreed for this build is recall@10 >= 0.85.

  summary-over-primary
                      How often a project-authored summary outranks the paper it
                      summarises. `CLAUDE.md` says a summary of a source is not a
                      second source, so a retriever that quietly prefers summaries
                      undermines the rule it exists to serve. Reported separately
                      because it is a *ranking* fault, not a recall fault.

  score separation    Where answerable queries stop and unanswerable ones begin.
                      This is what makes "no strong match" a calibrated statement
                      rather than a guess. It is measured on two axes, because score
                      alone turned out not to separate them: an unanswerable query
                      can score highly off one rare incidental word. Term coverage --
                      how many of the typed words are actually on the page -- does
                      separate them. Written to `.docindex/calibration.json`, which
                      `search.py` then reads.

Run: python -m docsearch.evaluate
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from docsearch.corpus import CACHE_DIR, REPO_ROOT
from docsearch.search import CALIBRATION_PATH, MIN_STRONG_COVERAGE, Searcher

GOLD_PATH = Path(__file__).resolve().parent / "gold.json"
RECALL_GATE = 0.85


@dataclass
class QueryOutcome:
    id: str
    question: str
    kind: str
    best_rank: int | None      # rank of the first correct hit, 1-based
    best_score: float
    coverage: int              # typed-query terms present in the top hit
    top_citation: str
    summary_beat_primary: bool


def _is_correct(citation: str, answer_paths: list[str]) -> bool:
    return any(citation.startswith(p + ":") for p in answer_paths)


def _is_project_summary(path: str) -> bool:
    """Project-authored documents that summarise external sources."""
    return path.startswith(("docs/project/RESEARCH_MAP.md", "docs/project/DECISIONS.md"))


def evaluate(k: int = 10, verbose: bool = True) -> dict:
    gold = json.loads(GOLD_PATH.read_text(encoding="utf-8"))
    searcher = Searcher(calibration=None)   # measure raw scores, uncalibrated

    outcomes: list[QueryOutcome] = []
    positive_scores: list[float] = []

    groups = [
        (gold["positives"], "technical"),
        (gold["presentation_positives"], "presentation"),
    ]
    for entries, collection in groups:
        for e in entries:
            result = searcher.search(e["question"], k=k, collection=collection)
            hits = result.hits

            best_rank = next(
                (h.rank for h in hits if _is_correct(h.citation, e["answer_paths"])), None
            )

            # Ranking fault: for a question about what an external source says, did a
            # project summary land above the primary?
            summary_beat = False
            if e["kind"] == "primary" and best_rank is not None:
                summary_rank = next(
                    (h.rank for h in hits if _is_project_summary(h.passage.source_path)), None
                )
                summary_beat = summary_rank is not None and summary_rank < best_rank

            outcomes.append(
                QueryOutcome(
                    id=e["id"],
                    question=e["question"],
                    kind=e["kind"],
                    best_rank=best_rank,
                    best_score=result.best_score,
                    coverage=result.top_coverage,
                    top_citation=hits[0].citation if hits else "—",
                    summary_beat_primary=summary_beat,
                )
            )
            positive_scores.append((result.best_score, result.top_coverage))

    negative_results = [searcher.search(q, k=k, collection=None) for q in gold["negatives"]]
    negative_scores = [(r.best_score, r.top_coverage) for r in negative_results]

    n = len(outcomes)
    recall_5 = sum(1 for o in outcomes if o.best_rank and o.best_rank <= 5) / n
    recall_10 = sum(1 for o in outcomes if o.best_rank and o.best_rank <= 10) / n
    mrr = sum(1.0 / o.best_rank for o in outcomes if o.best_rank) / n
    primaries = [o for o in outcomes if o.kind == "primary"]
    summary_rate = (
        sum(1 for o in primaries if o.summary_beat_primary) / len(primaries) if primaries else 0.0
    )

    pos = np.array(positive_scores, dtype=float)   # (n, 2): score, coverage
    neg = np.array(negative_scores, dtype=float)
    calib = _calibrate(pos, neg)

    if verbose:
        _report(outcomes, pos, neg, recall_5, recall_10, mrr, summary_rate, calib)

    calibration = {
        **calib,
        "min_strong_coverage": MIN_STRONG_COVERAGE,
        "n_positive": int(pos.shape[0]),
        "n_negative": int(neg.shape[0]),
        "recall_at_5": round(recall_5, 3),
        "recall_at_10": round(recall_10, 3),
        "mrr": round(mrr, 3),
        "summary_over_primary_rate": round(summary_rate, 3),
    }
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    CALIBRATION_PATH.write_text(json.dumps(calibration, indent=2), encoding="utf-8")
    return calibration


def _calibrate(pos: np.ndarray, neg: np.ndarray) -> dict:
    """Set the score floor, given that coverage does the separating.

    The floor is placed below the weakest *well-covered* positive, so a real question
    is not demoted for being answered by a short passage. Coverage >= 2 is what
    excludes the negatives, and `false_strong` re-checks that on the measured data
    rather than trusting the reasoning.
    """
    pos_ok = pos[pos[:, 1] >= MIN_STRONG_COVERAGE]
    floor_pool = pos_ok[:, 0] if pos_ok.size else pos[:, 0]

    strong_at = float(np.percentile(floor_pool, 5))
    weak_at = strong_at * 0.6

    would_be_strong = neg[(neg[:, 1] >= MIN_STRONG_COVERAGE) & (neg[:, 0] >= strong_at)]
    return {
        "strong_at": round(strong_at, 3),
        "weak_at": round(weak_at, 3),
        "false_strong": int(would_be_strong.shape[0]),
        "positive_score_min": round(float(pos[:, 0].min()), 3),
        "positive_coverage_min": int(pos[:, 1].min()),
        "negative_score_max": round(float(neg[:, 0].max()), 3),
        "negative_coverage_max": int(neg[:, 1].max()),
    }


def _report(outcomes, pos, neg, r5, r10, mrr, summary_rate, calib) -> None:
    print("=" * 82)
    print("docsearch retrieval evaluation")
    print("=" * 82)

    print(f"\n{len(outcomes)} gold questions, each verified against the passage it returns.\n")
    print(f"  {'id':26s} {'rank':>5s} {'score':>7s} {'cov':>4s}  top hit")
    print(f"  {'-' * 26} {'-' * 5} {'-' * 7} {'-' * 4}  {'-' * 32}")
    for o in sorted(outcomes, key=lambda o: (o.best_rank is None, o.best_rank or 0)):
        rank = str(o.best_rank) if o.best_rank else "MISS"
        flag = "  ⚠ summary above primary" if o.summary_beat_primary else ""
        cite = o.top_citation if len(o.top_citation) <= 44 else o.top_citation[:41] + "..."
        print(f"  {o.id:26s} {rank:>5s} {o.best_score:7.2f} {o.coverage:4d}  {cite}{flag}")

    print(f"\n{'recall@5':<28s} {r5:6.1%}")
    print(f"{'recall@10':<28s} {r10:6.1%}   gate: {RECALL_GATE:.0%}"
          f"   {'PASS' if r10 >= RECALL_GATE else 'FAIL'}")
    print(f"{'MRR':<28s} {mrr:6.3f}")
    print(f"{'summary-over-primary':<28s} {summary_rate:6.1%}   "
          "(project summary outranking the paper it summarises)")

    print("\nSeparating answerable from unanswerable:")
    print(f"  {'':12s} {'n':>3s}  {'score min':>10s} {'score max':>10s}  "
          f"{'cov min':>8s} {'cov max':>8s}")
    for name, arr in (("positives", pos), ("negatives", neg)):
        print(f"  {name:12s} {arr.shape[0]:3d}  {arr[:, 0].min():10.2f} {arr[:, 0].max():10.2f}  "
              f"{int(arr[:, 1].min()):8d} {int(arr[:, 1].max()):8d}")
    print(f"\n  Score alone does not separate them: the best negative scores "
          f"{calib['negative_score_max']:.2f},")
    print(f"  above the weakest positive at {calib['positive_score_min']:.2f}. "
          "Term coverage does:")
    print(f"  no unanswerable query put more than {calib['negative_coverage_max']} of its "
          f"own words on the page.")
    print(f"\n  rule: strong needs coverage >= {MIN_STRONG_COVERAGE} and score >= "
          f"{calib['strong_at']:.2f}   (weak >= {calib['weak_at']:.2f})")
    print(f"  negatives that would still pass as strong: {calib['false_strong']}")
    print(f"  written to {CALIBRATION_PATH.relative_to(REPO_ROOT)}")

    if r10 >= RECALL_GATE:
        print(f"\nVERDICT: lexical retrieval clears the {RECALL_GATE:.0%} gate. "
              "No embedding model needed.")
    else:
        print(f"\nVERDICT: below the {RECALL_GATE:.0%} gate — embeddings are worth adding.")
    print("=" * 82)


def _main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="python -m docsearch.evaluate")
    ap.add_argument("-k", type=int, default=10, help="depth to evaluate at (default 10)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    evaluate(k=args.k, verbose=not args.quiet)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
