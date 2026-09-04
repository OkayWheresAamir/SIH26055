"""Tests for the document retrieval layer.

These check the properties the provenance rules depend on: a passage can always name
where it came from, a known-unreadable file is reported rather than dropped, the
pitch-deck material cannot leak into a technical answer, and a query the corpus
cannot answer is refused instead of being served the least-bad page.
"""

from __future__ import annotations

import json

import pytest

from docsearch.corpus import (
    MAX_PASSAGE_CHARS,
    Corpus,
    FileRecord,
    Passage,
    _normalise,
    _split_long,
    load_markdown,
    load_or_build,
    load_python,
)
from docsearch.index import Bm25Index, expand_query, tokenize
from docsearch.search import MIN_STRONG_COVERAGE, Searcher, _classify


@pytest.fixture(scope="module")
def corpus() -> Corpus:
    return load_or_build()


@pytest.fixture(scope="module")
def searcher(corpus: Corpus) -> Searcher:
    return Searcher(corpus=corpus)


# ------------------------------------------------------------------- corpus --


def test_every_passage_can_cite_itself(corpus: Corpus) -> None:
    for p in corpus.passages:
        assert p.source_path.startswith("docs/")
        assert p.locator
        assert p.citation == f"{p.source_path}:{p.locator}"
        assert p.authority in {"project", "protocol", "reference", "teammate-work"}
        assert p.collection in {"technical", "presentation"}
        assert p.n_tokens > 0


def test_pdf_locators_are_page_numbers(corpus: Corpus) -> None:
    pdf = [p for p in corpus.passages if p.source_type == "pdf"]
    assert pdf, "expected PDF passages"
    for p in pdf:
        assert p.locator.startswith("p."), p.locator


def test_image_only_pdfs_are_recorded_not_dropped(corpus: Corpus) -> None:
    """The 8 image-only PDFs must be visible as blind spots.

    Silently omitting them would turn a known gap into an invisible one, which is
    the failure this repository exists to avoid.
    """
    blind = {f.source_path for f in corpus.blind_spots}
    assert "docs/reference/scheduling/me 2.0.pdf" in blind
    assert any("SIH_2024" in p for p in blind)
    for f in corpus.blind_spots:
        assert f.reason, f"{f.source_path} has no stated reason"
        assert not any(p.source_path == f.source_path for p in corpus.passages)


def test_ppt_material_is_a_separate_collection(corpus: Corpus) -> None:
    for p in corpus.passages:
        expected = "presentation" if "docs/reference/PPT/" in p.source_path else "technical"
        assert p.collection == expected, p.source_path


def test_passages_respect_the_length_cap(corpus: Corpus) -> None:
    assert all(len(p.text) <= MAX_PASSAGE_CHARS for p in corpus.passages)


def test_dehyphenation_rejoins_broken_words() -> None:
    assert "randomphase" in _normalise("random-\nphase")
    assert _normalise("well-known") == "well-known"     # a real hyphen survives


def test_split_long_keeps_offsets_in_order() -> None:
    text = "\n\n".join(f"paragraph {i} " + "word " * 60 for i in range(8))
    parts = _split_long(text, limit=500)
    assert len(parts) > 1
    assert [o for o, _ in parts] == sorted(o for o, _ in parts)
    assert all(len(c) <= 700 for _, c in parts)


def test_markdown_locator_is_the_heading_path(tmp_path) -> None:
    f = tmp_path / "x.md"
    f.write_text("# Top\nintro text here padded out to clear the minimum length.\n"
                 "## Middle\nmore body text, also padded out well past the minimum.\n")
    passages, record = load_markdown(f, "docs/project/x.md")
    locators = [p.locator for p in passages]
    assert "Top" in locators
    assert "Top > Middle" in locators
    assert record.n_passages == len(passages)


def test_python_locator_is_a_line_range(tmp_path) -> None:
    f = tmp_path / "m.py"
    f.write_text('"""Module doc, long enough to be kept as its own passage here."""\n\n'
                 "def alpha():\n    " + "x = 1\n    " * 12 + "\n")
    passages, _ = load_python(f, "docs/teammate-work/m.py")
    assert any(p.locator == "(module docstring)" for p in passages)
    assert any(p.locator.startswith("L") and p.section == "alpha" for p in passages)


# -------------------------------------------------------------------- index --


def test_tokenizer_keeps_domain_jargon() -> None:
    assert "100ms" in tokenize("a dwell of 100 ms")
    assert "-90dbm" in tokenize("threshold at -90 dBm")
    assert "alert-confirm" in tokenize("alert-confirm detection")
    assert "p_d" in tokenize("the P_d curve")


def test_tokenizer_depluralises_only_where_safe() -> None:
    assert "dwell" in tokenize("dwells")
    assert "emitter" in tokenize("emitters")
    assert "bias" in tokenize("bias")            # not 'bia'
    assert "analysis" in tokenize("analysis")    # not 'analysi'


def test_aliases_expand_both_ways() -> None:
    assert "probability" in expand_query("PoI")
    assert "poi" in expand_query("probability of intercept")


def test_idf_falls_as_a_term_gets_commoner(corpus: Corpus) -> None:
    idx = Bm25Index(corpus.passages[:200])
    common = max(idx.postings, key=lambda t: len(idx.postings[t].docs))
    rare = min(idx.postings, key=lambda t: len(idx.postings[t].docs))
    assert idx.idf(common) < idx.idf(rare)


# ------------------------------------------------------------------- search --


def test_scores_decrease_with_rank(searcher: Searcher) -> None:
    result = searcher.search("dwell time optimisation", k=10)
    scores = [h.score for h in result.hits]
    assert scores == sorted(scores, reverse=True)
    assert [h.rank for h in result.hits] == list(range(1, len(result.hits) + 1))


def test_scores_are_exposed_not_just_ranks(searcher: Searcher) -> None:
    result = searcher.search("probability of intercept", k=3)
    assert result.hits
    assert all(h.score > 0 for h in result.hits)
    assert result.best_score == result.hits[0].score


def test_technical_search_never_returns_pitch_decks(searcher: Searcher) -> None:
    for query in ("dwell scheduling", "probability of intercept", "emitter bands"):
        result = searcher.search(query, k=20)
        assert all("docs/reference/PPT/" not in h.passage.source_path for h in result.hits), query


def test_presentation_collection_is_reachable(searcher: Searcher) -> None:
    result = searcher.search("pitch deck slide structure", k=5, collection="presentation")
    assert result.hits
    assert all(h.passage.collection == "presentation" for h in result.hits)


def test_filters_apply_before_ranking(searcher: Searcher) -> None:
    result = searcher.search("intercept", k=5, authority="reference")
    assert result.hits
    assert all(h.passage.authority == "reference" for h in result.hits)


def test_off_corpus_query_is_refused(searcher: Searcher) -> None:
    """The negative control: a corpus of radar papers must decline a tap question."""
    result = searcher.search("how do I fix a leaking tap", k=5)
    assert result.confidence == "none"
    assert not result


def test_a_rare_incidental_word_is_not_enough_for_confidence(searcher: Searcher) -> None:
    """'UK passport' scores 16.9 off one affiliation line; coverage must catch it."""
    result = searcher.search("how do I renew a UK passport", k=5)
    assert result.top_coverage < MIN_STRONG_COVERAGE
    assert result.confidence != "strong"


def test_real_question_is_confident_and_cites_the_source(searcher: Searcher) -> None:
    result = searcher.search("alert confirm detection dwell time", k=5)
    assert result.confidence == "strong"
    assert result.top_coverage >= MIN_STRONG_COVERAGE
    assert any("Dwell_Time_Optimization" in h.citation for h in result.hits)


def test_classify_needs_both_coverage_and_score() -> None:
    calib = {"strong_at": 10.0, "weak_at": 6.0}
    assert _classify(20.0, 3, calib)[0] == "strong"
    assert _classify(20.0, 1, calib)[0] == "weak"      # high score, one word — not enough
    assert _classify(2.0, 5, calib)[0] == "none"       # many words, no weight
    assert _classify(20.0, 3, None)[0] == "uncalibrated"


def test_snippet_centres_on_the_match(searcher: Searcher) -> None:
    result = searcher.search("Farey series Diophantine approximation", k=1)
    snippet = result.hits[0].snippet(200)
    assert len(snippet) <= 230
    assert any(t in snippet.lower() for t in ("farey", "diophantine", "approximation"))


# --------------------------------------------------------------- evaluation --


def test_gold_set_answers_point_at_files_that_exist() -> None:
    from docsearch.corpus import REPO_ROOT
    from docsearch.evaluate import GOLD_PATH

    gold = json.loads(GOLD_PATH.read_text(encoding="utf-8"))
    for group in ("positives", "presentation_positives"):
        for entry in gold[group]:
            for path in entry["answer_paths"]:
                assert (REPO_ROOT / path).exists(), f"{entry['id']} -> missing {path}"


def test_retrieval_still_clears_the_recall_gate() -> None:
    """The gate that decided against embeddings. If this fails, revisit that call."""
    from docsearch.evaluate import RECALL_GATE, evaluate

    result = evaluate(verbose=False)
    assert result["recall_at_10"] >= RECALL_GATE
    assert result["false_strong"] == 0
