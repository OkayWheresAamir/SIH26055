"""Retrieval over `docs/`, built so every answer arrives with a citation.

    from docsearch import search
    for hit in search("dwell time optimisation", k=5):
        print(hit.score, hit.citation)
        print(hit.snippet())

The corpus is extracted on first use and cached in a gitignored `.docindex/`; it
rebuilds itself when a file under `docs/` changes.
"""

from docsearch.corpus import Corpus, Passage, build, load_or_build
from docsearch.index import Bm25Index, tokenize
from docsearch.search import Hit, SearchResult, Searcher, search

__all__ = [
    "Bm25Index",
    "Corpus",
    "Hit",
    "Passage",
    "SearchResult",
    "Searcher",
    "build",
    "load_or_build",
    "search",
    "tokenize",
]
