# `docsearch/`

Search over `docs/`, built so every answer arrives with a citation you can paste.

`docs/` holds 35 PDFs across 432 pages. Markdown is greppable; **PDFs are not**, which is how
things get glossed over in the older documents. This makes them reachable, and it returns
`file.pdf:p.7` rather than just a filename — the "Relevant section/page" the research protocol
asks for.

## Use it

```bash
.venv/bin/python -m docsearch "alert confirm dwell time"          # search
.venv/bin/python -m docsearch "..." --primary                     # external sources only
.venv/bin/python -m docsearch "..." --collection presentation     # the pitch decks
.venv/bin/python -m docsearch.corpus --blind-spots                # what it cannot read
.venv/bin/python -m docsearch.evaluate                            # measure the retriever
```

```python
from docsearch import search

for hit in search("probability of intercept", k=5):
    print(hit.score, hit.citation)     # 23.08  docs/reference/.../optimumsearch.pdf:p.46
    print(hit.snippet())
```

The index caches to a gitignored `.docindex/` and rebuilds itself when anything under `docs/`
changes. A full rebuild takes about 1.6 s.

## How it works, and why this shape

**Lexical (BM25), not embeddings.** At 432 pages the bottleneck was never ranking quality — it
was that the text could not be reached at all. The vocabulary here is fixed jargon (`PRI`,
`dwell`, `alert-confirm`), which lexical search matches exactly, and a lexical hit cannot be
hallucinated: the words are on the page. The decision was measured, not assumed —
`evaluate.py` reports **recall@10 = 100%, recall@5 = 90.9%, MRR = 0.70** over 22 verified gold
questions, against an agreed gate of 0.85. Re-run it if the corpus changes; if recall drops
below the gate, that is the signal to add embeddings.

**Everything is a `Passage`.** One record — text, path, locator, authority, collection — no
matter which loader produced it. Adding a source type means writing one function in
`corpus.py`; nothing downstream knows what a PDF is.

**Two metadata fields do real work.** `authority` (project / protocol / reference /
teammate-work) is read off the `docs/` layout, which already encodes how much weight a source
carries. `collection` (technical / presentation) keeps the pitch decks out of technical
answers — search defaults to `technical`.

**It can say no.** Any ranker returns a top-k for any query, including one its corpus cannot
answer. A `strong` verdict needs two conditions: a score floor *and* at least two distinct
words from your query actually appearing in the passage. The second is what matters — measured
on the gold set, "how do I renew a UK passport" scores 16.9, above four real questions, because
`UK` is rare here and matches an author's affiliation. Score alone cannot tell those apart;
coverage can. Every unanswerable query matched at most one word; every well-answered one
matched two or more.

## Known limits

- **Maths does not extract.** `T_rcv` comes out of `optimumsearch.pdf` as `rcv \n T`. The tool
  finds the page an equation is on; read the equation in the PDF.
- **Eight PDFs have no text layer** — `me 2.0.pdf` and seven pitch decks exported as images.
  They are recorded as blind spots, never silently dropped. OCR would flatten the decks into a
  bag of slide titles and lose the layout, which is the part worth learning; read those pages
  as images instead.
- **Our own summaries sometimes outrank the papers they summarise** — 12.5% of primary-source
  questions in the gold set. `--primary` restricts to external sources when that matters.

## Files

| File | What it does |
|---|---|
| `corpus.py` | `Passage`, the per-extension loaders, the cache, blind-spot recording |
| `index.py` | BM25 in numpy, the jargon-preserving tokenizer, the query alias map |
| `search.py` | `search()` / `Searcher`, filters, confidence, the CLI |
| `evaluate.py` | recall@k, MRR, summary-over-primary, and the confidence calibration |
| `gold.json` | 22 verified questions + 8 deliberately unanswerable ones |

Tests: `.venv/bin/python -m pytest tests/test_docsearch.py -q`
