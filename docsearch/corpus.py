"""The unit of retrieval, and the loaders that produce it.

Everything in `docs/` becomes a `Passage`: a span of text that knows where it came
from precisely enough to cite. That is the whole point of this layer -- the repo's
provenance rules ask for "Relevant section/page", so a passage that cannot name its
page is not worth indexing.

Nothing downstream knows what a PDF is. `index.py` and `search.py` see only
`Passage`, which is what lets a new source type (an HDF5 attribute dump, a web
page, another code file) be added by writing one loader and touching nothing else.

Two pieces of metadata do real work beyond citation:

    authority    project | protocol | reference | teammate-work -- read off the
                 docs/ folder layout, which already encodes how much weight a
                 document is allowed to carry (docs/README.md).
    collection   technical | presentation -- keeps the pitch-deck material and the
                 technical corpus apart. Default search is scoped to `technical`;
                 a slide from someone else's 2024 deck must never surface in an
                 answer about dwell scheduling.

Some PDFs have no text layer at all. They are recorded as blind spots rather than
dropped, because a known gap that is invisible is worse than one that is written
down.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterator

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_ROOT = REPO_ROOT / "docs"
CACHE_DIR = REPO_ROOT / ".docindex"
CORPUS_PATH = CACHE_DIR / "corpus.jsonl"
MANIFEST_PATH = CACHE_DIR / "manifest.json"

# A page of a paper runs 2-5k characters. Splitting long spans keeps BM25's length
# normalisation honest -- one 90 KB markdown section would otherwise swamp the
# average document length and distort every score in the index.
MAX_PASSAGE_CHARS = 2000
MIN_PASSAGE_CHARS = 40

# Below this many characters per page, a PDF has no usable text layer: it is a scan
# or a slide exported as pictures. Measured on this corpus, real text pages run
# 700-4900 chars/page and the image-only files return exactly 0.
MIN_CHARS_PER_PAGE = 50


@dataclass(frozen=True)
class Passage:
    """One citable span of text, whatever kind of file it came from."""

    text: str
    source_path: str              # repo-root-relative, per docs/README.md
    source_type: str              # pdf | markdown | html | python
    locator: str                  # "p.7" | "Document records > Köksal" | "L120-168"
    authority: str                # project | protocol | reference | teammate-work
    collection: str               # technical | presentation
    section: str | None           # nearest enclosing heading, where there is one
    char_span: tuple[int, int]    # offsets within the source unit (page/section)
    n_tokens: int

    @property
    def citation(self) -> str:
        """`docs/reference/scheduling/optimumsearch.pdf:p.46` -- paste-ready."""
        return f"{self.source_path}:{self.locator}"


@dataclass
class FileRecord:
    """What happened when we tried to read one file."""

    source_path: str
    source_type: str
    sha256: str
    n_passages: int
    n_chars: int
    n_pages: int | None = None
    readable: bool = True
    reason: str = ""


@dataclass
class Corpus:
    passages: list[Passage] = field(default_factory=list)
    files: list[FileRecord] = field(default_factory=list)

    @property
    def blind_spots(self) -> list[FileRecord]:
        return [f for f in self.files if not f.readable]

    def __len__(self) -> int:
        return len(self.passages)


# ------------------------------------------------------------------ classify --


def _authority_of(rel_path: str) -> str:
    parts = Path(rel_path).parts
    if len(parts) >= 2 and parts[0] == "docs":
        second = parts[1]
        if second in {"project", "protocol", "reference", "teammate-work"}:
            return second
    return "reference"


def _collection_of(rel_path: str) -> str:
    return "presentation" if "docs/reference/PPT/" in rel_path else "technical"


# --------------------------------------------------------------------- clean --


def _normalise(text: str) -> str:
    """Undo the artefacts of PDF text extraction.

    Words broken across a line by hyphenation ('random-\\nphase') are rejoined --
    rare in this corpus (5 in the 110 pages of optimumsearch.pdf) but each one
    otherwise costs a searchable term. Runs of blank lines and trailing spaces are
    collapsed so paragraph splitting downstream has something regular to work with.
    """
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = text.replace("­", "")            # soft hyphen
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _split_long(text: str, limit: int = MAX_PASSAGE_CHARS) -> list[tuple[int, str]]:
    """Break an over-long span at paragraph boundaries, keeping char offsets.

    Returns (start_offset, chunk) so a passage can still point at where inside the
    page or section it came from.
    """
    if len(text) <= limit:
        return [(0, text)]

    chunks: list[tuple[int, str]] = []
    start = 0
    while start < len(text):
        if len(text) - start <= limit:
            chunks.append((start, text[start:]))
            break
        window = text[start : start + limit]
        cut = window.rfind("\n\n")
        if cut < limit // 3:
            cut = window.rfind("\n")
        if cut < limit // 3:
            cut = window.rfind(". ")
            cut = cut + 1 if cut > 0 else -1
        if cut < limit // 3:
            cut = limit
        chunks.append((start, text[start : start + cut]))
        start += cut
    return [(o, c.strip()) for o, c in chunks if c.strip()]


def _count_tokens(text: str) -> int:
    return len(re.findall(r"\w+", text))


def _emit(
    text: str,
    *,
    rel_path: str,
    source_type: str,
    locator: str,
    section: str | None,
) -> Iterator[Passage]:
    """Turn one source unit (a page, a heading section, a def) into passages."""
    text = _normalise(text)
    if len(text) < MIN_PASSAGE_CHARS:
        return
    parts = _split_long(text)
    for i, (offset, chunk) in enumerate(parts):
        if len(chunk) < MIN_PASSAGE_CHARS:
            continue
        loc = locator if len(parts) == 1 else f"{locator} ({i + 1}/{len(parts)})"
        yield Passage(
            text=chunk,
            source_path=rel_path,
            source_type=source_type,
            locator=loc,
            authority=_authority_of(rel_path),
            collection=_collection_of(rel_path),
            section=section,
            char_span=(offset, offset + len(chunk)),
            n_tokens=_count_tokens(chunk),
        )


# ------------------------------------------------------------------- loaders --


def load_pdf(path: Path, rel_path: str) -> tuple[list[Passage], FileRecord]:
    """One passage per page (split further if the page is dense).

    Page numbers are 1-based to match how a human cites a paper, so `p.46` here is
    the page you would turn to.
    """
    import pymupdf

    passages: list[Passage] = []
    total_chars = 0
    with pymupdf.open(path) as doc:
        n_pages = doc.page_count
        for i in range(n_pages):
            raw = doc[i].get_text()
            total_chars += len(raw)
            passages.extend(
                _emit(
                    raw,
                    rel_path=rel_path,
                    source_type="pdf",
                    locator=f"p.{i + 1}",
                    section=None,
                )
            )

    readable = total_chars >= MIN_CHARS_PER_PAGE * max(n_pages, 1)
    record = FileRecord(
        source_path=rel_path,
        source_type="pdf",
        sha256=_sha256(path),
        n_passages=len(passages),
        n_chars=total_chars,
        n_pages=n_pages,
        readable=readable,
        reason=(
            ""
            if readable
            else f"no text layer ({total_chars} chars over {n_pages} pages) — "
            "scanned or exported as images; read visually"
        ),
    )
    return (passages if readable else []), record


_MD_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*$", re.MULTILINE)


def load_markdown(path: Path, rel_path: str) -> tuple[list[Passage], FileRecord]:
    """Split at headings, and carry the heading path as the locator.

    `DECISIONS.md` is 94 KB in one file; without this every hit in it would cite the
    whole file and tell you nothing about where to look.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    passages: list[Passage] = []

    matches = list(_MD_HEADING.finditer(text))
    if not matches:
        passages.extend(
            _emit(text, rel_path=rel_path, source_type="markdown", locator="(body)", section=None)
        )
    else:
        if matches[0].start() > 0:
            passages.extend(
                _emit(
                    text[: matches[0].start()],
                    rel_path=rel_path,
                    source_type="markdown",
                    locator="(preamble)",
                    section=None,
                )
            )
        trail: list[str] = []
        for m, nxt in zip(matches, matches[1:] + [None]):
            level, title = len(m.group(1)), m.group(2).strip()
            trail = trail[: level - 1] + [title]
            body = text[m.end() : (nxt.start() if nxt else len(text))]
            locator = " > ".join(trail)
            passages.extend(
                _emit(
                    f"{title}\n{body}",
                    rel_path=rel_path,
                    source_type="markdown",
                    locator=locator,
                    section=title,
                )
            )

    return passages, FileRecord(
        source_path=rel_path,
        source_type="markdown",
        sha256=_sha256(path),
        n_passages=len(passages),
        n_chars=len(text),
    )


_TAG = re.compile(r"<(script|style)\b.*?</\1>", re.DOTALL | re.IGNORECASE)
_ANY_TAG = re.compile(r"<[^>]+>")


def load_html(path: Path, rel_path: str) -> tuple[list[Passage], FileRecord]:
    """Strip tags and treat the result as text.

    Only two files today (`RL_LANE_HANDOFF.html` and a teammate write-up), both of
    which have a PDF sibling, so this exists mainly so the pair stays in sync.
    """
    raw = path.read_text(encoding="utf-8", errors="replace")
    text = _ANY_TAG.sub(" ", _TAG.sub(" ", raw))
    text = re.sub(r"&nbsp;?", " ", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&lt;", "<", text)
    text = re.sub(r"&gt;", ">", text)
    text = re.sub(r"[ \t]{2,}", " ", text)

    passages = list(
        _emit(text, rel_path=rel_path, source_type="html", locator="(body)", section=None)
    )
    return passages, FileRecord(
        source_path=rel_path,
        source_type="html",
        sha256=_sha256(path),
        n_passages=len(passages),
        n_chars=len(text),
    )


def load_python(path: Path, rel_path: str) -> tuple[list[Passage], FileRecord]:
    """One passage per top-level definition, cited by line range.

    Only `teammate-work/rf_env_grounded.py` today. Falls back to whole-file if the
    source will not parse, since a syntax error in someone else's file is not a
    reason to lose it from the index.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    passages: list[Passage] = []

    try:
        tree = ast.parse(text)
    except SyntaxError:
        passages.extend(
            _emit(text, rel_path=rel_path, source_type="python", locator="(file)", section=None)
        )
    else:
        if (doc := ast.get_docstring(tree)) is not None:
            passages.extend(
                _emit(
                    doc,
                    rel_path=rel_path,
                    source_type="python",
                    locator="(module docstring)",
                    section=None,
                )
            )
        for node in tree.body:
            if not isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ):
                continue
            start = node.lineno
            end = getattr(node, "end_lineno", start) or start
            passages.extend(
                _emit(
                    "\n".join(lines[start - 1 : end]),
                    rel_path=rel_path,
                    source_type="python",
                    locator=f"L{start}-{end}",
                    section=node.name,
                )
            )

    return passages, FileRecord(
        source_path=rel_path,
        source_type="python",
        sha256=_sha256(path),
        n_passages=len(passages),
        n_chars=len(text),
    )


LOADERS: dict[str, Callable[[Path, str], tuple[list[Passage], FileRecord]]] = {
    ".pdf": load_pdf,
    ".md": load_markdown,
    ".html": load_html,
    ".htm": load_html,
    ".py": load_python,
}


# --------------------------------------------------------------------- build --


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def iter_source_files(root: Path = DOCS_ROOT) -> Iterator[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in LOADERS:
            yield path


def build(root: Path = DOCS_ROOT, verbose: bool = False) -> Corpus:
    """Read every supported file under `root` into passages."""
    corpus = Corpus()
    for path in iter_source_files(root):
        rel = path.relative_to(REPO_ROOT).as_posix()
        loader = LOADERS[path.suffix.lower()]
        try:
            passages, record = loader(path, rel)
        except Exception as exc:  # a bad file must not take the whole index down
            corpus.files.append(
                FileRecord(
                    source_path=rel,
                    source_type=path.suffix.lstrip(".").lower(),
                    sha256="",
                    n_passages=0,
                    n_chars=0,
                    readable=False,
                    reason=f"{type(exc).__name__}: {exc}",
                )
            )
            continue
        corpus.passages.extend(passages)
        corpus.files.append(record)
        if verbose:
            mark = " " if record.readable else "!"
            print(f" {mark} {len(passages):4d} passages  {rel}")
    return corpus


# --------------------------------------------------------------------- cache --


def save(corpus: Corpus, cache_dir: Path = CACHE_DIR) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    with (cache_dir / "corpus.jsonl").open("w", encoding="utf-8") as fh:
        for p in corpus.passages:
            fh.write(json.dumps(asdict(p), ensure_ascii=False) + "\n")
    (cache_dir / "manifest.json").write_text(
        json.dumps({"files": [asdict(f) for f in corpus.files]}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load(cache_dir: Path = CACHE_DIR) -> Corpus:
    corpus_path, manifest_path = cache_dir / "corpus.jsonl", cache_dir / "manifest.json"
    if not corpus_path.exists() or not manifest_path.exists():
        raise FileNotFoundError(
            f"no index at {cache_dir}. Build it: python -m docsearch.corpus --rebuild"
        )
    passages = []
    with corpus_path.open(encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            d["char_span"] = tuple(d["char_span"])
            passages.append(Passage(**d))
    files = [FileRecord(**f) for f in json.loads(manifest_path.read_text(encoding="utf-8"))["files"]]
    return Corpus(passages=passages, files=files)


def is_stale(cache_dir: Path = CACHE_DIR, root: Path = DOCS_ROOT) -> bool:
    """True if any source file was added, removed or edited since the last build."""
    try:
        manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return True
    known = {f["source_path"]: f["sha256"] for f in manifest["files"]}
    seen = set()
    for path in iter_source_files(root):
        rel = path.relative_to(REPO_ROOT).as_posix()
        seen.add(rel)
        if rel not in known or (known[rel] and known[rel] != _sha256(path)):
            return True
    return seen != set(known)


def load_or_build(cache_dir: Path = CACHE_DIR, root: Path = DOCS_ROOT) -> Corpus:
    if is_stale(cache_dir, root):
        corpus = build(root)
        save(corpus, cache_dir)
        return corpus
    return load(cache_dir)


# ----------------------------------------------------------------------- cli --


def _main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="python -m docsearch.corpus")
    ap.add_argument("--rebuild", action="store_true", help="re-extract every file")
    ap.add_argument("--blind-spots", action="store_true", help="list unreadable files")
    ap.add_argument("--stats", action="store_true", help="summarise the index")
    args = ap.parse_args(argv)

    if args.rebuild:
        print(f"reading {DOCS_ROOT.relative_to(REPO_ROOT)}/ ...")
        corpus = build(verbose=True)
        save(corpus)
        print(f"\nwrote {len(corpus)} passages to {CORPUS_PATH.relative_to(REPO_ROOT)}")
    else:
        corpus = load_or_build()

    if args.blind_spots or args.rebuild:
        spots = corpus.blind_spots
        print(f"\nblind spots ({len(spots)}) — present but not searchable as text:")
        for f in spots:
            pages = f"{f.n_pages}pp" if f.n_pages else "?"
            print(f"  {f.source_path}  ({pages}) — {f.reason}")

    if args.stats:
        readable = [f for f in corpus.files if f.readable]
        print(f"\n{len(corpus)} passages from {len(readable)}/{len(corpus.files)} files")
        for coll in ("technical", "presentation"):
            n = sum(1 for p in corpus.passages if p.collection == coll)
            print(f"  {coll:14s} {n:5d} passages")
        for auth in ("project", "protocol", "reference", "teammate-work"):
            n = sum(1 for p in corpus.passages if p.authority == auth)
            print(f"  {auth:14s} {n:5d} passages")

    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
