"""Render a project markdown document to a print-quality PDF.

    .venv/bin/python -m scripts.md2pdf docs/project/RL_LANE_HANDOFF.md [more.md ...]

Markdown -> HTML (`markdown`) -> PDF (`xhtml2pdf`, which is ReportLab underneath).
Pure Python, no browser and no system libraries: headless Chromium was tried first
and does not run in this environment, and WeasyPrint would need pango from
Homebrew. Neither is worth a system dependency for two documents.

The stylesheet is the one the hand-written 2026-09-03 `RL_LANE_HANDOFF.html`
used, so the handoffs keep the look they had. What changed is the direction: the
markdown is the source and the HTML is derived, rather than the HTML being
maintained by hand and drifting from everything else.

Both the `.html` and the `.pdf` land beside the source and are overwritten each
run. **Regenerate after every content change** -- a stale PDF in somebody's
downloads folder is exactly how the previous handoff went out of date.
"""

from __future__ import annotations

import sys
from pathlib import Path

import markdown
from xhtml2pdf import pisa

# xhtml2pdf implements a subset of CSS2 and ignores what it does not know, so
# this is deliberately plain: no flexbox, no grid, no pseudo-selectors.
CSS = """
@page { size: a4 portrait; margin: 16mm 15mm 15mm 15mm; }
body { font-family: "Charter","Georgia",serif; font-size: 10.2pt; line-height: 1.45;
       color: #17181a; }
h1,h2,h3,h4 { font-family: "Helvetica","Arial",sans-serif; }
h1 { font-size: 21pt; margin: 0 0 3mm; }
h2 { font-size: 13.5pt; margin: 8mm 0 2.5mm; padding-bottom: 1.5mm;
     border-bottom: 1.2pt solid #17181a; }
h3 { font-size: 11pt; margin: 5mm 0 1.5mm; }
h4 { font-size: 10pt; margin: 4mm 0 1mm; color: #3a3d42; }
p  { margin: 0 0 2.5mm; }
ul,ol { margin: 0 0 2.8mm; }
li { margin-bottom: 1.1mm; }
code { font-family: "Courier","monospace"; font-size: 8.7pt; background-color: #f2f2f0; }
pre { font-family: "Courier","monospace"; font-size: 8.3pt; background-color: #f7f7f5;
      border: 0.5pt solid #dcdcd6; padding: 3mm; margin: 0 0 3mm; }
table { width: 100%; margin: 0 0 3.5mm; font-size: 8.9pt; }
th { text-align: left; font-size: 8.2pt; color: #55585e;
     border-bottom: 1pt solid #17181a; padding: 1.6mm 2mm 1.2mm; }
td { padding: 1.6mm 2mm; border-bottom: 0.5pt solid #e0e0dc; }
blockquote { border-left: 2.5pt solid #17181a; background-color: #f7f7f5;
             padding: 2.5mm 4mm; margin: 0 0 3.5mm; }
hr { border-top: 0.5pt solid #d5d5cf; margin: 5mm 0; }
a { color: #17181a; }
"""

TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>{css}</style></head><body>{body}</body></html>
"""


def render(source: Path) -> Path:
    body = markdown.markdown(
        source.read_text(encoding="utf-8"),
        extensions=["tables", "fenced_code", "sane_lists"],
    )
    html = TEMPLATE.format(title=source.stem, css=CSS, body=body)
    source.with_suffix(".html").write_text(html, encoding="utf-8")

    pdf_path = source.with_suffix(".pdf")
    with pdf_path.open("wb") as out:
        status = pisa.CreatePDF(html, dest=out, encoding="utf-8")
    if status.err:
        raise SystemExit(f"{source}: {status.err} rendering error(s)")
    return pdf_path


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    for arg in argv:
        source = Path(arg)
        if not source.exists():
            raise SystemExit(f"no such file: {source}")
        pdf = render(source)
        print(f"{source} -> {pdf}  ({pdf.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
