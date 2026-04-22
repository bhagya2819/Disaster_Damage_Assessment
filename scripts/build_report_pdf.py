"""Build a PDF from reports/final_report.md using WeasyPrint.

Why not pandoc? WeasyPrint is already a dependency (needed for the Streamlit
PDF damage report), so we avoid adding a second toolchain. It renders
GitHub-flavoured Markdown via the `markdown-it-py` bridge and outputs a
clean A4 PDF.

Usage:
    python scripts/build_report_pdf.py \
        --input reports/final_report.md \
        --output reports/final_report.pdf

If you'd prefer pandoc (e.g., for LaTeX-quality math or BibTeX references),
use:
    pandoc reports/final_report.md -o reports/final_report.pdf \
        --pdf-engine=weasyprint --toc --metadata title="DDA Report"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# A minimal, print-friendly stylesheet for the Markdown-rendered HTML.
_STYLESHEET = """
@page {
  size: A4;
  margin: 22mm 18mm 22mm 18mm;
  @bottom-center {
    content: "— " counter(page) " —";
    font-size: 9pt;
    color: #777;
  }
}
body {
  font-family: "Charter", Georgia, "Times New Roman", serif;
  font-size: 10.5pt;
  line-height: 1.45;
  color: #111;
}
h1 {
  font-size: 18pt;
  line-height: 1.2;
  margin-top: 0;
  margin-bottom: 6pt;
  page-break-before: always;
}
h1:first-of-type { page-break-before: avoid; }
h2 {
  font-size: 13pt;
  margin-top: 16pt;
  margin-bottom: 4pt;
  border-bottom: 1px solid #999;
  padding-bottom: 2pt;
  page-break-after: avoid;
}
h3 { font-size: 11.5pt; margin-top: 12pt; page-break-after: avoid; }
h4 { font-size: 10.5pt; margin-top: 10pt; }
p  { margin: 4pt 0; text-align: justify; }
code {
  background: #f3f3f3;
  padding: 1pt 3pt;
  border-radius: 2pt;
  font-family: "DejaVu Sans Mono", Menlo, Consolas, monospace;
  font-size: 9.5pt;
}
pre {
  background: #f7f7f7;
  padding: 6pt 8pt;
  border-radius: 3pt;
  border: 1px solid #e0e0e0;
  font-size: 9pt;
  line-height: 1.35;
  overflow-x: auto;
  page-break-inside: avoid;
}
pre code { background: transparent; padding: 0; }
table {
  width: 100%;
  border-collapse: collapse;
  margin: 6pt 0;
  font-size: 9.5pt;
  page-break-inside: avoid;
}
th, td { border: 1px solid #bbb; padding: 3pt 5pt; text-align: left; vertical-align: top; }
th { background: #eee; font-weight: 600; }
img { max-width: 100%; height: auto; }
blockquote {
  border-left: 3px solid #bbb;
  color: #444;
  margin: 6pt 0;
  padding: 2pt 10pt;
  font-style: italic;
}
hr { border: 0; border-top: 1px solid #ccc; margin: 12pt 0; }
ul, ol { margin: 4pt 0 4pt 18pt; }
li { margin: 2pt 0; }
"""


def markdown_to_html(md_text: str) -> str:
    """Render Markdown to an HTML document ready for WeasyPrint."""
    try:
        from markdown_it import MarkdownIt  # noqa: PLC0415
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "markdown-it-py is required. Install via `pip install markdown-it-py`."
        ) from e

    md = MarkdownIt("commonmark", {"html": True, "linkify": True, "typographer": True})
    # Enable tables + fenced code blocks.
    md.enable(["table", "strikethrough"])
    body_html = md.render(md_text)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>DDA Report</title>
<style>{_STYLESHEET}</style>
</head>
<body>
{body_html}
</body>
</html>
"""


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default="reports/final_report.md")
    p.add_argument("--output", default="reports/final_report.pdf")
    args = p.parse_args()

    src = Path(args.input)
    if not src.exists():
        sys.exit(f"Input not found: {src}")

    md_text = src.read_text(encoding="utf-8")
    # Drop YAML frontmatter if present — WeasyPrint would otherwise render it as text.
    if md_text.startswith("---"):
        parts = md_text.split("---", 2)
        if len(parts) >= 3:
            md_text = parts[2].lstrip("\n")

    html = markdown_to_html(md_text)

    try:
        from weasyprint import HTML  # noqa: PLC0415
    except ImportError as e:  # pragma: no cover
        sys.exit(
            "weasyprint is required. Install via `pip install weasyprint`. "
            f"Underlying error: {e}"
        )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    HTML(string=html, base_url=str(src.parent)).write_pdf(str(out))
    size_mb = out.stat().st_size / 1e6
    print(f"Wrote {out} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
