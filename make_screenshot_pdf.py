"""Build ``screenshots.pdf`` (screenshots + explanations) for the assignment submission.

Input
-----
* Images in ``screenshots/`` (``.png``/``.jpg``).
* Optional ``screenshots/captions.json`` — a list of
  ``{"file": "01_home.png", "title": "...", "caption": "..."}`` in the order
  they should appear (see ``screenshots/captions.example.json``). Without it,
  all images are used in filename order with the filename as title.

Output
------
1. ``screenshots.html`` — always written; images are embedded, Thai text renders
   perfectly. Open it in a browser and use *Print → Save as PDF* if needed.
2. ``screenshots.pdf`` — produced automatically with headless Google Chrome /
   Chromium / Edge when one is installed (best Thai rendering). Otherwise a
   Pillow fallback is used (Thai vowel/tone marks may be slightly misplaced
   because Pillow lacks complex text shaping without libraqm).

Usage::

    python make_screenshot_pdf.py
"""

from __future__ import annotations

import base64
import html
import json
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SHOT_DIR = ROOT / "screenshots"
HTML_OUT = ROOT / "screenshots.html"
PDF_OUT = ROOT / "screenshots.pdf"
IMAGE_EXT = {".png", ".jpg", ".jpeg"}
TITLE = "ภาพหน้าจอ: ผู้ช่วยผู้เชี่ยวชาญด้านการเกษตรและโรคพืช (RAG Chatbot)"

BROWSER_CANDIDATES = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    "google-chrome",
    "chromium",
    "chromium-browser",
    "msedge",
]
THAI_FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Tahoma.ttf",
    "/usr/share/fonts/truetype/tlwg/Garuda.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansThai-Regular.ttf",
    "C:/Windows/Fonts/tahoma.ttf",
]


def load_entries() -> list[dict[str, str]]:
    """Return the ordered list of screenshots with titles and captions."""
    captions_file = SHOT_DIR / "captions.json"
    if captions_file.exists():
        entries = json.loads(captions_file.read_text(encoding="utf-8"))
    else:
        entries = [
            {"file": p.name, "title": p.stem, "caption": ""}
            for p in sorted(SHOT_DIR.iterdir())
            if p.suffix.lower() in IMAGE_EXT
        ]
    missing = [e["file"] for e in entries if not (SHOT_DIR / e["file"]).exists()]
    if missing:
        sys.exit(f"Missing image(s) in screenshots/: {', '.join(missing)}")
    if not entries:
        sys.exit("No screenshots found. Put PNG/JPG files in screenshots/ first (see docs/screenshot_guide.md).")
    return entries


def write_html(entries: list[dict[str, str]]) -> None:
    """Write a print-friendly HTML file with embedded images."""
    sections = []
    for i, e in enumerate(entries, 1):
        path = SHOT_DIR / e["file"]
        mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        sections.append(
            f"""<section>
  <h2>{i}. {html.escape(e.get("title", ""))}</h2>
  <img src="data:{mime};base64,{data}" alt="{html.escape(e["file"])}">
  <p>{html.escape(e.get("caption", "")).replace(chr(10), "<br>")}</p>
</section>"""
        )
    HTML_OUT.write_text(
        f"""<!doctype html>
<html lang="th"><head><meta charset="utf-8"><title>{html.escape(TITLE)}</title>
<style>
  body {{ font-family: "Sarabun", "Tahoma", "Noto Sans Thai", sans-serif; margin: 24px; color: #1b2a1b; }}
  h1 {{ font-size: 20px; }} h2 {{ font-size: 16px; color: #2e7d32; }}
  section {{ page-break-after: always; }}
  img {{ max-width: 100%; max-height: 70vh; border: 1px solid #ccc; }}
  p {{ font-size: 13px; line-height: 1.6; }}
</style></head>
<body><h1>{html.escape(TITLE)}</h1>
{chr(10).join(sections)}
</body></html>""",
        encoding="utf-8",
    )
    print(f"Wrote {HTML_OUT.name}")


def pdf_with_browser() -> bool:
    """Print the HTML to PDF with a headless Chromium-based browser, if available."""
    for candidate in BROWSER_CANDIDATES:
        exe = candidate if Path(candidate).exists() else shutil.which(candidate)
        if not exe:
            continue
        cmd = [exe, "--headless", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={PDF_OUT}", HTML_OUT.as_uri()]
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=120)
        except (subprocess.SubprocessError, OSError):
            continue
        if PDF_OUT.exists():
            print(f"Wrote {PDF_OUT.name} (via {Path(exe).name})")
            return True
    return False


def pdf_with_pillow(entries: list[dict[str, str]]) -> None:
    """Fallback: one A4 page per screenshot with the caption drawn underneath."""
    from PIL import Image, ImageDraw, ImageFont

    font_path = next((f for f in THAI_FONT_CANDIDATES if Path(f).exists()), None)
    title_font = ImageFont.truetype(font_path, 34) if font_path else ImageFont.load_default()
    body_font = ImageFont.truetype(font_path, 26) if font_path else ImageFont.load_default()

    page_w, page_h, margin = 1654, 2339, 90  # A4 @ 200 dpi
    pages = []
    for i, e in enumerate(entries, 1):
        page = Image.new("RGB", (page_w, page_h), "white")
        draw = ImageDraw.Draw(page)
        draw.text((margin, margin), f"{i}. {e.get('title', '')}", fill="#2e7d32", font=title_font)
        shot = Image.open(SHOT_DIR / e["file"]).convert("RGB")
        shot.thumbnail((page_w - 2 * margin, int(page_h * 0.62)))
        page.paste(shot, (margin, margin + 70))
        y = margin + 70 + shot.height + 40
        for paragraph in e.get("caption", "").split("\n"):
            for line in textwrap.wrap(paragraph, width=70) or [""]:
                draw.text((margin, y), line, fill="black", font=body_font)
                y += 40
        pages.append(page)
    pages[0].save(PDF_OUT, save_all=True, append_images=pages[1:], resolution=200)
    print(f"Wrote {PDF_OUT.name} (Pillow fallback — check Thai rendering, or print screenshots.html instead)")


def main() -> None:
    entries = load_entries()
    write_html(entries)
    if not pdf_with_browser():
        pdf_with_pillow(entries)


if __name__ == "__main__":
    main()
