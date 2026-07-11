"""
fetch.py — Wikipedia article fetcher and HTML cleaner.

Fetches the Wikipedia "India" article for each of the four languages,
strips HTML, citation markers, and table markup, then saves clean plain
text to tokenizer/data/raw/{lang}.txt.

Content extracted (in document order):
  - <h2>, <h3>, <h4> section headings (rich vocabulary signal)
  - <p> paragraph text (main prose)
  - <li> list items (additional facts and terms)

Excluded:
  - Tables, navigation boxes, reference lists, infoboxes
  - Citation markers [1], [2], [note X]
  - Edit-section links, image captions

Usage:
    python tokenizer/src/fetch.py
"""

import re
import os
import requests
from bs4 import BeautifulSoup

ARTICLES = {
    "en": "https://en.wikipedia.org/wiki/India",
    "hi": "https://hi.wikipedia.org/wiki/%E0%A4%AD%E0%A4%BE%E0%A4%B0%E0%A4%A4",
    "te": "https://te.wikipedia.org/wiki/%E0%B0%AD%E0%B0%BE%E0%B0%B0%E0%B0%A4%E0%B0%A6%E0%B1%87%E0%B0%B6%E0%B0%82",
    "sa": "https://sa.wikipedia.org/wiki/%E0%A4%AD%E0%A4%BE%E0%A4%B0%E0%A4%A4%E0%A4%AE%E0%A5%8D",
}

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

# Tags whose *content* is noise — strip entirely before text extraction
_NOISE_TAGS = [
    "table", "sup", "sub",
    "div.navbox", "div.reflist", "div.refbegin",
    "div.hatnote", "div.thumb", "div.infobox",
    "div.toc", "div.noprint",
    "span.mw-editsection",
    "span.reference",
    "ol.references",
    "figure",
]


def fetch_wikipedia_text(url: str) -> str:
    """
    Fetch a Wikipedia article and extract clean paragraph + heading + list text.

    Extraction strategy: pull content tags in document order from the article
    body (div#mw-content-text), skipping tables, navboxes, and references.
    This gives richer vocabulary coverage than paragraphs alone, leading to
    more unique words and more training signal for BPE.
    """
    headers = {"User-Agent": "MultilingualBPETokenizer/1.0 (educational project)"}
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    # Remove noisy elements first
    for selector in _NOISE_TAGS:
        tag_name, *classes = selector.split(".")
        find_kwargs = {"class_": classes[0]} if classes else {}
        for el in soup.find_all(tag_name, **find_kwargs):
            el.decompose()

    # Also remove all table elements broadly (Wikipedia infoboxes etc.)
    for el in soup.find_all("table"):
        el.decompose()

    # Find the main article body
    body = soup.find("div", id="mw-content-text") or soup.find("div", id="bodyContent") or soup

    # Collect text from content tags in document order
    content_tags = body.find_all(["h2", "h3", "h4", "p", "li"])

    lines = []
    for tag in content_tags:
        text = tag.get_text(separator=" ", strip=True)
        if not text or len(text) < 3:
            continue
        # Skip Wikipedia section edit links that sneak through
        if re.match(r"^\[edit\]$", text, re.IGNORECASE):
            continue
        lines.append(text)

    text = "\n".join(lines)

    # Strip citation markers like [1], [2], [note 3], [citation needed]
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"\[note \d+\]", "", text)
    text = re.sub(r"\[citation needed\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\[edit\]", "", text, flags=re.IGNORECASE)

    # Remove bare coordinates / coordinates strings (appear in some articles)
    text = re.sub(r"\d+°\d+′[NS]\s+\d+°\d+′[EW]", "", text)

    # Collapse multiple whitespace / newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)

    return text.strip()


def fetch_all():
    os.makedirs(RAW_DIR, exist_ok=True)
    for lang, url in ARTICLES.items():
        print(f"Fetching {lang} from {url} …")
        text = fetch_wikipedia_text(url)
        out_path = os.path.join(RAW_DIR, f"{lang}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)
        words = len(text.split())
        print(f"  Saved {len(text):,} chars | {words:,} words → {out_path}")


if __name__ == "__main__":
    fetch_all()
