"""Parse supported document formats into citation-friendly sections."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from bs4 import BeautifulSoup


@dataclass(frozen=True, slots=True)
class Section:
    heading_path: str | None
    text: str


@dataclass(frozen=True, slots=True)
class ParsedDocument:
    title: str | None
    doc_type: str
    sections: tuple[Section, ...]


SUPPORTED_SUFFIXES = {".html", ".htm", ".md", ".markdown", ".pdf", ".txt"}
_SPACE = re.compile(r"[ \t]+")
_BLANKS = re.compile(r"\n{3,}")


def _clean(text: str) -> str:
    lines = [_SPACE.sub(" ", line).strip() for line in text.splitlines()]
    return _BLANKS.sub("\n\n", "\n".join(lines)).strip()


def _markdown(path: Path) -> ParsedDocument:
    sections: list[Section] = []
    headings: list[str] = []
    body: list[str] = []
    title: str | None = None

    def flush() -> None:
        text = _clean("\n".join(body))
        if text:
            sections.append(Section(" > ".join(headings) or None, text))
        body.clear()

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = re.match(r"^(#{1,6})\s+(.+?)\s*#*\s*$", line)
        if not match:
            body.append(line)
            continue
        flush()
        level = len(match.group(1))
        heading = match.group(2).strip()
        if title is None and level == 1:
            title = heading
        headings[level - 1 :] = [heading]
    flush()
    return ParsedDocument(title or path.stem, "markdown", tuple(sections))


def _html(path: Path) -> ParsedDocument:
    soup = BeautifulSoup(
        path.read_text(encoding="utf-8", errors="replace"), "html.parser"
    )
    for element in soup(["script", "style", "noscript"]):
        element.decompose()

    title = _clean(soup.title.get_text(" ")) if soup.title else path.stem
    headings: list[str] = []
    sections: list[Section] = []
    body: list[str] = []

    def flush() -> None:
        text = _clean("\n".join(body))
        if text:
            sections.append(Section(" > ".join(headings) or None, text))
        body.clear()

    root = soup.body or soup
    for element in root.find_all(
        ["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "pre"]
    ):
        text = _clean(element.get_text(" "))
        if not text:
            continue
        if element.name and element.name.startswith("h"):
            flush()
            level = int(element.name[1])
            headings[level - 1 :] = [text]
        else:
            body.append(text)
    flush()
    return ParsedDocument(title, "html", tuple(sections))


def _pdf(path: Path) -> ParsedDocument:
    import fitz

    sections: list[Section] = []
    with fitz.open(path) as document:
        title = _clean(document.metadata.get("title", "")) or path.stem
        for page_number, page in enumerate(document, start=1):
            text = _clean(page.get_text("text"))
            if text:
                sections.append(Section(f"Page {page_number}", text))
    return ParsedDocument(title, "pdf", tuple(sections))


def parse_document(path: Path) -> ParsedDocument:
    """Parse one supported file, preserving structure needed for citations."""
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"unsupported document type: {suffix or '<none>'}")
    if suffix in {".md", ".markdown"}:
        return _markdown(path)
    if suffix in {".html", ".htm"}:
        return _html(path)
    if suffix == ".pdf":
        return _pdf(path)

    text = _clean(path.read_text(encoding="utf-8", errors="replace"))
    sections = (Section(None, text),) if text else ()
    return ParsedDocument(path.stem, "text", sections)
