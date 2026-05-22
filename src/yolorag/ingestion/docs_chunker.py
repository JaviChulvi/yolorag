from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_DOCS_ROOT = Path(__file__).resolve().parents[3] / ".." / "ultralytics" / "docs"
DEFAULT_MAX_CHARS = 1800
DEFAULT_OVERLAP_CHARS = 0
MAX_INTACT_CODE_CHARS = 8000
MIN_INDEXABLE_CHARS = 60

FRONT_MATTER_RE = re.compile(r"\A---\s*\n(?P<body>.*?)\n---\s*(?:\n|\Z)", re.DOTALL)
HEADING_RE = re.compile(r"^(?P<marks>#{1,6})\s+(?P<title>.+?)\s*$")
INCLUDE_RE = re.compile(r'{%\s*include\s+["\'](?P<path>[^"\']+)["\']\s*%}')
REFERENCE_SYMBOL_RE = re.compile(r":::\s+(?P<symbol>[A-Za-z_][\w.]+)")
FENCE_RE = re.compile(r"^\s*(?P<marker>`{3,}|~{3,})")


@dataclass(frozen=True)
class DocsChunk:
    chunk_id: str
    doc_id: str
    chunk_index: int
    source_path: str
    url: str | None
    title: str
    headings: list[str]
    kind: str
    text: str
    content: str
    char_count: int
    estimated_tokens: int
    content_hash: str
    reference_symbols: list[str]

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


@dataclass(frozen=True)
class _Document:
    path: Path
    relative_path: Path
    doc_id: str
    url: str | None
    title: str
    frontmatter: dict[str, str]
    body: str
    kind: str


@dataclass(frozen=True)
class _Section:
    headings: list[str]
    body: str


def chunk_markdown_docs(
    docs_root: str | Path = DEFAULT_DOCS_ROOT,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
    include_reference: bool = True,
) -> list[DocsChunk]:
    root = Path(docs_root).expanduser().resolve()
    chunks: list[DocsChunk] = []

    for document in iter_markdown_documents(root, include_reference=include_reference):
        chunks.extend(
            chunk_document(
                document=document,
                max_chars=max_chars,
                overlap_chars=overlap_chars,
            )
        )

    return chunks


def iter_markdown_documents(
    docs_root: Path,
    include_reference: bool = True,
) -> Iterable[_Document]:
    content_root = docs_root / "en"
    if not content_root.exists():
        raise FileNotFoundError(f"Expected docs content at {content_root}")

    for path in sorted(content_root.rglob("*.md")):
        relative_path = path.relative_to(docs_root)
        kind = _document_kind(relative_path)
        if kind == "reference" and not include_reference:
            continue
        yield load_markdown_document(path=path, docs_root=docs_root, relative_path=relative_path)


def load_markdown_document(path: Path, docs_root: Path, relative_path: Path) -> _Document:
    raw = path.read_text(encoding="utf-8")
    frontmatter, body = _split_frontmatter(raw)
    body = _expand_includes(body, docs_root=docs_root)
    body = _normalize_markdown(body)
    title = frontmatter.get("title") or _first_heading(body) or _title_from_path(path)
    kind = _document_kind(relative_path)
    if kind != "reference" and REFERENCE_SYMBOL_RE.search(body):
        kind = "reference"

    return _Document(
        path=path,
        relative_path=relative_path,
        doc_id=_doc_id(relative_path),
        url=_docs_url(relative_path),
        title=title,
        frontmatter=frontmatter,
        body=body,
        kind=kind,
    )


def chunk_document(
    document: _Document,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
) -> list[DocsChunk]:
    chunks: list[DocsChunk] = []
    content_hash = hashlib.sha256(document.body.encode("utf-8")).hexdigest()

    for section in _split_sections(document.body, fallback_title=document.title):
        section_content = section.body.strip()
        if not _is_indexable(section_content):
            continue

        for part in _split_to_size(section_content, max_chars=max_chars, overlap_chars=overlap_chars):
            if not _is_indexable(part):
                continue
            chunk_index = len(chunks)
            reference_symbols = sorted(set(REFERENCE_SYMBOL_RE.findall(part)))
            content = _with_context_prefix(
                document=document,
                headings=section.headings,
                content=part,
            )
            chunk_id = _chunk_id(document.doc_id, chunk_index, content)
            chunks.append(
                DocsChunk(
                    chunk_id=chunk_id,
                    doc_id=document.doc_id,
                    chunk_index=chunk_index,
                    source_path=str(document.relative_path),
                    url=document.url,
                    title=document.title,
                    headings=section.headings,
                    kind=document.kind,
                    text=content,
                    content=part,
                    char_count=len(content),
                    estimated_tokens=_estimate_tokens(content),
                    content_hash=content_hash,
                    reference_symbols=reference_symbols,
                )
            )

    return chunks


def _split_frontmatter(raw: str) -> tuple[dict[str, str], str]:
    match = FRONT_MATTER_RE.match(raw)
    if not match:
        return {}, raw

    frontmatter_text = match.group("body")
    metadata: dict[str, str] = {}
    for line in frontmatter_text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip()] = value.strip().strip('"').strip("'")
    return metadata, raw[match.end() :]


def _expand_includes(body: str, docs_root: Path, seen: set[Path] | None = None) -> str:
    seen = seen or set()

    def replace(match: re.Match[str]) -> str:
        include_path = (docs_root / match.group("path")).resolve()
        if include_path in seen or not include_path.exists():
            return ""
        seen.add(include_path)
        included = include_path.read_text(encoding="utf-8")
        _, included_body = _split_frontmatter(included)
        return _expand_includes(included_body, docs_root=docs_root, seen=seen)

    return INCLUDE_RE.sub(replace, body)


def _normalize_markdown(body: str) -> str:
    body = re.sub(
        r'!!!\s+success\s+"Improvements"\s*\n(?:[ \t]+.*\n?)+',
        "",
        body,
        flags=re.IGNORECASE,
    )
    body = re.sub(r"!\[(?P<alt>[^\]]*)\]\([^)]+\)", r"\g<alt>", body)
    body = re.sub(r"</?(?:br|hr)\s*/?>", "\n", body, flags=re.IGNORECASE)
    body = re.sub(r"<iframe\b.*?</iframe>", "", body, flags=re.IGNORECASE | re.DOTALL)
    body = re.sub(r"<img\b[^>]*>", "", body, flags=re.IGNORECASE)
    body = re.sub(r"</?[A-Za-z][A-Za-z0-9:-]*(?:\s+[^<>]*)?>", "", body)
    body = re.sub(r"{%[^%]+%}", "", body)
    body = body.replace("\r\n", "\n").replace("\r", "\n")
    body = re.sub(r"\n{3,}", "\n\n", body)
    body = "\n".join(line.rstrip() for line in body.splitlines()).strip()
    return re.sub(r"\n[ \t]*\n(?:[ \t]*\n)+", "\n\n", body)


def _split_sections(body: str, fallback_title: str) -> list[_Section]:
    sections: list[_Section] = []
    heading_stack: list[str] = [fallback_title]
    current_lines: list[str] = []
    open_fence: tuple[str, int] | None = None

    def flush() -> None:
        text = "\n".join(current_lines).strip()
        if text:
            sections.append(_Section(headings=[item for item in heading_stack if item], body=text))
        current_lines.clear()

    for line in body.splitlines():
        fence = _fence_marker(line)
        if fence is not None:
            open_fence = _next_fence_state(open_fence=open_fence, fence=fence)
            current_lines.append(line)
            continue

        match = HEADING_RE.match(line) if open_fence is None else None
        if match is not None:
            flush()
            level = len(match.group("marks"))
            title = _clean_heading(match.group("title"))
            heading_stack = heading_stack[: max(level - 1, 0)]
            heading_stack.append(title)
        current_lines.append(line)

    flush()
    return sections


def _split_to_size(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]

    units = _paragraph_units(text)
    chunks: list[str] = []
    current = ""

    for unit in units:
        if _is_balanced_code_unit(unit) and len(unit) <= MAX_INTACT_CODE_CHARS:
            candidate = f"{current}\n\n{unit}".strip() if current else unit
            if current and len(candidate) > max_chars:
                chunks.append(current.strip())
                current = unit
            else:
                current = candidate
            continue

        if len(unit) > max_chars:
            if current:
                chunks.append(current.strip())
                current = ""
            chunks.extend(_split_long_unit(unit, max_chars=max_chars, overlap_chars=overlap_chars))
            continue

        candidate = f"{current}\n\n{unit}".strip() if current else unit
        if len(candidate) <= max_chars:
            current = candidate
            continue

        if current:
            chunks.append(current.strip())
        overlap = _overlap_tail(chunks[-1], overlap_chars) if chunks else ""
        current = f"{overlap}\n\n{unit}".strip() if overlap else unit

    if current:
        chunks.append(current.strip())

    return chunks


def _paragraph_units(text: str) -> list[str]:
    units: list[str] = []
    buffer: list[str] = []
    open_fence: tuple[str, int] | None = None

    for line in text.splitlines():
        fence = _fence_marker(line)
        if fence is not None:
            open_fence = _next_fence_state(open_fence=open_fence, fence=fence)
            buffer.append(line)
            continue
        if open_fence is None and not line.strip():
            if buffer:
                units.append("\n".join(buffer).strip())
                buffer = []
            continue
        buffer.append(line)

    if buffer:
        units.append("\n".join(buffer).strip())
    return units


def _split_long_unit(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            split_at = max(text.rfind("\n", start, end), text.rfind(". ", start, end), text.rfind(" ", start, end))
            if split_at > start + max_chars // 2:
                end = split_at + 1
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap_chars, start + 1)
    return chunks


def _is_balanced_code_unit(text: str) -> bool:
    return ("```" in text or "~~~" in text) and not has_unclosed_fence(text)


def has_unclosed_fence(text: str) -> bool:
    open_fence: tuple[str, int] | None = None
    for line in text.splitlines():
        fence = _fence_marker(line)
        if fence is not None:
            open_fence = _next_fence_state(open_fence=open_fence, fence=fence)
    return open_fence is not None


def _fence_marker(line: str) -> tuple[str, int] | None:
    match = FENCE_RE.match(line)
    if match is None:
        return None
    marker = match.group("marker")
    return marker[0], len(marker)


def _next_fence_state(
    open_fence: tuple[str, int] | None,
    fence: tuple[str, int],
) -> tuple[str, int] | None:
    if open_fence is None:
        return fence
    open_char, open_length = open_fence
    fence_char, fence_length = fence
    if fence_char == open_char and fence_length >= open_length:
        return None
    return open_fence


def _overlap_tail(text: str, overlap_chars: int) -> str:
    if overlap_chars <= 0 or len(text) <= overlap_chars:
        return "" if overlap_chars <= 0 else text
    if "```" in text[-overlap_chars * 2 :]:
        return ""
    tail = text[-overlap_chars:]
    split_at = tail.find("\n")
    return tail[split_at + 1 :].strip() if split_at >= 0 else tail.strip()


def _with_context_prefix(document: _Document, headings: list[str], content: str) -> str:
    lines = [
        f"Title: {document.title}",
        f"URL: {document.url}" if document.url else "",
        f"Source path: {document.relative_path}",
    ]
    lines = [line for line in lines if line]
    if headings:
        lines.append(f"Section: {' > '.join(headings)}")
    lines.append("")
    lines.append(content.strip())
    return "\n".join(lines).strip()


def _is_indexable(text: str) -> bool:
    if REFERENCE_SYMBOL_RE.search(text):
        return True

    meaningful_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if HEADING_RE.match(stripped):
            continue
        if stripped in {"---", "***"}:
            continue
        meaningful_lines.append(stripped)

    meaningful = re.sub(r"[^\w]+", "", " ".join(meaningful_lines))
    return len(meaningful) >= MIN_INDEXABLE_CHARS


def _first_heading(body: str) -> str | None:
    for line in body.splitlines():
        match = HEADING_RE.match(line)
        if match:
            return _clean_heading(match.group("title"))
    return None


def _clean_heading(title: str) -> str:
    title = re.sub(r"`([^`]+)`", r"\1", title)
    title = re.sub(r"\s+#*$", "", title)
    return title.strip()


def _title_from_path(path: Path) -> str:
    return path.stem.replace("-", " ").replace("_", " ").title()


def _document_kind(relative_path: Path) -> str:
    parts = relative_path.parts
    if len(parts) >= 2 and parts[1] == "reference":
        return "reference"
    return "article"


def _doc_id(relative_path: Path) -> str:
    path = relative_path.with_suffix("")
    if path.parts and path.parts[0] == "en":
        path = Path(*path.parts[1:])
    return path.as_posix()


def _docs_url(relative_path: Path) -> str | None:
    if not relative_path.parts or relative_path.parts[0] != "en":
        return None
    doc_path = relative_path.with_suffix("")
    parts = list(doc_path.parts[1:])
    if parts == ["index"]:
        parts = []
    elif parts and parts[-1] == "index":
        parts = parts[:-1]
    suffix = "/".join(parts)
    return f"https://docs.ultralytics.com/{suffix}/" if suffix else "https://docs.ultralytics.com/"


def _chunk_id(doc_id: str, chunk_index: int, content: str) -> str:
    digest = hashlib.sha1(content.encode("utf-8")).hexdigest()[:10]
    return f"{doc_id}#{chunk_index:04d}-{digest}"


def _estimate_tokens(text: str) -> int:
    return max(1, len(re.findall(r"\w+|[^\w\s]", text)))
