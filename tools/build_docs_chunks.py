#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


MAX_CHUNK_CHARS = 900
SKIP_DOCS = {
    "iso-build-recipe",
    "release-pipeline",
    "omarchy-heritage",
    "branding",
    "vision",
    "ui-patterns",
}
SKILL_DOCS = ("gui", "troubleshoot", "bar", "plugins", "wm", "feature")


@dataclass(frozen=True)
class SourcePage:
    path: Path
    source: str
    page_id: str
    url_base: str | None


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def heading_slug(value: str) -> str:
    value = re.sub(r"`([^`]*)`", r"\1", value).lower()
    value = re.sub(r"[^a-z0-9\s-]", "", value)
    return re.sub(r"[\s-]+", "-", value).strip("-")


def _frontmatter(text: str) -> tuple[dict[str, str], str]:
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---", 4)
    if end < 0:
        return {}, text
    fields: dict[str, str] = {}
    for line in text[4:end].splitlines():
        key, separator, value = line.partition(":")
        if separator and key.strip() and value.strip():
            fields[key.strip()] = value.strip().strip('"\'')
    body_start = end + len("\n---")
    return fields, text[body_start:].lstrip("\n")


def _strip_jsx(text: str) -> str:
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    text = re.sub(r"^\s*(?:import|export)\s+.*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"</?[A-Za-z][^>]*>", "", text, flags=re.DOTALL)
    text = re.sub(r"\{\s*/\*.*?\*/\s*\}", "", text, flags=re.DOTALL)
    text = re.sub(r"^\s*\{[^{}]*\}\s*$", "", text, flags=re.MULTILINE)
    return text


def _plain_line(line: str) -> str | None:
    stripped = line.strip()
    if re.fullmatch(r"\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?", stripped):
        return None
    if stripped.startswith("|") and stripped.endswith("|"):
        cells = [cell.strip() for cell in re.split(r"(?<!\\)\|", stripped[1:-1])]
        return " | ".join(cell.replace("\\|", "|") for cell in cells)
    line = re.sub(r"!\[([^]]*)]\([^)]+\)", r"\1", line)
    line = re.sub(r"\[([^]]+)]\([^)]+\)", r"\1", line)
    line = re.sub(r"\[([^]]+)]\[[^]]*]", r"\1", line)
    line = re.sub(r"^\s*>\s?", "", line)
    line = line.replace("**", "").replace("__", "")
    return line.rstrip()


def _plain_text(text: str) -> str:
    text = _strip_jsx(text)
    lines: list[str] = []
    in_code = False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            in_code = not in_code
            lines.append(line.rstrip())
            continue
        if in_code:
            lines.append(line.rstrip())
            continue
        plain = _plain_line(line)
        if plain is not None:
            lines.append(plain)
    return "\n".join(lines).strip()


def _sections(body: str, page_title: str) -> list[tuple[str, str]]:
    matches = list(re.finditer(r"^(#{1,6})\s+(.+?)\s*$", body, flags=re.MULTILINE))
    sections: list[tuple[str, str]] = []
    first_heading = matches[0].start() if matches else len(body)
    intro = body[:first_heading].strip()
    if intro:
        sections.append(("Overview", intro))
    for index, match in enumerate(matches):
        level = len(match.group(1))
        heading = _plain_text(match.group(2)).strip()
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        section_text = body[start:end].strip()
        if level == 1 and heading.casefold() == page_title.casefold():
            heading = "Overview"
        if section_text:
            sections.append((heading or "Overview", section_text))
    if not matches and body.strip() and not sections:
        sections.append(("Overview", body.strip()))
    return sections


def _paragraphs(text: str) -> list[str]:
    paragraphs: list[str] = []
    current: list[str] = []
    in_code = False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            current.append(line)
            in_code = not in_code
            continue
        if not line.strip() and not in_code:
            if current:
                paragraphs.append("\n".join(current).strip())
                current = []
            continue
        current.append(line)
    if current:
        paragraphs.append("\n".join(current).strip())
    return [paragraph for paragraph in paragraphs if paragraph]


def _split_long(value: str, limit: int) -> list[str]:
    if len(value) <= limit:
        return [value]
    pieces: list[str] = []
    current = ""
    tokens = re.findall(r"\S+\s*", value)
    for token in tokens:
        token = token.rstrip()
        separator = " " if current else ""
        if len(current) + len(separator) + len(token) <= limit:
            current += separator + token
            continue
        if current:
            pieces.append(current)
            current = ""
        while len(token) > limit:
            pieces.append(token[:limit])
            token = token[limit:]
        current = token
    if current:
        pieces.append(current)
    return pieces


def _pack(text: str, limit: int = MAX_CHUNK_CHARS) -> list[str]:
    units: list[str] = []
    for paragraph in _paragraphs(text):
        units.extend(_split_long(paragraph, limit))
    chunks: list[str] = []
    current = ""
    for unit in units:
        candidate = f"{current}\n\n{unit}" if current else unit
        if len(candidate) <= limit:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = unit
    if current:
        chunks.append(current)
    return chunks


def _docs_site_base(docs: Path) -> str:
    for name in ("docs.json", "mint.json"):
        path = docs / name
        if not path.is_file():
            continue
        try:
            config = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        canonical = (
            config.get("seo", {}).get("metatags", {}).get("canonical")
            if isinstance(config, dict)
            else None
        )
        if isinstance(canonical, str) and canonical.startswith("https://"):
            return canonical.rstrip("/")
    return "https://docs.ryoku.dev"


def _source_pages(docs: Path, repo: Path) -> Iterable[SourcePage]:
    docs_dir = docs / "docs"
    site_base = _docs_site_base(docs)
    for path in sorted(docs_dir.glob("*.mdx")):
        if path.stem in SKIP_DOCS:
            continue
        yield SourcePage(
            path=path,
            source=path.relative_to(docs).as_posix(),
            page_id=f"docs/{path.stem}",
            url_base=f"{site_base}/docs/{path.stem}",
        )

    skill_dir = repo / "ryoku/rashin/skills/ryoku"
    for name in SKILL_DOCS:
        path = skill_dir / f"{name}.md"
        if path.is_file():
            source = path.relative_to(repo).as_posix()
            yield SourcePage(path=path, source=source, page_id=source, url_base=None)

    wiki_dir = repo / "ryoku/rashin/wiki"
    if wiki_dir.is_dir():
        for path in sorted(wiki_dir.glob("*.md")):
            source = path.relative_to(repo).as_posix()
            yield SourcePage(path=path, source=source, page_id=source, url_base=None)


def collect_chunks(docs: Path, repo: Path) -> list[dict[str, object]]:
    docs = docs.resolve()
    repo = repo.resolve()
    chunks: list[dict[str, object]] = []
    used_ids: dict[str, int] = {}
    for page in _source_pages(docs, repo):
        fields, body = _frontmatter(page.path.read_text(encoding="utf-8"))
        first_h1 = re.search(r"^#\s+(.+?)\s*$", body, flags=re.MULTILINE)
        page_title = fields.get("title") or (
            _plain_text(first_h1.group(1)) if first_h1 else page.path.stem.replace("-", " ").title()
        )
        for heading, section in _sections(body, page_title):
            plain = _plain_text(section)
            if not plain:
                continue
            slug = heading_slug(heading) or "overview"
            base_id = f"doc:{page.page_id}#{slug}"
            url = f"{page.url_base}#{slug}" if page.url_base else None
            for text in _pack(plain):
                used_ids[base_id] = used_ids.get(base_id, 0) + 1
                number = used_ids[base_id]
                chunk_id = base_id if number == 1 else f"{base_id}:{number}"
                chunks.append(
                    {
                        "id": chunk_id,
                        "title": f"{page_title} > {heading}",
                        "url": url,
                        "text": text,
                        "source": page.source,
                    }
                )
    return sorted(chunks, key=lambda chunk: str(chunk["id"]))


def build_docs_chunks(docs: Path, repo: Path) -> dict[str, object]:
    docs = docs.resolve()
    repo = repo.resolve()
    return {
        "generated_from": {
            "docs_commit": _git(docs, "rev-parse", "--short", "HEAD"),
            "repo_commit": _git(repo, "rev-parse", "--short", "HEAD"),
        },
        "chunks": collect_chunks(docs, repo),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Nero's Ryoku docs chunks")
    parser.add_argument("--docs", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = build_docs_chunks(args.docs, args.repo)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
