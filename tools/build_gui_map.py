#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Iterable


PROVIDERS = ("hyprland", "niri", "wayfire")
AUXILIARY_SCHEMA_SECTIONS = {
    "GlobalPage.js": "global",
    "WifiTab.js": "connections",
    "BluetoothTab.js": "connections",
    "HotspotTab.js": "connections",
    "PluginPlacementEditor.js": "plugins",
}


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()

def _stable_version(repo: Path) -> str:
    version = (repo / "VERSION").read_text(encoding="utf-8").strip()
    if not version:
        raise ValueError(f"empty VERSION in {repo}")
    return version.removeprefix("v")


def _expression_after(text: str, marker: str, opener: str) -> str:
    marker_at = text.find(marker)
    if marker_at < 0:
        raise ValueError(f"missing {marker}")
    start = text.find(opener, marker_at + len(marker))
    if start < 0:
        raise ValueError(f"missing {opener!r} after {marker}")

    pairs = {"[": "]", "{": "}", "(": ")"}
    stack: list[str] = []
    quote: str | None = None
    escaped = False
    line_comment = False
    block_comment = False
    index = start
    while index < len(text):
        char = text[index]
        following = text[index + 1] if index + 1 < len(text) else ""
        if line_comment:
            if char == "\n":
                line_comment = False
            index += 1
            continue
        if block_comment:
            if char == "*" and following == "/":
                block_comment = False
                index += 2
            else:
                index += 1
            continue
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            index += 1
            continue
        if char in ('"', "'", "`"):
            quote = char
            index += 1
            continue
        if char == "/" and following == "/":
            line_comment = True
            index += 2
            continue
        if char == "/" and following == "*":
            block_comment = True
            index += 2
            continue
        if char in pairs:
            stack.append(pairs[char])
        elif stack and char == stack[-1]:
            stack.pop()
            if not stack:
                return text[start : index + 1]
        index += 1
    raise ValueError(f"unterminated expression after {marker}")


def _node_json(expression: str, declarations: str = "") -> Any:
    script = f"{declarations}\nconst value = ({expression});\nprocess.stdout.write(JSON.stringify(value));"
    result = subprocess.run(
        ["node", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def _hub_catalogue(hub_path: Path) -> tuple[list[dict[str, Any]], dict[str, str]]:
    text = hub_path.read_text(encoding="utf-8")
    groups = _node_json(_expression_after(text, "readonly property var groups:", "["))
    imports = {
        alias: filename
        for filename, alias in re.findall(
            r'^\s*import\s+"schema/([^"/]+\.js)"\s+as\s+(\w+)\s*$',
            text,
            flags=re.MULTILINE,
        )
    }
    section_expression = _expression_after(
        text, "readonly property var sectionRows:", "("
    )
    declarations = "\n".join(
        f"const {alias} = {{rows: {json.dumps(alias)}}};" for alias in imports
    )
    sections = _node_json(section_expression, declarations)
    section_files = {
        str(section): imports[str(alias)]
        for section, alias in sections.items()
        if str(alias) in imports
    }
    return groups, section_files


def _schema_rows(path: Path) -> list[dict[str, Any]] | None:
    source = path.read_text(encoding="utf-8")
    source = re.sub(r"^\s*\.pragma\s+library\s*$", "", source, flags=re.MULTILINE)
    if not re.search(r"\b(?:var|let|const)\s+rows\s*=", source):
        return None
    script = (
        "const vm = require('vm');\n"
        f"const source = {json.dumps(source)};\n"
        "const context = {}; vm.createContext(context);\n"
        "vm.runInContext(source, context, {timeout: 1000});\n"
        "process.stdout.write(JSON.stringify(context.rows));"
    )
    result = subprocess.run(
        ["node", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )
    value = json.loads(result.stdout)
    if not isinstance(value, list):
        raise ValueError(f"{path} rows is not an array")
    return [row for row in value if isinstance(row, dict)]


def _capability_names(repo: Path) -> tuple[dict[str, str], dict[str, set[str]]]:
    caps_path = repo / "ryoku/wm/caps.go"
    caps_text = caps_path.read_text(encoding="utf-8")
    constants = {
        symbol: value
        for symbol, value in re.findall(
            r"\b(Cap[A-Za-z0-9]+)\s+(?:Capability\s+)?=\s*\"([^\"]+)\"",
            caps_text,
        )
    }
    providers: dict[str, set[str]] = {}
    for provider in PROVIDERS:
        path = repo / "ryoku/wm" / provider / "caps.go"
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        try:
            manifest = _expression_after(text, "var capsManifest", "{")
        except ValueError:
            continue
        providers[provider] = {
            constants[symbol]
            for symbol in re.findall(r"\bwm\.(Cap[A-Za-z0-9]+)\b", manifest)
            if symbol in constants
        }
    return constants, providers


def _values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value if item]
    return [str(value)]


def _compositors_for_caps(
    caps: Iterable[str], providers: dict[str, set[str]]
) -> tuple[list[str], list[str]]:
    wanted = [cap for cap in caps if cap]
    if not wanted:
        return ["any"], []
    known = set().union(*providers.values()) if providers else set()
    if any(cap not in known for cap in wanted):
        return ["any"], wanted
    supported = [
        provider
        for provider in PROVIDERS
        if provider in providers and all(cap in providers[provider] for cap in wanted)
    ]
    if not supported:
        return ["any"], wanted
    return supported, []


def _words(value: Any) -> list[str]:
    if isinstance(value, list):
        source = " ".join(str(item) for item in value)
    else:
        source = str(value or "")
    source = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", source)
    return [word.lower() for word in re.findall(r"[A-Za-z0-9]+", source)]


def _unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        normalized = value.strip()
        if normalized and normalized.lower() not in seen:
            seen.add(normalized.lower())
            result.append(normalized)
    return result


def _repo_source(repo: Path, path: Path) -> str:
    return path.relative_to(repo).as_posix()


def _split_table_row(line: str) -> list[str]:
    value = line.strip()
    if value.startswith("|"):
        value = value[1:]
    if value.endswith("|"):
        value = value[:-1]
    cells = re.split(r"(?<!\\)\|", value)
    return [cell.strip().replace("\\|", "|") for cell in cells]


def _strip_markdown(value: str) -> str:
    value = re.sub(r"`([^`]*)`", r"\1", value)
    value = re.sub(r"\[([^]]+)]\([^)]+\)", r"\1", value)
    value = value.replace("**", "").replace("__", "")
    return " ".join(value.split())


def _table_under(text: str, heading: str) -> list[list[str]]:
    match = re.search(
        rf"^##\s+{re.escape(heading)}\s*$([\s\S]*?)(?=^##\s+|\Z)",
        text,
        flags=re.MULTILINE,
    )
    if not match:
        return []
    rows = [
        _split_table_row(line)
        for line in match.group(1).splitlines()
        if line.lstrip().startswith("|")
    ]
    if len(rows) < 2:
        return []
    return [row for row in rows[2:] if row]


def _guide_data(path: Path) -> tuple[dict[str, str], list[dict[str, Any]]]:
    if not path.is_file():
        return {}, []
    text = path.read_text(encoding="utf-8")
    descriptions: dict[str, str] = {}
    for row in _table_under(text, "The Hub pages"):
        if len(row) >= 4:
            descriptions[_strip_markdown(row[1])] = _strip_markdown(row[3])

    surfaces: list[dict[str, Any]] = []
    for row in _table_under(text, "Surfaces outside the Hub"):
        if len(row) < 4:
            continue
        intent, title, how, command = (_strip_markdown(cell) for cell in row[:4])
        slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
        surfaces.append(
            {
                "id": f"surface:{slug}",
                "kind": "surface",
                "title": title,
                "path": how,
                "open": how or None,
                "desc": f"{intent}. Behind it: {command}",
                "keywords": _unique(_words(intent) + _words(title)),
                "compositors": ["any"],
                "advanced": False,
                "source": path.as_posix(),
            }
        )
    return descriptions, surfaces


def _summarize_page(rows: list[dict[str, Any]], fallback: str) -> str:
    tabs = _unique(str(row.get("tab", "")) for row in rows)
    groups = _unique(str(row.get("group", "")) for row in rows)
    details: list[str] = []
    if tabs:
        details.append("Tabs: " + ", ".join(tabs))
    if groups:
        details.append("Groups: " + ", ".join(groups))
    summary = ". ".join(details)
    if fallback and summary:
        return f"{fallback}. {summary}."
    if fallback:
        return fallback
    if summary:
        return summary + "."
    return "Open this Ryoku Hub page."


def _setting_id(section: str, row: dict[str, Any], ordinal: int) -> str:
    key = str(row.get("key") or "").strip()
    if key:
        return f"setting:{section}/{key}"
    label = re.sub(r"[^a-z0-9]+", "-", str(row.get("label", "")).lower()).strip("-")
    return f"setting:{section}/{label or ordinal}"


def _format_chord(chord: str) -> str:
    casing = {"SUPER": "Super", "CTRL": "Ctrl", "SHIFT": "Shift", "ALT": "Alt"}
    return "+".join(casing.get(part.strip(), part.strip()) for part in chord.split("+"))


def _go_fields(body: str) -> dict[str, str]:
    return {
        key: value
        for key, value in re.findall(
            r"\b(ID|Category|Label|Hint|Chord)\s*:\s*\"((?:\\.|[^\"\\])*)\"",
            body,
        )
    }


def _top_level_objects(expression: str) -> list[str]:
    objects: list[str] = []
    depth = 0
    start: int | None = None
    quote: str | None = None
    escaped = False
    for index, char in enumerate(expression):
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in ('"', "'"):
            quote = char
            continue
        if char == "{":
            depth += 1
            if depth == 2:
                start = index + 1
        elif char == "}":
            if depth == 2 and start is not None:
                objects.append(expression[start:index])
                start = None
            depth -= 1
    return objects


def _keybind_docs_entries(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    entries: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    used_ids: dict[str, int] = {}
    for line in text.splitlines():
        if not line.lstrip().startswith("|"):
            continue
        cells = _split_table_row(line)
        if len(cells) < 2:
            continue
        chord = _strip_markdown(cells[0]).replace("Mod", "Super")
        title = _strip_markdown(cells[1]).rstrip(".")
        if not chord or not title or chord == "Binding" or set(chord) <= {"-", ":", " "}:
            continue
        chord = re.sub(r"\s*\+\s*", "+", chord)
        signature = (chord.casefold(), title.casefold())
        if signature in seen:
            continue
        seen.add(signature)
        slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-") or "bind"
        used_ids[slug] = used_ids.get(slug, 0) + 1
        suffix = f":{used_ids[slug]}" if used_ids[slug] > 1 else ""
        entries.append(
            {
                "id": f"keybind:docs-{slug}{suffix}",
                "kind": "keybind",
                "title": title,
                "path": chord,
                "open": None,
                "desc": title,
                "keywords": _unique(_words(title)),
                "compositors": ["any"],
                "advanced": False,
                "source": path.as_posix(),
            }
        )
    return entries


def _keybind_entries(repo: Path) -> list[dict[str, Any]]:
    path = repo / "ryoku/wm/binds.go"
    entries: list[dict[str, Any]] = []
    if path.is_file():
        text = path.read_text(encoding="utf-8")
        try:
            catalogue = _expression_after(text, "var catalogue", "{")
        except ValueError:
            catalogue = ""
        for body in _top_level_objects(catalogue):
            fields = _go_fields(body)
            if not {"ID", "Label", "Chord"}.issubset(fields):
                continue
            title = bytes(fields["Label"], "utf-8").decode("unicode_escape")
            hint = bytes(fields.get("Hint", ""), "utf-8").decode("unicode_escape")
            chord = bytes(fields["Chord"], "utf-8").decode("unicode_escape")
            entries.append(
                {
                    "id": f"keybind:{fields['ID']}",
                    "kind": "keybind",
                    "title": title,
                    "path": _format_chord(chord),
                    "open": None,
                    "desc": hint or title,
                    "keywords": _unique(
                        _words(title) + _words(hint) + _words(fields.get("Category", ""))
                    ),
                    "compositors": ["any"],
                    "advanced": False,
                    "source": _repo_source(repo, path),
                }
            )
    if entries:
        return entries
    docs_path = Path("/home/nero/.cache/nero-work/ryoku-docs/docs/keybindings.mdx")
    if docs_path.is_file():
        entries = _keybind_docs_entries(docs_path)
    if not entries:
        raise ValueError("could not parse keybinds from binds.go or keybindings.mdx")
    return entries


def collect_entries(repo: Path) -> list[dict[str, Any]]:
    repo = repo.resolve()
    hub_path = repo / "ryoku/hub/quickshell/Hub.qml"
    groups, section_files = _hub_catalogue(hub_path)
    _, providers = _capability_names(repo)
    guide_path = repo / "ryoku/rashin/skills/ryoku/gui.md"
    guide_descriptions, surfaces = _guide_data(guide_path)
    for surface in surfaces:
        surface["source"] = _repo_source(repo, guide_path)

    pages: list[dict[str, Any]] = []
    page_by_key: dict[str, dict[str, Any]] = {}
    for group in groups:
        for item in group.get("items", []):
            if not isinstance(item, dict) or not item.get("key") or not item.get("name"):
                continue
            page_by_key[str(item["key"])] = item

    rows_by_file: dict[str, list[dict[str, Any]]] = {}
    schema_dir = hub_path.parent / "schema"
    for path in sorted(schema_dir.glob("*.js")):
        rows = _schema_rows(path)
        if rows is not None:
            rows_by_file[path.name] = rows

    rows_by_section: dict[str, list[dict[str, Any]]] = {}
    mapped_files = set(section_files.values())
    files_for_section = [(section, filename) for section, filename in section_files.items()]
    files_for_section.extend(
        (section, filename)
        for filename, section in AUXILIARY_SCHEMA_SECTIONS.items()
        if filename in rows_by_file and filename not in mapped_files
    )
    known_files = mapped_files | set(AUXILIARY_SCHEMA_SECTIONS)
    unmapped_files = [
        filename
        for filename, rows in rows_by_file.items()
        if filename not in known_files and any(row.get("label") for row in rows)
    ]
    if unmapped_files:
        raise ValueError("labelled schema rows have no Hub page: " + ", ".join(unmapped_files))
    for section, filename in files_for_section:
        path = schema_dir / filename
        for row in rows_by_file.get(filename, []):
            sourced_row = dict(row)
            sourced_row["_source"] = path
            rows_by_section.setdefault(section, []).append(sourced_row)

    for section, item in page_by_key.items():
        name = str(item["name"])
        needs = item.get("needs") if isinstance(item.get("needs"), dict) else {}
        caps = _values(needs.get("cap"))
        compositors, unknown_caps = _compositors_for_caps(caps, providers)
        keywords = _unique(_words(name) + _words(section) + _words(unknown_caps))
        rows = rows_by_section.get(section, [])
        pages.append(
            {
                "id": f"page:{section}",
                "kind": "page",
                "title": name,
                "path": f"Ryoku Hub > {name}",
                "open": f"Super+comma, then {name} (or `ryoku-shell hub open {section}`)",
                "desc": _summarize_page(rows, guide_descriptions.get(section, "")),
                "keywords": keywords,
                "compositors": compositors,
                "advanced": bool(item.get("adv", False)),
                "source": _repo_source(repo, hub_path),
            }
        )

    settings: list[dict[str, Any]] = []
    used_ids: dict[str, int] = {}
    for section, rows in rows_by_section.items():
        page = page_by_key.get(section)
        if not page:
            continue
        page_name = str(page["name"])
        tabs = _unique(str(row.get("tab", "")) for row in rows)
        include_tab = len(tabs) > 1
        page_open = f"Super+comma, then {page_name} (or `ryoku-shell hub open {section}`)"
        for ordinal, row in enumerate(rows, start=1):
            label = str(row.get("label") or "").strip()
            if not label:
                continue
            entry_id = _setting_id(section, row, ordinal)
            used_ids[entry_id] = used_ids.get(entry_id, 0) + 1
            if used_ids[entry_id] > 1:
                entry_id = f"{entry_id}:{used_ids[entry_id]}"
            caps = _values(row.get("caps"))
            host_caps = _values(row.get("hostCaps"))
            if host_caps:
                compositors, unknown_caps = ["any"], host_caps
            else:
                compositors, unknown_caps = _compositors_for_caps(caps, providers)
            tab = str(row.get("tab") or "").strip()
            path_parts = ["Ryoku Hub", page_name]
            if include_tab and tab:
                path_parts.append(tab)
            path_parts.append(label)
            keywords = _unique(
                _words(row.get("keywords"))
                + _words(row.get("group"))
                + _words(row.get("key"))
                + _words(unknown_caps)
            )
            settings.append(
                {
                    "id": entry_id,
                    "kind": "setting",
                    "title": label,
                    "path": " > ".join(path_parts),
                    "open": page_open,
                    "desc": str(row.get("desc") or ""),
                    "keywords": keywords,
                    "compositors": compositors,
                    "advanced": bool(row.get("adv", False) or page.get("adv", False)),
                    "source": _repo_source(repo, row["_source"]),
                }
            )

    entries = pages + settings + _keybind_entries(repo) + surfaces
    return sorted(entries, key=lambda entry: entry["id"])


def merge_channel_entries(
    stable_entries: list[dict[str, Any]],
    unstable_entries: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    stable = {str(entry["id"]): entry for entry in stable_entries}
    unstable = {str(entry["id"]): entry for entry in (unstable_entries or [])}
    merged: list[dict[str, Any]] = []
    for entry_id in sorted(stable.keys() | unstable.keys()):
        stable_entry = stable.get(entry_id)
        unstable_entry = unstable.get(entry_id)
        if stable_entry and unstable_entry:
            same_location = (
                stable_entry["title"] == unstable_entry["title"]
                and stable_entry["path"] == unstable_entry["path"]
            )
            if same_location:
                merged.append({**stable_entry, "channels": ["stable", "unstable"]})
            else:
                merged.append({**stable_entry, "channels": ["stable"]})
                merged.append(
                    {
                        **unstable_entry,
                        "id": f"{entry_id}@unstable",
                        "channels": ["unstable"],
                    }
                )
        elif stable_entry:
            merged.append({**stable_entry, "channels": ["stable"]})
        elif unstable_entry:
            merged.append({**unstable_entry, "channels": ["unstable"]})
    return sorted(merged, key=lambda entry: entry["id"])


def build_gui_map(repo: Path, unstable: Path | None = None) -> dict[str, Any]:
    repo = repo.resolve()
    stable_commit = _git(repo, "rev-parse", "--short", "HEAD")
    unstable_repo = unstable.resolve() if unstable else None
    unstable_commit = (
        _git(unstable_repo, "rev-parse", "--short", "HEAD")
        if unstable_repo
        else None
    )
    unstable_entries = collect_entries(unstable_repo) if unstable_repo else None
    return {
        "generated_from": {
            "repo": str(repo),
            "commit": stable_commit,
            "date": _git(repo, "show", "-s", "--format=%cI", "HEAD"),
            "stable_version": _stable_version(repo),
            "stable_commit": stable_commit,
            "unstable_commit": unstable_commit,
        },
        "entries": merge_channel_entries(collect_entries(repo), unstable_entries),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Nero's Ryoku GUI map")
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--unstable", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = build_gui_map(args.repo, args.unstable)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
