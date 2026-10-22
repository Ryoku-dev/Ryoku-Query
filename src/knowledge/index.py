from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from support import (
    RetrievalDecision,
    SupportCard,
    SupportRetriever,
    _raw_tokens,
    _tokens,
    load_support_cards,
    normalize_text,
)


@dataclass(frozen=True)
class Evidence:
    kind: str
    id: str
    title: str
    text: str
    path: str | None
    url: str | None
    risk: str | None
    score: float


@dataclass(frozen=True)
class _SearchItem:
    evidence: Evidence
    match_text: str
    tokens: tuple[str, ...]
    terms: frozenset[str]
    exact_terms: tuple[str, ...] = ()
    channels: frozenset[str] = frozenset()


_SYNONYM_GROUPS = (
    ("blur", "frost", "haze", "darken", "darkening", "dim", "dimming"),
    ("bar", "panel", "topbar", "waybar", "statusbar"),
    ("dock", "taskbar"),
    ("launcher", "app menu", "start menu", "app drawer"),
    ("wallpaper", "background", "backdrop"),
    ("settings", "hub", "control panel", "preferences"),
    ("keybind", "shortcut", "hotkey", "keyboard shortcut"),
    ("screenshot", "screen capture", "print screen", "snip"),
    ("alt+f4", "alt f4", "close"),
    ("minimize", "minimise", "minimizar", "hide", "scratchpad"),
    ("record", "recording", "recorder"),
    ("notification", "notif", "toast"),
    ("lockscreen", "lock screen", "greeter", "login screen", "sddm"),
    ("update", "upgrade"),
    ("theme", "colors", "colours", "palette", "matugen"),
)
_SUPPORT_GLOSSARY = (
    (
        ("configurações", "configuracoes", "configuração", "ajustes", "configuración"),
        ("settings", "configuration"),
    ),
    (("papel de parede", "fundo", "plano de fundo", "fondo de pantalla", "fondo"), ("wallpaper",)),
    (("barra",), ("bar",)),
    (("dock",), ("dock",)),
    (("lançador", "lanzador", "menu de apps"), ("launcher",)),
    (("atalho", "atalhos", "atajo", "atajos", "tecla", "teclas"), ("shortcut", "keybind")),
    (("captura de tela", "print", "captura de pantalla"), ("screenshot",)),
    (
        (
            "gravador",
            "gravador de tela",
            "gravar",
            "gravar tela",
            "gravação",
            "gravação de tela",
            "grabar",
            "grabar pantalla",
            "grabación",
            "grabación de pantalla",
        ),
        ("record", "recording"),
    ),
    (("área de transferência", "portapapeles"), ("clipboard",)),
    (("tela de bloqueio", "pantalla de bloqueo"), ("lockscreen",)),
    (("tela de login", "inicio de sesión"), ("login",)),
    (("atualizar", "atualização", "actualizar", "actualización"), ("update",)),
    (
        (
            "desligar computador",
            "desligar o computador",
            "desligar meu computador",
            "desligar pc",
            "desligar o pc",
            "desligar meu pc",
            "apagar computador",
            "apagar o computador",
            "apagar meu computador",
            "apagar equipo",
            "apagar el equipo",
            "apagar mi equipo",
            "apagar pc",
            "apagar o pc",
            "apagar el pc",
            "apagar meu pc",
            "apagar mi pc",
        ),
        ("power off", "shut down"),
    ),
    (("reiniciar",), ("restart", "reboot")),
    (("instalar", "instalação", "instalación"), ("install",)),
    (("tema",), ("theme",)),
    (("cores", "colores"), ("colors",)),
    (("janela", "janelas", "ventana", "ventanas"), ("window",)),
    (("desfoque", "desenfoque"), ("blur",)),
    (("escurecer", "oscurecer"), ("darken",)),
    (("escuro", "oscuro"), ("dark",)),
    (("tela", "pantalla", "resolução", "resolución"), ("display", "monitor")),
    (("som", "áudio", "sonido"), ("sound", "audio")),
    (("rede", "red", "wifi"), ("network", "wifi")),
    (("não funciona", "nao funciona", "no funciona"), ("not working",)),
    (("mudar", "trocar", "alterar", "cambiar"), ("change",)),
    (("desativar", "desligar", "desactivar", "apagar"), ("turn off", "disable")),
    (("ativar", "ligar", "activar"), ("turn on", "enable")),
    (("tela preta", "pantalla negra"), ("black screen",)),
    (("rato", "ratón"), ("mouse", "cursor")),
    (("teclado",), ("keyboard",)),
    (("ícones", "iconos"), ("icons",)),
    (("fonte", "fuente"), ("font",)),
    (("notificação", "notificación"), ("notification",)),
)


def _without_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", normalize_text(text))
    return "".join(character for character in decomposed if not unicodedata.combining(character))


def _build_glossary_matches() -> tuple[
    tuple[tuple[tuple[str, ...], ...], tuple[str, ...]], ...
]:
    matches = []
    for phrases, english_terms in _SUPPORT_GLOSSARY:
        normalized_phrases = tuple(
            dict.fromkeys(_raw_tokens(_without_accents(phrase)) for phrase in phrases)
        )
        matches.append((normalized_phrases, english_terms))
    return tuple(matches)


_GLOSSARY_MATCHES = _build_glossary_matches()


def _build_synonyms() -> dict[str, tuple[str, ...]]:
    merged: dict[str, list[str]] = {}
    for group in _SYNONYM_GROUPS:
        words = tuple(
            dict.fromkeys(
                token
                for phrase in group
                for token in _tokens(normalize_text(phrase))
            )
        )
        for word in words:
            bucket = merged.setdefault(word, [])
            bucket.extend(candidate for candidate in words if candidate not in bucket)
    return {word: tuple(candidates) for word, candidates in merged.items()}


_SYNONYMS = _build_synonyms()
_PATH_SEPARATOR_RE = re.compile(r"\s*>\s*")
_COMPOSITOR_NAMES = {
    "hyprland": "Hyprland",
    "niri": "niri",
    "wayfire": "Wayfire",
}


def normalize_path(path: str) -> str:
    """Normalize a GUI path; callers comparing paths must use this helper too."""

    return _PATH_SEPARATOR_RE.sub(" > ", normalize_text(path))


def _expanded_tokens(text: str) -> tuple[str, ...]:
    expanded: list[str] = []
    for token in _tokens(_without_accents(text)):
        expanded.extend(_SYNONYMS.get(token, (token,)))
    return tuple(expanded)


def _normalized_rows(vectors: Any, expected_rows: int) -> np.ndarray:
    values = np.asarray(vectors, dtype=np.float32)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2 or values.shape[0] != expected_rows:
        raise ValueError("encoder returned an unexpected vector shape")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    normalized = values / np.clip(norms, 1e-12, None)
    return np.nan_to_num(normalized, copy=False)


def _has_token_phrase(query: tuple[str, ...], phrase: tuple[str, ...]) -> bool:
    if not phrase or len(phrase) > len(query):
        return False
    width = len(phrase)
    return any(query[index : index + width] == phrase for index in range(len(query) - width + 1))


def _query_expansion(query: str) -> tuple[tuple[str, ...], str]:
    raw_query = _raw_tokens(_without_accents(query))
    english_terms: list[str] = []
    for phrases, terms in _GLOSSARY_MATCHES:
        if any(_has_token_phrase(raw_query, phrase) for phrase in phrases):
            english_terms.extend(term for term in terms if term not in english_terms)
    embedding_text = " ".join((query, *english_terms))
    return _expanded_tokens(embedding_text), embedding_text


def _read_collection(path: Path | None, key: str) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    value = json.loads(path.read_text(encoding="utf-8"))
    rows = value.get(key, []) if isinstance(value, dict) else []
    return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []


def _joined(*parts: Any) -> str:
    values: list[str] = []
    for part in parts:
        if isinstance(part, str):
            if part.strip():
                values.append(part.strip())
        elif isinstance(part, (list, tuple)):
            values.extend(str(value).strip() for value in part if str(value).strip())
    return " ".join(values)


def _compositor_note(compositors: tuple[str, ...]) -> str:
    if compositors == ("any",) or not compositors:
        return ""
    names = "/".join(_COMPOSITOR_NAMES.get(name, name) for name in compositors)
    return f" ({names} only)"


def _channel_note(channels: tuple[str, ...]) -> str:
    selected = frozenset(channel.lower() for channel in channels)
    if selected == {"unstable"}:
        return " (unstable channel only)"
    if selected == {"stable"}:
        return " (stable channel only)"
    return ""


class KnowledgeIndex:
    def __init__(
        self,
        cards: list[SupportCard],
        gui_entries: list[dict[str, Any]],
        docs_chunks: list[dict[str, Any]],
        encoder: Any,
        *,
        max_cards: int = 4,
        max_gui: int = 6,
        max_docs: int = 3,
    ):
        self._cards = tuple(cards)
        self._cards_by_id = {card.id: card for card in cards}
        self._retriever = SupportRetriever(cards, encoder)
        self._encoder = encoder
        self._quotas = {
            "card": max(0, max_cards),
            "gui": max(0, max_gui),
            "doc": max(0, max_docs),
        }

        items: list[_SearchItem] = []
        for card in cards:
            match_text = _joined(
                card.title,
                card.examples,
                card.keywords,
                card.exact_terms,
            )
            evidence = Evidence(
                kind="card",
                id=f"card:{card.id}",
                title=card.title,
                text=card.answer,
                path=None,
                url=card.docs_url,
                risk=card.risk,
                score=0.0,
            )
            tokens = _expanded_tokens(match_text)
            items.append(
                _SearchItem(
                    evidence,
                    match_text,
                    tokens,
                    frozenset(tokens),
                    card.exact_terms,
                )
            )

        known_paths: set[str] = set()
        for entry in gui_entries:
            title = str(entry.get("title", "")).strip()
            path = str(entry.get("path", "")).strip()
            desc = str(entry.get("desc", "")).strip()
            entry_id = str(entry.get("id", "")).strip()
            if not entry_id or not title:
                continue
            keywords = entry.get("keywords", [])
            if not isinstance(keywords, list):
                keywords = []
            raw_compositors = entry.get("compositors", ["any"])
            compositors = (
                tuple(str(value) for value in raw_compositors)
                if isinstance(raw_compositors, list)
                else ("any",)
            )
            raw_channels = entry.get("channels", ["stable", "unstable"])
            channels = (
                tuple(str(value) for value in raw_channels)
                if isinstance(raw_channels, list)
                else ("stable", "unstable")
            )
            open_value = entry.get("open")
            open_text = open_value.strip() if isinstance(open_value, str) else ""
            evidence_text = f"{path}: {desc}"
            if open_text:
                evidence_text += f" Open: {open_text}"
            evidence_text += _compositor_note(compositors)
            evidence_text += _channel_note(channels)
            match_text = _joined(title, path, desc, keywords)
            tokens = _expanded_tokens(match_text)
            evidence = Evidence(
                kind="gui",
                id=entry_id,
                title=title,
                text=evidence_text,
                path=path or None,
                url=None,
                risk=None,
                score=0.0,
            )
            items.append(
                _SearchItem(
                    evidence,
                    match_text,
                    tokens,
                    frozenset(tokens),
                    channels=frozenset(channel.lower() for channel in channels),
                )
            )

            if entry.get("kind") in {"page", "setting"} and path:
                known_paths.add(normalize_path(path))
                parts = [part.strip() for part in path.split(">")]
                if len(parts) >= 2 and normalize_text(parts[0]) == "ryoku hub":
                    known_paths.add(normalize_path(" > ".join(parts[:2])))

        for chunk in docs_chunks:
            title = str(chunk.get("title", "")).strip()
            text = str(chunk.get("text", "")).strip()
            chunk_id = str(chunk.get("id", "")).strip()
            if not chunk_id or not title:
                continue
            match_text = _joined(title, text)
            tokens = _expanded_tokens(match_text)
            url_value = chunk.get("url")
            evidence = Evidence(
                kind="doc",
                id=chunk_id,
                title=title,
                text=text,
                path=None,
                url=url_value if isinstance(url_value, str) else None,
                risk=None,
                score=0.0,
            )
            items.append(_SearchItem(evidence, match_text, tokens, frozenset(tokens)))

        self._items = tuple(items)
        self._kind_counts = {
            kind: sum(item.evidence.kind == kind for item in items)
            for kind in ("card", "gui", "doc")
        }
        self._known_paths = frozenset(known_paths)
        self._lengths = np.asarray([len(item.tokens) for item in items], dtype=np.float32)
        self._average_length = float(np.mean(self._lengths)) if items else 1.0
        self._term_counts = tuple(Counter(item.tokens) for item in items)
        frequencies: Counter[str] = Counter()
        for item in items:
            frequencies.update(item.terms)
        count = len(items)
        self._idf = {
            term: math.log(1.0 + (count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in frequencies.items()
        }
        self._vectors = _normalized_rows(
            encoder.encode([item.match_text for item in items]),
            len(items),
        )

    @classmethod
    def from_paths(
        cls,
        support_path: Path,
        gui_map_path: Path | None,
        docs_path: Path | None,
        encoder: Any,
    ) -> KnowledgeIndex:
        cards = load_support_cards(support_path)
        gui_entries = _read_collection(gui_map_path, "entries")
        docs_chunks = _read_collection(docs_path, "chunks")
        return cls(cards, gui_entries, docs_chunks, encoder)

    @property
    def cards(self) -> tuple[SupportCard, ...]:
        return self._cards

    @property
    def kind_counts(self) -> dict[str, int]:
        return dict(self._kind_counts)

    def card_decision(self, query: str) -> RetrievalDecision:
        return self._retriever.retrieve(query)

    def card(self, card_id: str) -> SupportCard | None:
        return self._cards_by_id.get(card_id)

    def known_paths(self) -> frozenset[str]:
        return self._known_paths

    def search(self, query: str, k: int = 10) -> list[Evidence]:
        if k <= 0 or not self._items:
            return []
        query_tokens, embedding_text = _query_expansion(query)
        if not query_tokens:
            return []
        query_terms = frozenset(query_tokens)
        lexical = self._bm25(query_tokens)
        query_vector = _normalized_rows(self._encoder.encode([embedding_text]), 1)[0]
        semantic = np.nan_to_num(self._vectors @ query_vector, copy=False)

        lexical_order = sorted(
            range(len(self._items)),
            key=lambda index: (-lexical[index], self._items[index].evidence.id),
        )
        semantic_order = sorted(
            range(len(self._items)),
            key=lambda index: (-float(semantic[index]), self._items[index].evidence.id),
        )
        lexical_rank = {index: rank for rank, index in enumerate(lexical_order, 1)}
        semantic_rank = {index: rank for rank, index in enumerate(semantic_order, 1)}
        raw_query = _raw_tokens(normalize_text(query))

        candidates: list[tuple[float, int]] = []
        for index, item in enumerate(self._items):
            has_overlap = not query_terms.isdisjoint(item.terms)
            similarity = float(semantic[index])
            if not has_overlap and similarity < 0.30:
                continue
            score = 1.0 / (60 + lexical_rank[index])
            score += 1.0 / (60 + semantic_rank[index])
            exact_matches = sum(
                _has_token_phrase(raw_query, _raw_tokens(term))
                for term in item.exact_terms
            )
            if exact_matches:
                score += 0.05 + 0.005 * (exact_matches - 1)
            candidates.append((score, index))

        unique_candidates: list[tuple[float, int]] = []
        gui_candidates: dict[tuple[str, str], tuple[float, int]] = {}
        both_channels = {"stable", "unstable"}
        for candidate in candidates:
            score, index = candidate
            item = self._items[index]
            evidence = item.evidence
            if evidence.kind != "gui":
                unique_candidates.append(candidate)
                continue
            identity = (
                normalize_text(evidence.title),
                normalize_path(evidence.path or ""),
            )
            current = gui_candidates.get(identity)
            if current is None:
                gui_candidates[identity] = candidate
                continue
            current_score, current_index = current
            current_item = self._items[current_index]
            preference = (
                both_channels.issubset(item.channels),
                score,
                float(semantic[index]),
            )
            current_preference = (
                both_channels.issubset(current_item.channels),
                current_score,
                float(semantic[current_index]),
            )
            if preference > current_preference or (
                preference == current_preference
                and evidence.id < current_item.evidence.id
            ):
                gui_candidates[identity] = candidate
        candidates = unique_candidates + list(gui_candidates.values())

        candidates.sort(
            key=lambda pair: (
                -pair[0],
                -float(semantic[pair[1]]),
                self._items[pair[1]].evidence.id,
            )
        )
        used = {"card": 0, "gui": 0, "doc": 0}
        results: list[Evidence] = []
        for score, index in candidates:
            evidence = self._items[index].evidence
            if used[evidence.kind] >= self._quotas[evidence.kind]:
                continue
            used[evidence.kind] += 1
            results.append(
                Evidence(
                    kind=evidence.kind,
                    id=evidence.id,
                    title=evidence.title,
                    text=evidence.text,
                    path=evidence.path,
                    url=evidence.url,
                    risk=evidence.risk,
                    score=score,
                )
            )
            if len(results) >= k:
                break
        return results

    def _bm25(self, query_tokens: tuple[str, ...]) -> np.ndarray:
        scores = np.zeros(len(self._items), dtype=np.float32)
        k1 = 1.5
        b = 0.75
        for index, (counts, length) in enumerate(zip(self._term_counts, self._lengths)):
            normalization = k1 * (1.0 - b + b * float(length) / self._average_length)
            score = 0.0
            for token in set(query_tokens):
                frequency = counts.get(token, 0)
                if frequency:
                    score += self._idf.get(token, 0.0) * (
                        frequency * (k1 + 1.0) / (frequency + normalization)
                    )
            scores[index] = score
        return scores
