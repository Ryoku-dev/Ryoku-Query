import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.knowledge.index import KnowledgeIndex, normalize_path
from support import normalize_text


class FakeEncoder:
    def __init__(self):
        self.calls = []

    def encode(self, texts):
        self.calls.append(tuple(texts))
        vectors = []
        for text in texts:
            normalized = normalize_text(text)
            vectors.append(
                (
                    float(any(word in normalized for word in ("blur", "frost", "darken", "dim"))),
                    float("network" in normalized),
                    float("doctor" in normalized),
                    float("screenshot" in normalized),
                )
            )
        return np.asarray(vectors, dtype=np.float32)


def card(card_id, topic="network", **overrides):
    value = {
        "id": card_id,
        "title": f"Help with {topic} {card_id}",
        "examples": [f"first {topic} question {card_id}", f"second {topic} question {card_id}"],
        "keywords": [topic],
        "exact_terms": [],
        "answer": f"answer for {card_id}",
        "risk": "informational",
        "docs_url": None,
        "source_hints": [],
        "clarifies_with": [],
    }
    value.update(overrides)
    return value


def gui_entry(entry_id, title, path, desc, **overrides):
    value = {
        "id": entry_id,
        "kind": "setting",
        "title": title,
        "path": path,
        "open": "Super+comma, then Desktop",
        "desc": desc,
        "keywords": ["network"],
        "compositors": ["any"],
        "advanced": False,
        "source": "ryoku/hub/example.qml",
    }
    value.update(overrides)
    return value


class KnowledgeIndexTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def write_json(self, name, value):
        path = self.root / name
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def build(self, cards, entries=(), chunks=(), encoder=None):
        support_path = self.write_json("support.json", cards)
        gui_path = self.write_json(
            "gui_map.json",
            {"generated_from": {}, "entries": list(entries)},
        )
        docs_path = self.write_json(
            "docs_chunks.json",
            {"generated_from": {}, "chunks": list(chunks)},
        )
        return KnowledgeIndex.from_paths(
            support_path,
            gui_path,
            docs_path,
            encoder or FakeEncoder(),
        )

    def test_mixed_results_respect_per_kind_quotas(self):
        cards = [card(f"card-{index}") for index in range(5)]
        entries = [
            gui_entry(
                f"setting:network/value-{index}",
                f"Network setting {index}",
                f"Ryoku Hub > Network > Setting {index}",
                "Change a shared network option.",
            )
            for index in range(7)
        ]
        chunks = [
            {
                "id": f"doc:network-{index}",
                "title": f"Network > Topic {index}",
                "url": f"https://docs.ryoku.dev/docs/network#topic-{index}",
                "text": "Shared network troubleshooting guidance.",
                "source": "network.mdx",
            }
            for index in range(4)
        ]
        index = self.build(cards, entries, chunks)

        results = index.search("network", k=30)
        counts = {
            kind: sum(result.kind == kind for result in results)
            for kind in ("card", "gui", "doc")
        }

        self.assertEqual(counts, {"card": 4, "gui": 6, "doc": 3})
        self.assertEqual(index.kind_counts, {"card": 5, "gui": 7, "doc": 4})
        self.assertEqual(len(results), 13)
        self.assertTrue(all(result.score > 0 for result in results))

    def test_missing_optional_files_are_empty_sources(self):
        support_path = self.write_json("support.json", [card("health.doctor", "doctor")])

        index = KnowledgeIndex.from_paths(
            support_path,
            self.root / "missing-gui.json",
            None,
            FakeEncoder(),
        )

        results = index.search("doctor")
        self.assertEqual([result.kind for result in results], ["card"])
        self.assertEqual(results[0].id, "card:health.doctor")

    def test_darkening_synonym_ranks_deck_frost_first_among_gui_items(self):
        entries = [
            gui_entry(
                "setting:desktop/deckFrostEnabled",
                "Deck frost",
                "Ryoku Hub > Desktop > General > Deck frost",
                "Blurs and darkens the desktop behind open windows.",
                keywords=["window effect"],
                compositors=["hyprland", "niri"],
                channels=["unstable"],
            ),
            gui_entry(
                "setting:desktop/windowFocus",
                "Window focus",
                "Ryoku Hub > Desktop > General > Window focus",
                "Controls focus behavior for open windows.",
                keywords=["windows"],
            ),
        ]
        index = self.build([card("desktop.effects", "effects")], entries)

        gui_results = [
            result
            for result in index.search("stop the darkening behind windows")
            if result.kind == "gui"
        ]

        self.assertEqual(gui_results[0].title, "Deck frost")
        self.assertIn("Blurs and darkens", gui_results[0].text)
        self.assertIn("Open: Super+comma", gui_results[0].text)
        self.assertIn("(Hyprland/niri only)", gui_results[0].text)
        self.assertIn("(unstable channel only)", gui_results[0].text)

    def test_portuguese_glossary_retrieves_english_gui_entry(self):
        encoder = FakeEncoder()
        index = self.build(
            [card("desktop.effects", "effects")],
            [
                gui_entry(
                    "setting:desktop/windowBlur",
                    "Window blur",
                    "Ryoku Hub > Desktop > General > Window blur",
                    "Turn off blur behind open windows.",
                    keywords=["blur", "window"],
                )
            ],
            encoder=encoder,
        )

        gui_results = [
            result
            for result in index.search("como desativar o desfoque atrás das janelas")
            if result.kind == "gui"
        ]

        self.assertEqual([result.id for result in gui_results], ["setting:desktop/windowBlur"])
        self.assertIn("blur", encoder.calls[-1][0])
        self.assertIn("window", encoder.calls[-1][0])

    def test_spanish_glossary_retrieves_english_gui_entry(self):
        encoder = FakeEncoder()
        index = self.build(
            [card("desktop.wallpaper", "desktop")],
            [
                gui_entry(
                    "setting:desktop/wallpaper",
                    "Wallpaper",
                    "Ryoku Hub > Desktop > Wallpaper",
                    "Change the desktop background.",
                    keywords=["wallpaper", "background"],
                )
            ],
            encoder=encoder,
        )

        gui_results = [
            result
            for result in index.search("cómo cambiar el fondo de pantalla")
            if result.kind == "gui"
        ]

        self.assertEqual([result.id for result in gui_results], ["setting:desktop/wallpaper"])
        self.assertIn("wallpaper", encoder.calls[-1][0])
        self.assertIn("change", encoder.calls[-1][0])

    def test_gui_duplicates_prefer_both_channels_before_quota(self):
        entries = [
            gui_entry(
                "setting:display/scale-stable",
                "Display scale",
                "Ryoku Hub > Displays > Scale",
                "Change the display scale.",
                channels=["stable"],
            ),
            gui_entry(
                "setting:display/scale-all",
                " display SCALE ",
                " ryoku hub>displays > scale ",
                "Change the display scale.",
                channels=["stable", "unstable"],
            ),
            *[
                gui_entry(
                    f"setting:network/alternative-{index}",
                    f"Network alternative {index}",
                    f"Ryoku Hub > Network > Alternative {index}",
                    "Change another network setting.",
                )
                for index in range(5)
            ],
        ]
        index = self.build([card("network.help")], entries)

        gui_results = [
            result for result in index.search("network", k=20) if result.kind == "gui"
        ]
        result_ids = {result.id for result in gui_results}

        self.assertEqual(len(gui_results), 6)
        self.assertIn("setting:display/scale-all", result_ids)
        self.assertNotIn("setting:display/scale-stable", result_ids)
        self.assertIn("setting:network/alternative-4", result_ids)

    def test_known_paths_include_full_setting_and_page_prefix(self):
        full_path = " Ryoku Hub> Desktop > General > Deck frost "
        index = self.build(
            [card("desktop.effects", "effects")],
            [
                gui_entry(
                    "setting:desktop/deckFrostEnabled",
                    "Deck frost",
                    full_path,
                    "Blurs the desktop.",
                )
            ],
        )

        self.assertIn(normalize_path(full_path), index.known_paths())
        self.assertIn("ryoku hub > desktop", index.known_paths())

    def test_card_decision_delegates_to_support_retriever(self):
        doctor = card(
            "health.doctor",
            "doctor",
            exact_terms=["ryoku doctor"],
            examples=["inspect doctor drift", "repair doctor drift"],
        )
        report = card(
            "health.report",
            "report",
            examples=["save a support report", "inspect a support report"],
        )
        index = self.build([doctor, report])

        decision = index.card_decision("ryoku doctor")

        self.assertEqual(decision.kind, "answer")
        self.assertEqual(decision.card.id, "health.doctor")
        self.assertEqual(index.card("health.doctor"), decision.card)
        self.assertIsNone(index.card("card:health.doctor"))
        self.assertEqual(tuple(card.id for card in index.cards), ("health.doctor", "health.report"))

    def test_unified_item_texts_are_encoded_in_one_batch(self):
        encoder = FakeEncoder()
        index = self.build(
            [card("desktop.help", "desktop")],
            [
                gui_entry(
                    "page:desktop",
                    "Desktop",
                    "Ryoku Hub > Desktop",
                    "Desktop preferences.",
                    kind="page",
                )
            ],
            [
                {
                    "id": "doc:desktop",
                    "title": "Desktop > Overview",
                    "url": None,
                    "text": "Desktop overview.",
                    "source": "desktop.mdx",
                }
            ],
            encoder,
        )

        unified_batches = [call for call in encoder.calls if len(call) == 3]
        self.assertEqual(len(unified_batches), 1)
        self.assertEqual(len(index.search("desktop")), 3)
        self.assertEqual(len(encoder.calls[-1]), 1)


if __name__ == "__main__":
    unittest.main()
