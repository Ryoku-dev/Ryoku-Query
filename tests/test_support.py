import json
import re
import tempfile
import unittest
from pathlib import Path

import numpy as np

from support import (
    CatalogError,
    SupportCard,
    SupportRetriever,
    load_support_cards,
    normalize_text,
    requires_safety_confirmation,
)


def card(card_id="health.doctor", **overrides):
    value = {
        "id": card_id,
        "title": "Run Ryoku Doctor",
        "examples": ["check my ryoku system", "repair configuration drift"],
        "keywords": ["doctor", "health"],
        "exact_terms": ["ryoku doctor"],
        "answer": "Run `ryoku doctor` to inspect and reconcile supported drift.",
        "risk": "informational",
        "docs_url": "https://docs.ryoku.dev/docs/troubleshoot",
        "source_hints": ["docs/cli.md", "ryoku/cli/"],
        "clarifies_with": [],
    }
    value.update(overrides)
    return value


class SupportCatalogTests(unittest.TestCase):
    def load(self, rows):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "support.json"
        path.write_text(json.dumps(rows), encoding="utf-8")
        return load_support_cards(path)

    def test_loads_valid_card_as_immutable_tuples(self):
        loaded = self.load([card()])

        self.assertEqual(loaded[0].id, "health.doctor")
        self.assertEqual(
            loaded[0].examples,
            ("check my ryoku system", "repair configuration drift"),
        )

    def test_rejects_unknown_field(self):
        with self.assertRaisesRegex(CatalogError, "unknown fields.*surprise"):
            self.load([card(surprise=True)])

    def test_rejects_duplicate_id(self):
        with self.assertRaisesRegex(CatalogError, "duplicate.*health.doctor"):
            self.load([card(), card()])

    def test_rejects_invalid_risk(self):
        with self.assertRaisesRegex(CatalogError, "risk"):
            self.load([card(risk="reckless")])

    def test_requires_two_distinct_examples(self):
        with self.assertRaisesRegex(CatalogError, "examples"):
            self.load([card(examples=["same", "same"])])

    def test_rejects_empty_answer(self):
        with self.assertRaisesRegex(CatalogError, "answer"):
            self.load([card(answer="  ")])

    def test_rejects_non_https_documentation_url(self):
        with self.assertRaisesRegex(CatalogError, "docs_url"):
            self.load([card(docs_url="http://docs.example.test")])

    def test_allows_null_documentation_url(self):
        self.assertIsNone(self.load([card(docs_url=None)])[0].docs_url)

    def test_rejects_missing_clarification_target(self):
        with self.assertRaisesRegex(CatalogError, "clarifies_with.*missing"):
            self.load([card(clarifies_with=["missing.intent"])])

    def test_normalization_preserves_command_path_digits_and_punctuation(self):
        value = normalize_text("  RYOKU\u00a0Doctor --Check /tmp/GPU-2  ")

        self.assertEqual(value, "ryoku doctor --check /tmp/gpu-2")

    def test_normalization_collapses_split_reset(self):
        value = normalize_text("how do i re set ryoku")

        self.assertEqual(value, "how do i reset ryoku")


class MappingEncoder:
    def __init__(self, vectors):
        self.vectors = vectors

    def encode(self, texts):
        return np.asarray(
            [self.vectors.get(text, (0.0, 0.0, 0.0)) for text in texts],
            dtype=np.float32,
        )


def support_card(card_id, examples, **overrides):
    values = {
        "id": card_id,
        "title": card_id.replace(".", " "),
        "examples": tuple(examples),
        "keywords": (),
        "exact_terms": (),
        "answer": f"answer for {card_id}",
        "risk": "informational",
        "docs_url": None,
        "source_hints": (),
        "clarifies_with": (),
    }
    values.update(overrides)
    return SupportCard(**values)


class SupportRetrieverTests(unittest.TestCase):
    def test_unique_exact_term_selects_the_card(self):
        cards = [
            support_card(
                "health.doctor",
                ("inspect drift", "repair system"),
                exact_terms=("ryoku doctor",),
            ),
            support_card(
                "health.report",
                ("write report", "save diagnostics"),
            ),
        ]
        encoder = MappingEncoder(
            {
                "inspect drift": (1, 0, 0),
                "repair system": (1, 0, 0),
                "write report": (0, 1, 0),
                "save diagnostics": (0, 1, 0),
                "ryoku doctor": (1, 0, 0),
            }
        )

        decision = SupportRetriever(cards, encoder).retrieve("ryoku doctor")

        self.assertEqual(decision.kind, "answer")
        self.assertEqual(decision.card.id, "health.doctor")

    def test_exact_term_is_token_bounded(self):
        card_value = support_card(
            "health.doctor",
            ("inspect drift", "repair system"),
            exact_terms=("ryoku doctor",),
        )
        encoder = MappingEncoder(
            {
                "inspect drift": (1, 0, 0),
                "repair system": (1, 0, 0),
                "myryoku doctoring notes": (1, 0, 0),
            }
        )

        decision = SupportRetriever([card_value], encoder).retrieve(
            "myryoku doctoring notes"
        )

        self.assertEqual(decision.kind, "no_match")

    def test_related_close_candidates_request_clarification(self):
        first = support_card(
            "health.report",
            ("doctor report location", "save diagnostic report"),
            keywords=("report", "doctor"),
            clarifies_with=("health.privacy",),
        )
        second = support_card(
            "health.privacy",
            ("doctor report privacy", "report safe to share"),
            keywords=("report", "doctor"),
            clarifies_with=("health.report",),
        )
        encoder = MappingEncoder(
            {
                **{text: (1, 0, 0) for text in first.examples},
                **{text: (1, 0, 0) for text in second.examples},
                "doctor report": (1, 0, 0),
            }
        )

        decision = SupportRetriever([first, second], encoder).retrieve(
            "doctor report"
        )

        self.assertEqual(decision.kind, "clarify")
        self.assertEqual(
            {card.id for card in decision.alternatives},
            {"health.report", "health.privacy"},
        )

    def test_semantic_similarity_without_lexical_evidence_is_rejected(self):
        card_value = support_card(
            "health.doctor",
            ("inspect drift", "repair system"),
        )
        encoder = MappingEncoder(
            {
                "inspect drift": (1, 0, 0),
                "repair system": (1, 0, 0),
                "weather tomorrow": (1, 0, 0),
            }
        )

        decision = SupportRetriever([card_value], encoder).retrieve(
            "weather tomorrow"
        )

        self.assertEqual(decision.kind, "no_match")

    def test_destructive_exact_match_requests_clarification(self):
        card_value = support_card(
            "recovery.reset",
            ("recover broken install", "reset ryoku"),
            exact_terms=("ryoku recovery", "reset ryoku"),
            risk="destructive",
        )
        encoder = MappingEncoder(
            {
                "recover broken install": (1, 0, 0),
                "reset ryoku": (1, 0, 0),
                "ryoku recovery": (1, 0, 0),
            }
        )

        decision = SupportRetriever([card_value], encoder).retrieve(
            "ryoku recovery"
        )

        self.assertEqual(decision.kind, "clarify")
        self.assertEqual(decision.alternatives, (card_value,))

    def test_split_reset_phrase_matches_destructive_recovery_clarification(self):
        card_value = support_card(
            "recovery.reset",
            ("recover broken install", "reset ryoku"),
            exact_terms=("ryoku recovery", "reset ryoku"),
            risk="destructive",
        )
        encoder = MappingEncoder(
            {
                "recover broken install": (1, 0, 0),
                "reset ryoku": (1, 0, 0),
                "ryoku recovery": (1, 0, 0),
                "how do i reset ryoku": (1, 0, 0),
            }
        )

        decision = SupportRetriever([card_value], encoder).retrieve(
            "how do i re set ryoku"
        )

        self.assertEqual(decision.kind, "clarify")
        self.assertEqual(decision.alternatives, (card_value,))

    def test_reviewed_destructive_installer_exact_phrase_clarifies(self):
        card_value = support_card(
            "install.dedicated-drive",
            (
                "install on dedicated drive",
                "leave alongside windows install mode",
            ),
            exact_terms=(
                "Erase whole disk",
                "leave alongside windows install mode",
            ),
            risk="destructive",
        )
        encoder = MappingEncoder(
            {
                "install on dedicated drive": (1, 0, 0),
                "leave alongside windows install mode": (1, 0, 0),
            }
        )

        decision = SupportRetriever([card_value], encoder).retrieve(
            "how do i leave alongside windows install mode"
        )

        self.assertEqual(decision.kind, "clarify")
        self.assertEqual(decision.alternatives, (card_value,))

    def test_irrelevant_destructive_candidate_is_rejected(self):
        card_value = support_card(
            "install.erase",
            ("erase dedicated drive", "install to whole disk"),
            keywords=("drive", "disk"),
            risk="destructive",
        )
        encoder = MappingEncoder(
            {
                "erase dedicated drive": (1, 0, 0),
                "install to whole disk": (1, 0, 0),
                "playlist for a long drive": (1, 0, 0),
            }
        )

        decision = SupportRetriever([card_value], encoder).retrieve(
            "playlist for a long drive"
        )

        self.assertEqual(decision.kind, "no_match")

    def test_high_risk_requests_require_safety_confirmation(self):
        requests = (
            "find the most destructive command in the source",
            "remove every snapshot before checking the failure",
            "skip the warnings and remove all user settings",
            "reset the whole desktop without confirmation",
        )
        for request in requests:
            with self.subTest(request=request):
                self.assertTrue(requires_safety_confirmation(request))

    def test_explicit_or_between_related_intents_clarifies(self):
        rollback = support_card(
            "updates.rollback",
            ("undo bad update", "return to snapshot"),
            exact_terms=("roll back",),
            clarifies_with=("recovery.reset",),
        )
        recovery = support_card(
            "recovery.reset",
            ("recover broken install", "reset desktop"),
            clarifies_with=("updates.rollback",),
        )
        encoder = MappingEncoder(
            {
                "undo bad update": (1, 0, 0),
                "return to snapshot": (1, 0, 0),
                "recover broken install": (0, 1, 0),
                "reset desktop": (0, 1, 0),
                "recover or roll back": (1, 0, 0),
            }
        )

        decision = SupportRetriever([rollback, recovery], encoder).retrieve(
            "recover or roll back"
        )

        self.assertEqual(decision.kind, "clarify")

    def test_enumerative_or_does_not_force_clarification(self):
        privacy = support_card(
            "health.privacy",
            ("report privacy details", "safe diagnostic sharing"),
            keywords=("privacy", "personal", "usernames", "hardware"),
            clarifies_with=("health.report",),
        )
        report = support_card(
            "health.report",
            ("create diagnostic report", "save report bundle"),
            keywords=("report", "bundle"),
            clarifies_with=("health.privacy",),
        )
        encoder = MappingEncoder(
            {
                "report privacy details": (1, 0, 0),
                "safe diagnostic sharing": (1, 0, 0),
                "create diagnostic report": (0.4, 0.9, 0),
                "save report bundle": (0.4, 0.9, 0),
                "does the report include usernames paths or hardware ids": (
                    1,
                    0,
                    0,
                ),
            }
        )

        decision = SupportRetriever([privacy, report], encoder).retrieve(
            "does the report include usernames, paths, or hardware ids"
        )

        self.assertNotEqual(decision.kind, "clarify")

    def test_clear_lexical_winner_answers_with_sufficient_semantic_signal(self):
        resolution = support_card(
            "display.resolution",
            ("monitor only offers one mode", "normal choices are missing"),
            keywords=("monitor", "mode", "resolution", "choices"),
        )
        other = support_card(
            "display.color",
            ("display colors look wrong", "adjust screen color"),
            keywords=("display", "color"),
        )
        encoder = MappingEncoder(
            {
                "monitor only offers one mode": (1, 0, 0),
                "normal choices are missing": (1, 0, 0),
                "display colors look wrong": (0, 0, 1),
                "adjust screen color": (0, 0, 1),
                "monitor only offers one resolution mode": (0.45, 0.89, 0),
            }
        )

        decision = SupportRetriever([resolution, other], encoder).retrieve(
            "monitor only offers one resolution mode"
        )

        self.assertEqual(decision.kind, "answer")
        self.assertEqual(decision.card.id, "display.resolution")

    def test_clear_lexical_winner_rejects_weak_semantic_signal(self):
        resolution = support_card(
            "display.resolution",
            ("monitor only offers one mode", "normal choices are missing"),
            keywords=("monitor", "mode", "resolution", "choices"),
        )
        other = support_card(
            "display.color",
            ("display colors look wrong", "adjust screen color"),
        )
        encoder = MappingEncoder(
            {
                "monitor only offers one mode": (1, 0, 0),
                "normal choices are missing": (1, 0, 0),
                "display colors look wrong": (0, 0, 1),
                "adjust screen color": (0, 0, 1),
                "monitor only offers one resolution mode": (0.2, 0.98, 0),
            }
        )

        decision = SupportRetriever([resolution, other], encoder).retrieve(
            "monitor only offers one resolution mode"
        )

        self.assertEqual(decision.kind, "no_match")

    def test_common_question_words_do_not_create_a_match(self):
        card_value = support_card(
            "health.doctor",
            ("what should i do when ryoku breaks", "how can i fix my system"),
        )
        encoder = MappingEncoder(
            {
                "what should i do when ryoku breaks": (1, 0, 0),
                "how can i fix my system": (1, 0, 0),
                "what should i cook for dinner": (1, 0, 0),
            }
        )

        decision = SupportRetriever([card_value], encoder).retrieve(
            "what should i cook for dinner"
        )

        self.assertEqual(decision.kind, "no_match")

    def test_exact_phrase_keeps_stopwords(self):
        card_value = support_card(
            "hardware.gpu-choice",
            ("select graphics card", "prefer discrete gpu"),
            keywords=("choose",),
            exact_terms=("choose my",),
        )
        encoder = MappingEncoder(
            {
                "select graphics card": (1, 0, 0),
                "prefer discrete gpu": (1, 0, 0),
                "which option should i choose": (1, 0, 0),
            }
        )

        decision = SupportRetriever([card_value], encoder).retrieve(
            "which option should i choose"
        )

        self.assertEqual(decision.ranked[0].exact_score, 0)

    def test_lexical_only_answer_requires_minimum_semantic_relevance(self):
        service = support_card(
            "logs.service",
            ("service returned an error", "inspect service exceptions"),
            keywords=("service", "return", "exceptions"),
        )
        other = support_card(
            "display.mode",
            ("monitor mode", "screen resolution"),
        )
        encoder = MappingEncoder(
            {
                "service returned an error": (1, 0, 0),
                "inspect service exceptions": (1, 0, 0),
                "monitor mode": (0, 1, 0),
                "screen resolution": (0, 1, 0),
                "asyncio service return exceptions": (0, 0, 1),
            }
        )

        decision = SupportRetriever([service, other], encoder).retrieve(
            "asyncio service return exceptions"
        )

        self.assertEqual(decision.kind, "no_match")

    def test_non_ryoku_query_needs_strong_semantic_relevance(self):
        updates = support_card(
            "updates.rollback",
            ("update broke system state", "return after failed update"),
            keywords=("update", "state"),
        )
        encoder = MappingEncoder(
            {
                "update broke system state": (1, 0, 0),
                "return after failed update": (1, 0, 0),
                "react state update renders twice": (0.33, 0.94, 0),
            }
        )

        decision = SupportRetriever([updates], encoder).retrieve(
            "react state update renders twice"
        )

        self.assertEqual(decision.kind, "no_match")

    def test_single_generic_lexical_match_is_rejected(self):
        card_value = support_card(
            "hardware.gpu-detect",
            ("what gpu is recommended", "detect graphics devices"),
            keywords=("recommend",),
        )
        encoder = MappingEncoder(
            {
                "what gpu is recommended": (1, 0, 0),
                "detect graphics devices": (1, 0, 0),
                "recommend dinner": (0.19, 0.98, 0),
            }
        )

        decision = SupportRetriever([card_value], encoder).retrieve(
            "recommend dinner"
        )

        self.assertEqual(decision.kind, "no_match")

    def test_unrelated_tied_candidates_are_rejected(self):
        first = support_card(
            "topic.first",
            ("shared diagnostic topic", "inspect shared issue"),
            keywords=("shared", "diagnostic"),
        )
        second = support_card(
            "topic.second",
            ("shared diagnostic topic", "inspect shared issue"),
            keywords=("shared", "diagnostic"),
        )
        encoder = MappingEncoder(
            {
                "shared diagnostic topic": (1, 0, 0),
                "inspect shared issue": (1, 0, 0),
                "shared diagnostic issue": (1, 0, 0),
            }
        )

        decision = SupportRetriever([first, second], encoder).retrieve(
            "shared diagnostic issue"
        )

        self.assertEqual(decision.kind, "no_match")

    def test_repeated_query_is_deterministic(self):
        card_value = support_card(
            "health.doctor",
            ("inspect drift", "repair system"),
            keywords=("doctor", "repair"),
        )
        encoder = MappingEncoder(
            {
                "inspect drift": (1, 0, 0),
                "repair system": (1, 0, 0),
                "doctor repair": (1, 0, 0),
            }
        )
        retriever = SupportRetriever([card_value], encoder)

        decisions = [retriever.retrieve("doctor repair") for _ in range(100)]

        self.assertEqual(decisions, [decisions[0]] * 100)

class RealCatalogTests(unittest.TestCase):
    def test_expanded_catalog_is_valid_and_covers_benchmark_answers(self):
        cards = load_support_cards(Path("data/support.json"))
        card_ids = {card.id for card in cards}
        benchmark = json.loads(
            Path("data/benchmark.json").read_text(encoding="utf-8")
        )
        expected_ids = {
            row["intent_id"]
            for row in benchmark["cases"]
            if row.get("intent_id") is not None
        }

        self.assertGreaterEqual(len(cards), 40)
        self.assertGreaterEqual(sum(len(card.examples) for card in cards), 200)
        self.assertEqual(expected_ids - card_ids, set())

    def test_catalog_hub_paths_exist_in_gui_map(self):
        cards = load_support_cards(Path("data/support.json"))
        gui_map = json.loads(
            Path("data/gui_map.json").read_text(encoding="utf-8")
        )
        valid_paths = {
            " ".join(entry["path"].split()).casefold()
            for entry in gui_map["entries"]
            if entry["kind"] in {"page", "setting"}
        }
        breadcrumb_pattern = re.compile(
            r"(?:Ryoku Hub|Ryoku Settings)\s*>\s*[^.;,)\u2014\n`]+",
            re.IGNORECASE,
        )
        invalid = []

        for catalog_card in cards:
            for match in breadcrumb_pattern.finditer(catalog_card.answer):
                path = match.group(0).strip(" \t`*")
                path = re.sub(
                    r"^Ryoku Settings\b",
                    "Ryoku Hub",
                    path,
                    flags=re.IGNORECASE,
                )
                normalized = " ".join(path.split()).casefold()
                exists = any(
                    normalized == valid
                    or (
                        normalized.startswith(valid)
                        and not normalized[len(valid)].isalnum()
                    )
                    for valid in valid_paths
                )
                if not exists:
                    invalid.append((catalog_card.id, path))

        self.assertEqual(
            invalid,
            [],
            "Catalog breadcrumbs missing from data/gui_map.json "
            f"(card id, path): {invalid}",
        )

    def test_shell_recovery_card_is_state_changing_without_rollback_advice(self):
        cards = load_support_cards(Path("data/support.json"))
        shell = next(card for card in cards if card.id == "shell.missing")

        self.assertEqual(shell.risk, "state-changing")
        self.assertNotIn("roll back", shell.answer.lower())

    def test_vague_newcomer_health_question_never_routes_to_a_risky_card(self):
        cards = load_support_cards(Path("data/support.json"))
        decision = SupportRetriever(cards, MappingEncoder({})).retrieve(
            "I'm new to Ryoku and just want a safe health check before I touch anything."
        )

        served = decision.card if decision.kind == "answer" else None
        if decision.kind == "clarify" and decision.alternatives:
            served = decision.alternatives[0]
        if served is not None:
            self.assertEqual(served.risk, "informational")

    def test_benchmark_queries_are_not_catalog_examples_or_exact_terms(self):
        cards = load_support_cards(Path("data/support.json"))
        benchmark = json.loads(
            Path("data/benchmark.json").read_text(encoding="utf-8")
        )
        catalog_text = {
            normalize_text(value)
            for card in cards
            for value in card.examples + card.exact_terms
        }

        duplicates = [
            row["id"]
            for row in benchmark["cases"]
            if normalize_text(row["query"]) in catalog_text
        ]

        self.assertEqual(duplicates, [])

if __name__ == "__main__":
    unittest.main()
