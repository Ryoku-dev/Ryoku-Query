import unittest

from prowl import ProwlResult, SourceHit
from src.agent.cloud import LLMResult
from src.agent.pipeline import SupportPipeline, Turn
from src.agent.prompt import build_messages
from src.knowledge.index import Evidence
from support import RetrievalDecision, SupportCard


def make_card(
    *,
    card_id="desktop.frost",
    title="Deck frost",
    answer="Open Ryoku Hub > Desktop > General and turn off Deck frost.",
    risk="informational",
):
    return SupportCard(
        id=card_id,
        title=title,
        examples=("turn off frost",),
        keywords=("frost",),
        exact_terms=("deck frost",),
        answer=answer,
        risk=risk,
        docs_url="https://docs.ryoku.dev/docs/desktop#deck-frost",
        source_hints=("src/desktop/frost.go",),
        clarifies_with=(),
    )


def card_evidence(card):
    return Evidence(
        kind="card",
        id=f"card:{card.id}",
        title=card.title,
        text=card.answer,
        path="Ryoku Hub > Desktop > General",
        url=card.docs_url,
        risk=card.risk,
        score=1.0,
    )


def gui_evidence(
    *,
    evidence_id="setting:desktop/deckFrostEnabled",
    title="Deck frost",
    text="Turn Deck frost on or off.",
    path="Ryoku Hub > Desktop > General > Deck frost",
    channels=frozenset(),
    stable_version=None,
    source_commit=None,
    score=0.9,
):
    return Evidence(
        kind="gui",
        id=evidence_id,
        title=title,
        text=text,
        path=path,
        url=None,
        risk=None,
        score=score,
        channels=channels,
        stable_version=stable_version,
        source_commit=source_commit,
    )


class FakeIndex:
    def __init__(
        self, evidence, decision, cards=(), known_paths=(), stable_cli_usage=()
    ):
        self.evidence = list(evidence)
        self.decision = decision
        self.cards = tuple(cards)
        self._known_paths = frozenset(known_paths)
        self._stable_cli_usage = dict(stable_cli_usage)
        self.stable_version = next(
            (item.stable_version for item in self.evidence if item.stable_version),
            None,
        )
        self.search_queries = []

    def search(self, query, k=10):
        self.search_queries.append(query)
        return self.evidence[:k]

    def card_decision(self, query):
        return self.decision

    def known_paths(self):
        return self._known_paths

    def stable_cli_usage(self, command):
        return self._stable_cli_usage.get(command, "")


    def card(self, card_id):
        return next((card for card in self.cards if card.id == card_id), None)


class FakeCloud:
    def __init__(self, result, model="gpt-6-luna"):
        self.result = result
        self.model = model
        self.calls = []

    async def complete(self, messages, *, max_tokens=500):
        self.calls.append((messages, max_tokens))
        return self.result


class FakeProwl:
    def __init__(self, result):
        self.result = result
        self.calls = []

    async def search(self, query, source_hints=()):
        self.calls.append((query, source_hints))
        return self.result


class PipelineTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.card = make_card()
        self.evidence = [card_evidence(self.card), gui_evidence()]
        self.known_paths = {
            "Ryoku Hub > Desktop",
            "Ryoku Hub > Desktop > General",
            "Ryoku Hub > Desktop > General > Deck frost",
        }

    def pipeline(
        self,
        cloud,
        *,
        decision=None,
        evidence=None,
        cards=None,
        prowl=None,
        known_paths=None,
        stable_cli_usage=(),
    ):
        index = FakeIndex(
            self.evidence if evidence is None else evidence,
            decision or RetrievalDecision("answer", card=self.card),
            (self.card,) if cards is None else cards,
            self.known_paths if known_paths is None else known_paths,
            stable_cli_usage,
        )
        return SupportPipeline(index, cloud, prowl), index

    async def test_destructive_request_uses_safety_route_without_cloud(self):
        cloud = FakeCloud(LLMResult("ok", text="unsafe"))
        pipeline, _ = self.pipeline(cloud)

        answer = await pipeline.answer("wipe everything and reinstall")

        self.assertEqual(answer.route, "safety")
        self.assertIn("whether the desktop or a TTY still works", answer.text)
        self.assertEqual(cloud.calls, [])
        self.assertIsNone(answer.model)

    async def test_consequence_question_about_update_calls_cloud(self):
        cloud = FakeCloud(
            LLMResult(
                "ok",
                text="Open Ryoku Hub > Desktop > General > Deck frost and turn it off.",
            )
        )
        pipeline, _ = self.pipeline(cloud)

        answer = await pipeline.answer("Will an update erase hypr/user.lua?")

        self.assertEqual(answer.route, "cloud")
        self.assertEqual(len(cloud.calls), 1)

    async def test_portuguese_consequence_question_calls_cloud(self):
        cloud = FakeCloud(
            LLMResult(
                "ok",
                text="Open Ryoku Hub > Desktop > General > Deck frost and turn it off.",
            )
        )
        pipeline, _ = self.pipeline(cloud)

        answer = await pipeline.answer("vai apagar minhas configurações?")

        self.assertEqual(answer.route, "cloud")
        self.assertEqual(len(cloud.calls), 1)

    async def test_destructive_card_is_verbatim_and_never_calls_cloud(self):
        reviewed = "Use the recovery screen, then choose the reviewed rollback entry."
        card = make_card(risk="destructive", answer=reviewed)
        cloud = FakeCloud(LLMResult("ok", text="Paraphrased procedure"))
        decision = RetrievalDecision("answer", card=card)
        pipeline, _ = self.pipeline(
            cloud,
            decision=decision,
            evidence=[card_evidence(card)],
            cards=(card,),
        )

        answer = await pipeline.answer("How do I use the recovery screen?")

        self.assertEqual(answer.route, "card")
        self.assertEqual(answer.text, reviewed)
        self.assertEqual(answer.risk, "destructive")
        self.assertEqual(cloud.calls, [])

    async def test_valid_cloud_answer_uses_cloud_route(self):
        cloud = FakeCloud(
            LLMResult(
                "ok",
                text="Open Ryoku Hub > Desktop > General > Deck frost and turn it off.",
            )
        )
        pipeline, _ = self.pipeline(cloud)

        answer = await pipeline.answer("How do I turn off the frosted desktop?")

        self.assertEqual(answer.route, "cloud")
        self.assertEqual(answer.model, "gpt-6-luna")
        self.assertEqual(answer.card_id, self.card.id)
        self.assertEqual(cloud.calls[0][1], 450)

    async def test_known_hub_path_followed_by_comma_is_accepted(self):
        cloud = FakeCloud(
            LLMResult(
                "ok",
                text=(
                    "Open Ryoku Hub > Desktop > General > Deck frost, "
                    "then turn it off."
                ),
            )
        )
        pipeline, _ = self.pipeline(cloud)

        answer = await pipeline.answer("How do I turn off frost?")

        self.assertEqual(answer.route, "cloud")

    async def test_arrow_breadcrumb_is_validated(self):
        cloud = FakeCloud(
            LLMResult(
                "ok",
                text="Open Ryoku Hub → Desktop › General » Deck frost and turn it off.",
            )
        )
        pipeline, _ = self.pipeline(cloud)

        answer = await pipeline.answer("How do I turn off frost?")

        self.assertEqual(answer.route, "cloud")

    async def test_gui_paths_allow_prose_after_the_longest_known_label(self):
        known_paths = {
            "Ryoku Hub > General > Region & fonts > System font",
            "Ryoku Hub > Updates",
            "Ryoku Hub > Keybinds",
            "Ryoku Hub > Recording",
            "Ryoku Hub > Performance > Reduce motion",
        }
        answers = (
            (
                "Open Ryoku Hub > General > Region & fonts for "
                "**System font**—that controls the shell font."
            ),
            "Open Ryoku Hub > Updates with Super+comma to access the release channel.",
            "Open Ryoku Hub > Updates with Super+comma, then choose Update now.",
            (
                "Abra Ryoku Hub > Keybinds com Super+comma; o Kit é apenas "
                "para consultar atalhos."
            ),
            "Abra Ryoku Hub > Recording com Super+comma.",
            (
                "Ryoku Hub > Performance > Reduce motion does not control "
                "Close animation."
            ),
        )

        for text in answers:
            with self.subTest(text=text):
                cloud = FakeCloud(LLMResult("ok", text=text))
                pipeline, _ = self.pipeline(cloud, known_paths=known_paths)

                answer = await pipeline.answer("Where is that setting?")

                self.assertEqual(answer.route, "cloud")
                self.assertEqual(answer.note, "")

    async def test_unmapped_surface_path_is_not_validated(self):
        cloud = FakeCloud(
            LLMResult(
                "ok",
                text=(
                    "Ryogami > Settings > Advanced > General > Restore "
                    "wallpaper on reload."
                ),
            )
        )
        pipeline, _ = self.pipeline(cloud)

        answer = await pipeline.answer("How do I restore my wallpaper?")

        self.assertEqual(answer.route, "cloud")
        self.assertEqual(answer.note, "")


    async def test_evidence_citations_are_stripped(self):
        cloud = FakeCloud(
            LLMResult(
                "ok",
                text=(
                    "Open Ryoku Hub > Desktop > General > Deck frost and "
                    "turn it off. [E7]"
                ),
            )
        )
        pipeline, _ = self.pipeline(cloud)

        answer = await pipeline.answer("How do I turn off frost?")

        self.assertEqual(answer.route, "cloud")
        self.assertNotIn("[E7]", answer.text)

    async def test_cloud_clarify_prefix_returns_only_the_question(self):
        cloud = FakeCloud(
            LLMResult("ok", text="clarify: Are you using niri or Hyprland?")
        )
        pipeline, _ = self.pipeline(cloud)

        answer = await pipeline.answer("Where is that setting?")

        self.assertEqual(answer.route, "clarify")
        self.assertEqual(answer.text, "Are you using niri or Hyprland?")

    async def test_cloud_no_answer_without_relevant_items_has_no_related_list(self):
        cloud = FakeCloud(LLMResult("ok", text="NO_ANSWER."))
        pipeline, _ = self.pipeline(cloud)

        answer = await pipeline.answer("Can Ryoku make coffee?")

        self.assertEqual(answer.route, "no_match")
        self.assertNotIn("Related:", answer.text)
        self.assertNotIn("Deck frost", answer.text)
        self.assertIn("#help", answer.text)
        self.assertIn("`ryoku doctor --check`", answer.text)

    async def test_cloud_no_answer_hides_below_threshold_related_item(self):
        cloud = FakeCloud(LLMResult("ok", text="NO_ANSWER."))
        evidence = [
            gui_evidence(
                title="Wallpaper picker",
                path="Super+W",
                score=0.029,
            )
        ]
        pipeline, _ = self.pipeline(cloud, evidence=evidence)

        answer = await pipeline.answer("Where is the wallpaper picker?")

        self.assertEqual(answer.route, "no_match")
        self.assertNotIn("Related:", answer.text)


    async def test_invented_hub_path_falls_back_to_card(self):
        cloud = FakeCloud(
            LLMResult("ok", text="Open Ryoku Hub > Teleportation > Wormholes.")
        )
        pipeline, _ = self.pipeline(cloud)

        answer = await pipeline.answer("How do I turn off frost?")

        self.assertEqual(answer.route, "card")
        self.assertEqual(answer.text, self.card.answer)
        self.assertTrue(answer.note.startswith("invalid:hub_path:"))

    async def test_invented_child_under_known_hub_path_falls_back_to_card(self):
        cloud = FakeCloud(
            LLMResult(
                "ok",
                text=(
                    "Open Ryoku Hub > Desktop > General > Deck frost "
                    "> Intensity."
                ),
            )
        )
        pipeline, _ = self.pipeline(cloud)

        answer = await pipeline.answer("How do I turn off frost?")

        self.assertEqual(answer.route, "card")
        self.assertEqual(answer.text, self.card.answer)
        self.assertTrue(answer.note.startswith("invalid:hub_path:"))

    async def test_known_stage_editor_path_is_validated(self):
        path = "Stage Editor > Desktop > Widgets"
        cloud = FakeCloud(LLMResult("ok", text=f"Open {path}, then add one."))
        evidence = [
            gui_evidence(
                evidence_id="surface:stage-editor/widgets",
                title="Desktop widgets",
                text="Add a desktop widget.",
                path=path,
            )
        ]
        pipeline, _ = self.pipeline(
            cloud,
            evidence=evidence,
            known_paths={path},
        )

        answer = await pipeline.answer("How do I add a desktop widget?")

        self.assertEqual(answer.route, "cloud")


    async def test_invented_ryoku_settings_path_falls_back_to_card(self):
        cloud = FakeCloud(
            LLMResult("ok", text="Open Ryoku Settings > Advanced > Rashin.")
        )
        pipeline, _ = self.pipeline(cloud)

        answer = await pipeline.answer("How do I turn off frost?")

        self.assertEqual(answer.route, "card")
        self.assertEqual(answer.text, self.card.answer)
        self.assertTrue(answer.note.startswith("invalid:hub_path:"))

    async def test_invented_ryoku_verb_falls_back_to_card(self):
        cloud = FakeCloud(LLMResult("ok", text="Run `ryoku teleport` to fix it."))
        pipeline, _ = self.pipeline(cloud)

        answer = await pipeline.answer("How do I turn off frost?")

        self.assertEqual(answer.route, "card")
        self.assertEqual(answer.text, self.card.answer)
        self.assertEqual(answer.note, "invalid:command:ryoku teleport")

    async def test_evidenced_ryoku_command_with_invented_flag_falls_back(self):
        cloud = FakeCloud(
            LLMResult("ok", text="Run `ryoku update --teleport` to fix it.")
        )
        evidence = [
            card_evidence(self.card),
            gui_evidence(
                title="Update Ryoku",
                text="Run ryoku update to bring Ryoku current.",
            ),
        ]
        pipeline, _ = self.pipeline(cloud, evidence=evidence)

        answer = await pipeline.answer("How do I update?")

        self.assertEqual(answer.route, "card")
        self.assertEqual(answer.note, "invalid:command:ryoku update --teleport")

    async def test_stable_cli_usage_allows_documented_flag(self):
        cloud = FakeCloud(LLMResult("ok", text="Run `ryoku update --wait`."))
        evidence = [
            gui_evidence(
                title="Update Ryoku",
                text="Run ryoku update to bring Ryoku current.",
            )
        ]
        pipeline, _ = self.pipeline(
            cloud,
            evidence=evidence,
            stable_cli_usage=(("ryoku update", "ryoku update --wait"),),
        )

        answer = await pipeline.answer("How do I wait for an update?")

        self.assertEqual(answer.route, "cloud")


    async def test_system_prompt_commands_are_allowed_without_evidence(self):
        commands = (
            "ryoku doctor",
            "ryoku doctor --check",
            "ryoku doctor --report",
            "ryoku status",
        )
        for command in commands:
            with self.subTest(command=command):
                cloud = FakeCloud(LLMResult("ok", text=f"Run `{command}`."))
                pipeline, _ = self.pipeline(cloud)

                answer = await pipeline.answer("What should I run?")

                self.assertEqual(answer.route, "cloud")
                self.assertEqual(answer.note, "")

    async def test_unavailable_cloud_falls_back_to_card(self):
        cloud = FakeCloud(LLMResult("unavailable", error="busy"))
        pipeline, _ = self.pipeline(cloud)

        answer = await pipeline.answer("How do I turn off frost?")

        self.assertEqual(answer.route, "card")
        self.assertEqual(answer.text, self.card.answer)
        self.assertEqual(answer.model, cloud.model)
        self.assertEqual(answer.note, "cloud_unavailable:busy")

    async def test_disagreeing_card_fallback_lists_related_evidence(self):
        cloud = FakeCloud(
            LLMResult("ok", text="Open Ryoku Hub > Advanced > Rashin.")
        )
        evidence = [gui_evidence()]
        pipeline, _ = self.pipeline(cloud, evidence=evidence)

        answer = await pipeline.answer("How do I turn off frost?")

        self.assertEqual(answer.route, "no_match")
        self.assertEqual(answer.note, "card_disagrees")
        self.assertIn("#help", answer.text)

    async def test_unavailable_cloud_without_card_lists_relevant_evidence(self):
        cloud = FakeCloud(LLMResult("unavailable", error="timeout"))
        evidence = [
            gui_evidence(title="Wallpaper picker", path="Super+W"),
            gui_evidence(
                evidence_id="page:desktop",
                title="Desktop page",
                text="Desktop controls.",
                path="Ryoku Hub > Desktop",
            ),
        ]
        pipeline, _ = self.pipeline(
            cloud,
            decision=RetrievalDecision("no_match"),
            evidence=evidence,
            cards=(),
        )

        answer = await pipeline.answer("Where are wallpaper controls?")

        self.assertEqual(answer.route, "no_match")
        self.assertIn("Wallpaper picker — Super+W", answer.text)
        self.assertNotIn("Desktop page — Ryoku Hub > Desktop", answer.text)
        self.assertIn("#help", answer.text)
        self.assertEqual(answer.note, "cloud_unavailable:timeout")

    async def test_short_follow_up_searches_with_previous_user_turn(self):
        cloud = FakeCloud(LLMResult("unavailable", error="offline"))
        pipeline, index = self.pipeline(cloud)
        history = (
            Turn("user", "How do I change the desktop frost?"),
            Turn("nero", "Open the Desktop page."),
        )

        await pipeline.answer("Where exactly?", history)

        self.assertEqual(
            index.search_queries,
            ["How do I change the desktop frost? Where exactly?"],
        )

    async def test_source_answer_uses_citations_and_source_route(self):
        cloud = FakeCloud(LLMResult("ok", text="The setting is defined in the cited Go file."))
        prowl = FakeProwl(
            ProwlResult(
                "ok",
                hits=(SourceHit("src/desktop/frost.go", 12, 24, "Deck frost setting", False),),
            )
        )
        pipeline, _ = self.pipeline(cloud, prowl=prowl)

        answer = await pipeline.answer("Where in the source is deck frost defined?")

        self.assertEqual(answer.route, "source")
        self.assertEqual(answer.evidence_ids[0], "source:src/desktop/frost.go:12-24")
        self.assertEqual(prowl.calls[0][1], self.card.source_hints)
        self.assertIn("question is about Ryoku's code", cloud.calls[0][0][0]["content"])


class PromptTests(unittest.TestCase):
    def test_evidence_block_respects_character_limit(self):
        evidence = [
            gui_evidence(text="x" * 600),
            gui_evidence(
                evidence_id="page:global",
                title="Global",
                text="y" * 600,
                path="Ryoku Hub > Global",
            ),
        ]

        messages = build_messages(
            "Where is it?", evidence, (), max_evidence_chars=240
        )
        evidence_block = messages[1]["content"].split("\n\n", 1)[0]

        self.assertLessEqual(len(evidence_block), 240)
        self.assertTrue(evidence_block.startswith("Evidence:"))
        self.assertIn("[E1]", evidence_block)
        self.assertNotIn("[E2]", evidence_block)
        self.assertGreaterEqual(len(evidence_block.splitlines()[1]), 120)

    def test_prompt_renders_stable_version_and_current_evidence_precedence(self):
        old_card = make_card(
            card_id="widgets.desktop",
            title="Desktop widgets",
            answer="Desktop widgets are available only on unstable.",
        )
        evidence = [
            card_evidence(old_card),
            gui_evidence(
                evidence_id="surface:stage-editor",
                title="Stage Editor",
                text="Desktop widgets are available on stable.",
                path="Stage Editor > Desktop",
                channels=frozenset({"stable"}),
                stable_version="1.0.5",
                source_commit="e344edb",
            ),
        ]

        messages = build_messages("Can I use desktop widgets?", evidence, ())
        system = messages[0]["content"]
        evidence_block = messages[1]["content"]

        self.assertIn("Ryoku stable is v1.0.5 (the current release).", system)
        self.assertIn("current stable docs and the stable GUI map win", system)
        self.assertNotIn("owner <path>` exists only", system)
        self.assertIn("available only on unstable", evidence_block)
        self.assertIn(
            "(gui; channels=stable; stable=v1.0.5; commit=e344edb)",
            evidence_block,
        )

    def test_last_four_history_turns_are_included_and_bounded(self):
        history = (
            Turn("user", "old turn"),
            Turn("nero", "first kept"),
            Turn("user", "u" * 700),
            Turn("nero", "third kept"),
            Turn("user", "fourth kept"),
        )

        messages = build_messages("Now what?", self_evidence(), history)
        user_message = messages[1]["content"]

        self.assertNotIn("old turn", user_message)
        self.assertIn("Nero: first kept", user_message)
        self.assertIn("User: " + "u" * 600, user_message)
        self.assertNotIn("u" * 601, user_message)
        self.assertIn("Question: Now what?", user_message)


def self_evidence():
    return [gui_evidence()]


if __name__ == "__main__":
    unittest.main()
