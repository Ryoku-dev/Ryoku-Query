import os
import unittest
from pathlib import Path
from unittest.mock import patch

from bot import UserRateLimiter, extract_query, handle_message, load_config
from src.agent.pipeline import PipelineAnswer, Turn


class User:
    def __init__(self, user_id, bot=False):
        self.id = user_id
        self.bot = bot


class Reference:
    def __init__(self, resolved=None, message_id=None):
        self.resolved = resolved
        self.message_id = (
            message_id if message_id is not None else getattr(resolved, "id", None)
        )


class TypingIndicator:
    def __init__(self, channel):
        self.channel = channel

    async def __aenter__(self):
        self.channel.typing_enters += 1

    async def __aexit__(self, exc_type, exc, traceback):
        self.channel.typing_exits += 1


class Channel:
    def __init__(
        self,
        channel_id=0,
        *,
        parent_id=None,
        name="support",
    ):
        self.id = channel_id
        self.parent_id = parent_id
        self.name = name
        self.calls = []
        self.messages = {}
        self.typing_enters = 0
        self.typing_exits = 0
        self.response = None

    def typing(self):
        return TypingIndicator(self)

    async def fetch_message(self, message_id):
        return self.messages[message_id]

    async def send(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class Message:
    def __init__(
        self,
        content,
        *,
        message_id=1,
        author=None,
        mentions=(),
        reference=None,
        channel=None,
        channel_id=0,
        embeds=(),
    ):
        self.id = message_id
        self.content = content
        self.author = author or User(7)
        self.mentions = list(mentions)
        self.reference = reference
        self.channel = channel or Channel(channel_id)
        self.embeds = list(embeds)
        self.reactions = []

    async def reply(self, **kwargs):
        return await self.channel.send(**kwargs)

    async def add_reaction(self, emoji):
        self.reactions.append(emoji)


class Pipeline:
    def __init__(self, answer=None):
        self.result = answer or pipeline_answer()
        self.queries = []
        self.histories = []

    async def answer(self, query, history=()):
        self.queries.append(query)
        self.histories.append(history)
        return self.result


def pipeline_answer(**overrides):
    values = {
        "text": "Open Ryoku Hub > General and enable the setting.",
        "route": "card",
        "title": "Ryoku setting",
        "evidence_ids": ("card:setting",),
        "card_id": "setting",
        "docs_url": "https://docs.ryoku.dev/settings",
        "risk": "informational",
        "model": None,
        "latency_ms": 2,
    }
    values.update(overrides)
    return PipelineAnswer(**values)


class InvocationTests(unittest.TestCase):
    def test_ordinary_message_is_ignored(self):
        self.assertIsNone(extract_query(Message("hello"), bot_user_id=42))

    def test_bot_message_is_ignored(self):
        message = Message("<@42> hello", author=User(42, bot=True))
        self.assertIsNone(extract_query(message, bot_user_id=42))

    def test_mention_is_removed_from_query(self):
        message = Message(
            "<@42>   how do I open settings?",
            mentions=(User(42, bot=True),),
        )
        self.assertEqual(
            extract_query(message, bot_user_id=42),
            "how do I open settings?",
        )

    def test_reply_to_bot_is_handled(self):
        nero = Message("answer", author=User(42, bot=True))
        message = Message(
            "where is that setting?",
            reference=Reference(nero),
        )
        self.assertEqual(
            extract_query(message, bot_user_id=42),
            "where is that setting?",
        )

    def test_support_channel_accepts_a_normal_question(self):
        message = Message("how do I open settings?", channel_id=777)
        self.assertEqual(
            extract_query(message, bot_user_id=42, support_channel_id=777),
            "how do I open settings?",
        )

    def test_support_channel_skips_short_non_request(self):
        message = Message("thanks!", channel_id=777)
        self.assertIsNone(
            extract_query(message, bot_user_id=42, support_channel_id=777)
        )

    def test_support_channel_skips_reply_to_another_member(self):
        other_member = Message("Try changing the scale.", author=User(8))
        message = Message(
            "Where is that setting?",
            reference=Reference(other_member),
            channel_id=777,
        )
        self.assertIsNone(
            extract_query(message, bot_user_id=42, support_channel_id=777)
        )

    def test_support_channel_skips_message_mentioning_another_member(self):
        message = Message(
            "<@8> where is that setting?",
            mentions=(User(8),),
            channel_id=777,
        )
        self.assertIsNone(
            extract_query(message, bot_user_id=42, support_channel_id=777)
        )

    def test_mention_bypasses_support_channel_filter(self):
        message = Message(
            "<@42> thanks!",
            mentions=(User(42, bot=True),),
            channel_id=777,
        )
        self.assertEqual(
            extract_query(message, bot_user_id=42, support_channel_id=777),
            "thanks!",
        )


class ConfigTests(unittest.TestCase):
    def test_loads_cloud_and_knowledge_configuration(self):
        values = {
            "TOKEN": "secret",
            "RYOKU_REPO_PATH": "/srv/ryoku-stable",
            "SUPPORT_PATH": "data/support-custom.json",
            "GUI_MAP_PATH": "data/gui-custom.json",
            "DOCS_CHUNKS_PATH": "data/docs-custom.json",
            "PROWL_TIMEOUT_SECONDS": "3.5",
            "PROWL_RESULT_LIMIT": "12",
            "PROWL_AGENT_PATH": "/opt/tools/prowl-agent",
            "SUPPORT_CHANNEL_ID": "123",
            "HELP_FORUM_ID": "456",
            "NERO_CLOUD": "0",
            "NERO_LLM_SOCKET": "/tmp/nero.sock",
            "NERO_LLM_MODEL": "gpt-6-luna",
            "NERO_LLM_TIMEOUT_SECONDS": "18",
            "NERO_USER_LIMIT": "3/45",
        }
        with patch("bot.load_dotenv"), patch.dict(
            os.environ,
            values,
            clear=True,
        ):
            config = load_config()

        self.assertEqual(config.token, "secret")
        self.assertEqual(config.ryoku_repo_path, Path("/srv/ryoku-stable"))
        self.assertEqual(config.support_path, Path("data/support-custom.json"))
        self.assertEqual(config.gui_map_path, Path("data/gui-custom.json"))
        self.assertEqual(config.docs_path, Path("data/docs-custom.json"))
        self.assertEqual(config.prowl_timeout, 3.5)
        self.assertEqual(config.prowl_result_limit, 12)
        self.assertEqual(config.prowl_executable, "/opt/tools/prowl-agent")
        self.assertEqual(config.support_channel_id, 123)
        self.assertEqual(config.help_forum_id, 456)
        self.assertEqual(config.user_limit_count, 3)
        self.assertEqual(config.user_limit_seconds, 45)
        self.assertFalse(config.cloud_enabled)
        self.assertEqual(config.llm_socket, "/tmp/nero.sock")
        self.assertEqual(config.llm_model, "gpt-6-luna")
        self.assertEqual(config.llm_timeout, 18)

    def test_cloud_configuration_defaults(self):
        with patch("bot.load_dotenv"), patch.dict(
            os.environ,
            {"TOKEN": "secret"},
            clear=True,
        ):
            config = load_config()

        self.assertTrue(config.cloud_enabled)
        self.assertEqual(config.llm_socket, "/run/nero-llm/llm.sock")
        self.assertEqual(config.llm_model, "gpt-6.1-sol")
        self.assertEqual(config.llm_timeout, 25)
        self.assertEqual(config.gui_map_path, Path("data/gui_map.json"))
        self.assertEqual(config.docs_path, Path("data/docs_chunks.json"))
        self.assertEqual(config.help_forum_id, 0)
        self.assertEqual(config.user_limit_count, 6)
        self.assertEqual(config.user_limit_seconds, 300)

    def test_rejects_invalid_cloud_switch(self):
        with patch("bot.load_dotenv"), patch.dict(
            os.environ,
            {"TOKEN": "secret", "NERO_CLOUD": "yes"},
            clear=True,
        ), self.assertRaisesRegex(RuntimeError, "NERO_CLOUD"):
            load_config()

    def test_rejects_llm_timeout_outside_range(self):
        for timeout in ("4.9", "60.1"):
            with self.subTest(timeout=timeout), patch(
                "bot.load_dotenv"
            ), patch.dict(
                os.environ,
                {"TOKEN": "secret", "NERO_LLM_TIMEOUT_SECONDS": timeout},
                clear=True,
            ), self.assertRaisesRegex(RuntimeError, "NERO_LLM_TIMEOUT_SECONDS"):
                load_config()

    def test_rejects_negative_help_forum_id(self):
        with patch("bot.load_dotenv"), patch.dict(
            os.environ,
            {"TOKEN": "secret", "HELP_FORUM_ID": "-1"},
            clear=True,
        ), self.assertRaisesRegex(RuntimeError, "HELP_FORUM_ID"):
            load_config()

    def test_zero_user_limit_disables_rate_limiting(self):
        with patch("bot.load_dotenv"), patch.dict(
            os.environ,
            {"TOKEN": "secret", "NERO_USER_LIMIT": "0"},
            clear=True,
        ):
            config = load_config()

        self.assertEqual(config.user_limit_count, 0)

    def test_rejects_invalid_user_limit(self):
        for value in ("bad", "2/0", "-1/60"):
            with self.subTest(value=value), patch(
                "bot.load_dotenv"
            ), patch.dict(
                os.environ,
                {"TOKEN": "secret", "NERO_USER_LIMIT": value},
                clear=True,
            ), self.assertRaisesRegex(RuntimeError, "NERO_USER_LIMIT"):
                load_config()


class HandlerTests(unittest.IsolatedAsyncioTestCase):
    async def test_help_forum_starter_is_answered_without_mention(self):
        channel = Channel(400, parent_id=300, name="Display scaling is wrong")
        message = Message(
            "The slider does not stick.",
            message_id=400,
            channel=channel,
        )
        pipeline = Pipeline()

        handled = await handle_message(
            message,
            42,
            pipeline,
            Path("missing.png"),
            help_forum_id=300,
        )

        self.assertTrue(handled)
        self.assertEqual(len(channel.calls), 1)
        self.assertTrue(pipeline.queries[0].startswith("Display scaling is wrong\n"))

    async def test_help_forum_nonstarter_is_ignored_without_mention(self):
        channel = Channel(400, parent_id=300, name="Display scaling is wrong")
        message = Message(
            "I have the same issue.",
            message_id=401,
            channel=channel,
        )
        pipeline = Pipeline()

        handled = await handle_message(
            message,
            42,
            pipeline,
            Path("missing.png"),
            help_forum_id=300,
        )

        self.assertFalse(handled)
        self.assertEqual(pipeline.queries, [])
        self.assertEqual(channel.calls, [])

    async def test_reply_chain_produces_oldest_first_history_with_nero_turn(self):
        first = Message("My display is too small.", message_id=10)
        nero = Message(
            "",
            message_id=11,
            author=User(42, bot=True),
            reference=Reference(first),
            embeds=(type("Embed", (), {"description": "Open Display settings."})(),),
        )
        message = Message(
            "Where is that?",
            message_id=12,
            reference=Reference(nero),
        )
        pipeline = Pipeline()

        await handle_message(
            message,
            42,
            pipeline,
            Path("missing.png"),
        )

        self.assertEqual(
            pipeline.histories[0],
            (
                Turn("user", "My display is too small."),
                Turn("nero", "Open Display settings."),
            ),
        )

    async def test_bots_are_ignored(self):
        message = Message(
            "<@42> how do I open settings?",
            author=User(8, bot=True),
            mentions=(User(42, bot=True),),
        )
        pipeline = Pipeline()

        handled = await handle_message(message, 42, pipeline, Path("missing.png"))

        self.assertFalse(handled)
        self.assertEqual(pipeline.queries, [])

    async def test_support_channel_question_is_answered_once(self):
        message = Message("Where is the display setting?", channel_id=777)
        pipeline = Pipeline()

        handled = await handle_message(
            message,
            42,
            pipeline,
            Path("missing.png"),
            support_channel_id=777,
        )

        self.assertTrue(handled)
        self.assertEqual(pipeline.queries, ["Where is the display setting?"])
        self.assertEqual(len(message.channel.calls), 1)
        call = message.channel.calls[0]
        self.assertEqual(call["embed"].title, "Ryoku setting")
        self.assertEqual(call["embed"].footer.text, "Nero • Ryoku support")
        self.assertTrue(call["mention_author"])
        self.assertFalse(call["allowed_mentions"].everyone)
        self.assertEqual(message.channel.typing_enters, 1)
        self.assertEqual(message.channel.typing_exits, 1)

    async def test_support_channel_thanks_is_skipped(self):
        message = Message("thanks!", channel_id=777)
        pipeline = Pipeline()

        handled = await handle_message(
            message,
            42,
            pipeline,
            Path("missing.png"),
            support_channel_id=777,
        )

        self.assertFalse(handled)
        self.assertEqual(pipeline.queries, [])

    async def test_mention_is_always_answered(self):
        message = Message(
            "<@42> thanks!",
            mentions=(User(42, bot=True),),
            channel_id=777,
        )
        pipeline = Pipeline()

        handled = await handle_message(
            message,
            42,
            pipeline,
            Path("missing.png"),
            support_channel_id=777,
        )

        self.assertTrue(handled)
        self.assertEqual(pipeline.queries, ["thanks!"])


    async def test_rate_limit_reacts_without_calling_pipeline(self):
        channel = Channel(777)
        pipeline = Pipeline()
        limiter = UserRateLimiter(2, 60)

        for message_id in (1, 2):
            handled = await handle_message(
                Message(
                    "Where is the display setting?",
                    message_id=message_id,
                    author=User(7),
                    channel=channel,
                ),
                42,
                pipeline,
                Path("missing.png"),
                support_channel_id=777,
                rate_limiter=limiter,
            )
            self.assertTrue(handled)

        limited = Message(
            "Where is the display setting?",
            message_id=3,
            author=User(7),
            channel=channel,
        )
        handled = await handle_message(
            limited,
            42,
            pipeline,
            Path("missing.png"),
            support_channel_id=777,
            rate_limiter=limiter,
        )

        self.assertFalse(handled)
        self.assertEqual(
            pipeline.queries,
            [
                "Where is the display setting?",
                "Where is the display setting?",
            ],
        )
        self.assertEqual(limited.reactions, ["⏳"])

    async def test_rate_limit_window_slides(self):
        channel = Channel(777)
        pipeline = Pipeline()
        clock = Clock()
        limiter = UserRateLimiter(1, 60, clock=clock)

        first = Message(
            "Where is the display setting?",
            author=User(7),
            channel=channel,
        )
        self.assertTrue(
            await handle_message(
                first,
                42,
                pipeline,
                Path("missing.png"),
                support_channel_id=777,
                rate_limiter=limiter,
            )
        )
        clock.now = 59
        blocked = Message(
            "Where is the display setting?",
            author=User(7),
            channel=channel,
        )
        self.assertFalse(
            await handle_message(
                blocked,
                42,
                pipeline,
                Path("missing.png"),
                support_channel_id=777,
                rate_limiter=limiter,
            )
        )
        clock.now = 60
        after_window = Message(
            "Where is the display setting?",
            author=User(7),
            channel=channel,
        )
        self.assertTrue(
            await handle_message(
                after_window,
                42,
                pipeline,
                Path("missing.png"),
                support_channel_id=777,
                rate_limiter=limiter,
            )
        )
        self.assertEqual(len(pipeline.queries), 2)

    async def test_zero_rate_limit_disables_limiter(self):
        channel = Channel(777)
        pipeline = Pipeline()
        limiter = UserRateLimiter(0, 60)

        for message_id in range(8):
            handled = await handle_message(
                Message(
                    "Where is the display setting?",
                    message_id=message_id,
                    author=User(7),
                    channel=channel,
                ),
                42,
                pipeline,
                Path("missing.png"),
                support_channel_id=777,
                rate_limiter=limiter,
            )
            self.assertTrue(handled)

        self.assertEqual(len(pipeline.queries), 8)

if __name__ == "__main__":
    unittest.main()
