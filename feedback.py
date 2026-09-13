from __future__ import annotations

import sqlite3
from pathlib import Path

import discord


VERDICTS = ("correct", "partially_correct", "incorrect")
DECISION_KINDS = ("answer", "clarify", "no_match", "safety", "unknown")
MODEL_ROUTES = ("gemma", "lfm", "deterministic")
SOURCE_STATUSES = ("ok", "no_match", "unavailable", "not_requested")
ANSWER_REVISION = 2
BUTTON_LABELS = {
    "correct": "Correct",
    "partially_correct": "Partially correct",
    "incorrect": "Incorrect",
}


class FeedbackButton(discord.ui.Button):
    def __init__(self, verdict: str, store: "FeedbackStore"):
        super().__init__(
            label=BUTTON_LABELS[verdict],
            style={
                "correct": discord.ButtonStyle.success,
                "partially_correct": discord.ButtonStyle.secondary,
                "incorrect": discord.ButtonStyle.danger,
            }[verdict],
            custom_id=f"nero_feedback:{verdict}",
        )
        self.store = store
        self.verdict = verdict

    async def callback(self, interaction: discord.Interaction) -> None:
        message = interaction.message
        if message is None:
            await interaction.response.send_message(
                "That feedback target is unavailable.", ephemeral=True
            )
            return
        outcome = self.store.record_feedback(
            answer_message_id=message.id,
            actor_id=interaction.user.id,
            verdict=self.verdict,
        )
        response = {
            "recorded": f"Marked {self.verdict}. Thanks.",
            "updated": f"Updated to {self.verdict}.",
            "not_requester": "Only the person who asked can rate this answer.",
            "unknown_answer": "That answer is no longer available for feedback.",
        }[outcome]
        await interaction.response.send_message(response, ephemeral=True)


class FeedbackView(discord.ui.View):
    def __init__(self, store: "FeedbackStore"):
        super().__init__(timeout=None)
        self.add_item(FeedbackButton("correct", store))
        self.add_item(FeedbackButton("partially_correct", store))
        self.add_item(FeedbackButton("incorrect", store))


class FeedbackStore:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS answers (
                    answer_message_id INTEGER PRIMARY KEY,
                    request_message_id INTEGER NOT NULL,
                    requester_id INTEGER NOT NULL,
                    decision_kind TEXT NOT NULL DEFAULT 'unknown',
                    card_id TEXT,
                    model_route TEXT NOT NULL DEFAULT 'deterministic',
                    source_status TEXT NOT NULL DEFAULT 'not_requested',
                    answer_revision INTEGER NOT NULL DEFAULT 1
                );
                CREATE TABLE IF NOT EXISTS feedback (
                    answer_message_id INTEGER NOT NULL,
                    actor_id INTEGER NOT NULL,
                    verdict TEXT NOT NULL CHECK (verdict IN ('correct', 'partially_correct', 'incorrect')),
                    PRIMARY KEY (answer_message_id, actor_id),
                    FOREIGN KEY (answer_message_id) REFERENCES answers(answer_message_id)
                );
                """
            )
            self._migrate_answers_schema(connection)
            self._migrate_feedback_schema(connection)
            connection.execute(f"PRAGMA user_version = {ANSWER_REVISION}")

    @staticmethod
    def _migrate_answers_schema(connection: sqlite3.Connection) -> None:
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(answers)")
        }
        additions = (
            ("decision_kind", "TEXT NOT NULL DEFAULT 'unknown'"),
            ("card_id", "TEXT"),
            ("model_route", "TEXT NOT NULL DEFAULT 'deterministic'"),
            ("source_status", "TEXT NOT NULL DEFAULT 'not_requested'"),
            ("answer_revision", "INTEGER NOT NULL DEFAULT 1"),
        )
        for name, definition in additions:
            if name not in columns:
                connection.execute(f"ALTER TABLE answers ADD COLUMN {name} {definition}")

    @staticmethod
    def _migrate_feedback_schema(connection: sqlite3.Connection) -> None:
        definition = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'feedback'"
        ).fetchone()[0]
        if "partially_correct" in definition:
            return
        connection.executescript(
            """
            ALTER TABLE feedback RENAME TO feedback_legacy;
            CREATE TABLE feedback (
                answer_message_id INTEGER NOT NULL,
                actor_id INTEGER NOT NULL,
                verdict TEXT NOT NULL CHECK (verdict IN ('correct', 'partially_correct', 'incorrect')),
                PRIMARY KEY (answer_message_id, actor_id),
                FOREIGN KEY (answer_message_id) REFERENCES answers(answer_message_id)
            );
            INSERT INTO feedback (answer_message_id, actor_id, verdict)
            SELECT answer_message_id, actor_id, verdict FROM feedback_legacy;
            DROP TABLE feedback_legacy;
            """
        )

    def record_answer(
        self,
        *,
        answer_message_id: int,
        request_message_id: int,
        requester_id: int,
        decision_kind: str = "unknown",
        card_id: str | None = None,
        model_route: str = "deterministic",
        source_status: str = "not_requested",
        answer_revision: int = ANSWER_REVISION,
    ) -> None:
        if decision_kind not in DECISION_KINDS:
            raise ValueError("invalid decision_kind")
        if model_route not in MODEL_ROUTES:
            raise ValueError("invalid model_route")
        if source_status not in SOURCE_STATUSES:
            raise ValueError("invalid source_status")
        if not 1 <= answer_revision <= ANSWER_REVISION:
            raise ValueError("invalid answer_revision")
        if card_id is not None and (
            not isinstance(card_id, str)
            or not card_id
            or len(card_id) > 120
            or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789._-" for character in card_id)
        ):
            raise ValueError("invalid card_id")
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO answers (
                    answer_message_id, request_message_id, requester_id, decision_kind,
                    card_id, model_route, source_status, answer_revision
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    answer_message_id,
                    request_message_id,
                    requester_id,
                    decision_kind,
                    card_id,
                    model_route,
                    source_status,
                    answer_revision,
                ),
            )

    def record_feedback(
        self,
        answer_message_id: int,
        actor_id: int,
        verdict: str,
    ) -> str:
        if verdict not in VERDICTS:
            raise ValueError(f"verdict must be one of {', '.join(VERDICTS)}")
        with self._connect() as connection:
            answer = connection.execute(
                "SELECT requester_id FROM answers WHERE answer_message_id = ?",
                (answer_message_id,),
            ).fetchone()
            if answer is None:
                return "unknown_answer"
            if answer[0] != actor_id:
                return "not_requester"
            existing = connection.execute(
                """
                SELECT verdict FROM feedback
                WHERE answer_message_id = ? AND actor_id = ?
                """,
                (answer_message_id, actor_id),
            ).fetchone()
            connection.execute(
                """
                INSERT INTO feedback (answer_message_id, actor_id, verdict)
                VALUES (?, ?, ?)
                ON CONFLICT(answer_message_id, actor_id)
                DO UPDATE SET verdict = excluded.verdict
                """,
                (answer_message_id, actor_id, verdict),
            )
        return "updated" if existing is not None else "recorded"

    def recent_feedback(self, limit: int = 100) -> list[dict[str, int | str]]:
        if not 1 <= limit <= 500:
            raise ValueError("limit must be between 1 and 500")
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    answers.answer_message_id,
                    answers.request_message_id,
                    answers.requester_id,
                    answers.decision_kind,
                    answers.card_id,
                    answers.model_route,
                    answers.source_status,
                    answers.answer_revision,
                    feedback.actor_id,
                    feedback.verdict
                FROM feedback
                JOIN answers USING (answer_message_id)
                ORDER BY answers.answer_message_id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [
            {
                "answer_message_id": row[0],
                "request_message_id": row[1],
                "requester_id": row[2],
                "decision_kind": row[3],
                "card_id": row[4],
                "model_route": row[5],
                "source_status": row[6],
                "answer_revision": row[7],
                "actor_id": row[8],
                "verdict": row[9],
            }
            for row in rows
        ]

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.execute("PRAGMA foreign_keys = ON")
        return connection
