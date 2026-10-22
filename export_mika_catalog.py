"""Export Nero's reviewed support catalog for Mika's grounded help bank."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from support import load_support_cards

_SCHEMA = "nero.mika-catalog/v1"


def build_catalog(support_path: Path) -> dict[str, Any]:
    """Return the allowlisted support fields Mika may index as evidence."""
    source = support_path.read_bytes()
    cards = load_support_cards(support_path)
    return {
        "schema": _SCHEMA,
        "source_sha256": hashlib.sha256(source).hexdigest(),
        "cards": [
            {
                "id": card.id,
                "title": card.title,
                "examples": list(card.examples),
                "answer": card.answer,
                "docs_url": card.docs_url,
                "exact_terms": list(card.exact_terms),
                "risk": card.risk,
            }
            for card in cards
            if card.docs_url is not None
        ],
    }


def write_catalog(support_path: Path, destination: Path) -> int:
    """Atomically publish the curated catalog without exporting telemetry or transcripts."""
    catalog = build_catalog(support_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pending = destination.with_suffix(destination.suffix + ".tmp")
    pending.write_text(
        json.dumps(catalog, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    pending.replace(destination)
    return len(catalog["cards"])


def main() -> int:
    """Write a catalog snapshot suitable for transfer to Mika."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--support", type=Path, default=Path("data/support.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(f"exported_cards={write_catalog(args.support, args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
