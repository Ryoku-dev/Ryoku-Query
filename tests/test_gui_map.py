import shutil
import tempfile
import unittest
from pathlib import Path

from tools.build_docs_chunks import collect_chunks
from tools.build_gui_map import collect_entries, merge_channel_entries


@unittest.skipUnless(shutil.which("node"), "node is required to evaluate QML schemas")
class GuiMapTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.repo = Path(directory.name)
        hub_dir = self.repo / "ryoku/hub/quickshell"
        schema_dir = hub_dir / "schema"
        schema_dir.mkdir(parents=True)
        (hub_dir / "Hub.qml").write_text(
            '''import "schema/DesktopPage.js" as DesktopSchema
import "schema/GlobalPage.js" as GlobalSchema
readonly property var groups: [
 { name: "DESKTOP", items: [
  { key: "desktop", name: "Desktop" },
  { key: "global", name: "General" }
 ] }
]
readonly property var sectionRows: ({
 "desktop": DesktopSchema.rows,
 "global": GlobalSchema.rows
})
''',
            encoding="utf-8",
        )
        (schema_dir / "DesktopPage.js").write_text(
            '''.pragma library
var rows = [
 { tab: "General", group: "LOOK", key: "deckFrostEnabled", label: "Deck frost", desc: "Frost behind windows", ctl: "sw" },
 { tab: "Effects", group: "DETAIL", key: "grain", label: "Grain", desc: "Paper texture", adv: true },
 { tab: "Effects", group: "DETAIL", key: "internal", desc: "Not a setting" }
];
''',
            encoding="utf-8",
        )
        (schema_dir / "GlobalPage.js").write_text(
            '''.pragma library
function label(value) { return value; }
var rows = [
 { tab: "General", group: "REGION", key: "language", label: label("Language"), desc: "Display language" }
];
''',
            encoding="utf-8",
        )
        wm_dir = self.repo / "ryoku/wm"
        wm_dir.mkdir(parents=True)
        (wm_dir / "caps.go").write_text(
            'package wm\nconst ( CapMonitorConfig Capability = "monitorConfig" )\n',
            encoding="utf-8",
        )
        (wm_dir / "binds.go").write_text(
            '''package wm
var catalogue = []CatalogBind{
 {ID: "shell.launcher", Category: "Shell", Label: "App launcher", Hint: "Open apps", Chord: "SUPER + Space"},
}
''',
            encoding="utf-8",
        )
        guide = self.repo / "ryoku/rashin/skills/ryoku/gui.md"
        guide.parent.mkdir(parents=True)
        guide.write_text(
            '''# GUI
## The Hub pages
| Rail name | `hub open` key | Page | What it owns |
|---|---|---|---|
| Desktop | `desktop` | DesktopPage.qml | desktop appearance |
| General | `global` | GlobalPage.qml | shared preferences |

## Surfaces outside the Hub
| Intent | GUI surface | How to open | The command behind it |
|---|---|---|---|
| Change wallpaper | Wallpaper picker | Super+W | `ryogami wallpaper ui` |
''',
            encoding="utf-8",
        )

    def test_pages_and_settings_follow_hub_catalogue(self):
        entries = collect_entries(self.repo)
        by_id = {entry["id"]: entry for entry in entries}

        self.assertEqual(
            by_id["page:desktop"]["open"],
            "Super+comma, then Desktop (or `ryoku-shell hub open desktop`)",
        )
        self.assertEqual(
            by_id["setting:desktop/deckFrostEnabled"]["path"],
            "Ryoku Hub > Desktop > General > Deck frost",
        )
        self.assertEqual(
            by_id["setting:global/language"]["path"],
            "Ryoku Hub > General > Language",
        )
        self.assertTrue(by_id["setting:desktop/grain"]["advanced"])
        self.assertFalse(any(entry["title"] == "" for entry in entries if entry["kind"] == "setting"))
        self.assertNotIn("setting:desktop/internal", by_id)


class ChannelMergeTests(unittest.TestCase):
    def test_merges_channel_variants_without_hiding_moved_rows(self):
        stable = [
            {"id": "page:both", "title": "Both", "path": "Ryoku Hub > Both", "value": "stable"},
            {"id": "page:stable", "title": "Stable", "path": "Ryoku Hub > Stable"},
            {"id": "setting:moved", "title": "Moved", "path": "Ryoku Hub > Old"},
        ]
        unstable = [
            {"id": "page:both", "title": "Both", "path": "Ryoku Hub > Both", "value": "unstable"},
            {"id": "page:unstable", "title": "Unstable", "path": "Ryoku Hub > Unstable"},
            {"id": "setting:moved", "title": "Moved", "path": "Ryoku Hub > New"},
        ]

        by_id = {
            entry["id"]: entry for entry in merge_channel_entries(stable, unstable)
        }

        self.assertEqual(by_id["page:both"]["channels"], ["stable", "unstable"])
        self.assertEqual(by_id["page:both"]["value"], "stable")
        self.assertEqual(by_id["page:stable"]["channels"], ["stable"])
        self.assertEqual(by_id["page:unstable"]["channels"], ["unstable"])
        self.assertEqual(by_id["setting:moved"]["path"], "Ryoku Hub > Old")
        self.assertEqual(by_id["setting:moved"]["channels"], ["stable"])
        self.assertEqual(
            by_id["setting:moved@unstable"]["channels"], ["unstable"]
        )
        self.assertEqual(
            by_id["setting:moved@unstable"]["path"], "Ryoku Hub > New"
        )


class DocsChunkTests(unittest.TestCase):
    def test_heading_urls_length_cap_and_jsx_stripping(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        docs = root / "docs-repo"
        repo = root / "ryoku-repo"
        (docs / "docs").mkdir(parents=True)
        repo.mkdir()
        long_text = " ".join(["word"] * 230)
        (docs / "docs/sample.mdx").write_text(
            f'''---
title: Sample Page
---

## Hello, World!

<Card title="Not part of the text">
Visible guidance stays.
</Card>

{long_text}
''',
            encoding="utf-8",
        )

        chunks = collect_chunks(docs, repo)

        self.assertGreaterEqual(len(chunks), 2)
        self.assertTrue(all(len(str(chunk["text"])) <= 900 for chunk in chunks))
        self.assertTrue(
            all(
                chunk["url"] == "https://docs.ryoku.dev/docs/sample#hello-world"
                for chunk in chunks
            )
        )
        self.assertTrue(all("<Card" not in str(chunk["text"]) for chunk in chunks))
        self.assertTrue(any("Visible guidance stays." in str(chunk["text"]) for chunk in chunks))


if __name__ == "__main__":
    unittest.main()
