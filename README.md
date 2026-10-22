<div align="center">

<img src="https://raw.githubusercontent.com/ryoku-dev/ryoku-arch/main/ryoku/assets/brand/logo-mark.png" alt="Ryoku" width="160" />

# Nero

**The Ryoku support bot for Discord** &middot; *reviewed answers, safety prompts, and source-backed replies.*

Nero is the private Discord bot for Ryoku support. It retrieves reviewed
support cards, Ryoku's GUI map, documentation, and stable source evidence,
then produces short GUI-first answers with deterministic fallback.

[![License: GPL-3.0](https://img.shields.io/badge/license-GPL--3.0-E2342A?style=for-the-badge)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-E2342A?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org)
[![discord.py](https://img.shields.io/badge/discord.py-2.6%2B-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discordpy.readthedocs.io)
[![Ryoku Docs](https://img.shields.io/badge/docs-ryoku.dev-E2342A?style=for-the-badge)](https://docs.ryoku.dev)

<kbd>[Docs](https://docs.ryoku.dev)</kbd> &middot; <kbd>[Support catalog](data/support.json)</kbd> &middot; <kbd>[Benchmarks](data/benchmark.json)</kbd> &middot; <kbd>[Discord](https://discord.gg/8KjBmUEyKA)</kbd>

</div>

---

## What it does

- Answers reviewed Ryoku support questions in Discord

## Quick start

### 1. Create a virtual environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example env
```

### 2. Clone the stable Ryoku repo

The bot uses a local stable Ryoku checkout for source-backed answers.

```bash
git clone https://github.com/ryoku-dev/ryoku-arch.git /path/to/ryoku
```

Point `RYOKU_REPO_PATH` in `env` at that checkout.

### 3. Initialize the Ryoku checkout for Prowl

Run this once inside the cloned Ryoku repo:

```bash
cd /path/to/ryoku
prowl-agent init --no-input --integrations none
```

### 4. Configure `env`

Set:

- `TOKEN` to Nero's Discord bot token
- `RYOKU_REPO_PATH` to your cloned Ryoku checkout
- `PROWL_AGENT_PATH` to the absolute Prowl executable, e.g. `/home/neur0map/workspace/prowl-agent/prowl-agent`. This prevents service `PATH` drift.
- `SUPPORT_CHANNEL_ID` to the channel where ordinary requests are answered; replies or mentions aimed at other members are ignored unless Nero is also addressed.
- `FEEDBACK_DB_PATH` to a local SQLite file for Correct/Incorrect signals. It defaults to `runtime/nero-feedback.sqlite3` and must be backed up separately from Git.
- `NERO_CLOUD`, `NERO_LLM_SOCKET`, `NERO_LLM_MODEL`, and `NERO_LLM_TIMEOUT_SECONDS` for the cloud answer bridge
- `NERO_USER_LIMIT` to the per-user answer limit as `count/seconds` (default `6/300`); set `0` to disable it.
- `GUI_MAP_PATH` and `DOCS_CHUNKS_PATH` for the generated knowledge files
- `HELP_FORUM_ID` to answer new posts in the support forum automatically
- optionally `MODEL_NAME`, `PROWL_TIMEOUT_SECONDS`, and `PROWL_RESULT_LIMIT`

### How Nero answers

Nero retrieves a compact evidence pack from reviewed support cards, the
generated Ryoku GUI map, documentation chunks, and Prowl source results when
needed. It sends that evidence through the local `nero-llm` bridge
(`llm_bridge/`, `deploy/nero-llm.service`) to a fast cloud model. The bridge
runs as the account that holds the Hermes OAuth login. Nero validates the
result and falls back to a deterministic card answer whenever the model fails.

Nero automatically answers the starter message in the configured help forum.
In the ordinary support channel it ignores greetings, thanks, and other short
messages that are not requests. Mentions and replies always get an answer.

Regenerate the knowledge files from current Ryoku checkouts:

```bash
python tools/build_gui_map.py --repo <stable ryoku> --unstable <unstable ryoku> --out data/gui_map.json
python tools/build_docs_chunks.py --docs <ryoku-docs> --repo <ryoku> --out data/docs_chunks.json
```

## Deploy

`deploy/` contains the local LLM bridge, bot, and daily refresh systemd units. The
refresh keeps shallow stable, unstable, and docs checkouts, initializes Prowl in
the stable checkout, validates regenerated knowledge, and restarts Nero only
when the knowledge content changes.

The live host sets these keys in `/etc/nero/nero.env`:

```dotenv
GUI_MAP_PATH=/var/lib/nero/knowledge/gui_map.json
DOCS_CHUNKS_PATH=/var/lib/nero/knowledge/docs_chunks.json
RYOKU_REPO_PATH=/var/lib/nero/sources/ryoku-stable
HELP_FORUM_ID=...
SUPPORT_CHANNEL_ID=...
NERO_LLM_MODEL=...
```

Run a refresh manually with:

```bash
sudo systemctl start nero-refresh.service
```


## Run

```bash
uv run --python 3.12 \
  --with-requirements requirements.txt \
  --env-file env \
  python -B bot.py
```

In `SUPPORT_CHANNEL_ID`, users can ask Nero normal Ryoku/Arch support questions without mentioning it. In every other channel, call Nero with `@Nero` or reply to one of Nero's messages. Mentions and replies work in all channels.

Every Nero reply has **Correct** and **Incorrect** buttons. Only the person who asked can submit or revise a rating. Nero stores the rating and the Discord message IDs needed to join it to the future cleaned message archive; it does not add question or answer text to this feedback journal.

## Test

Run the unit test suite:

```bash
uv run --python 3.12 \
  --with-requirements requirements.txt \
  python -B -m unittest discover -s tests -v
```

Inspect the answer-evaluation options:

```bash
python -m eval.run_eval --help
```

Run the regression benchmark:

```bash
uv run --python 3.12 \
  --with-requirements requirements.txt \
  python -B benchmark.py --mode hybrid \
  --benchmark data/regressions.json \
  --ryoku-repo /path/to/ryoku \
  --check
```

## Benchmark

Run the full benchmark with replay sampling:

```bash
uv run --python 3.12 \
  --with-requirements requirements.txt \
  python -B benchmark.py --mode hybrid \
  --ryoku-repo /path/to/ryoku \
  --replays 3 --check
```
