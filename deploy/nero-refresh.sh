#!/usr/bin/env bash
set -euo pipefail

STATE_DIR=/var/lib/nero
SOURCES_DIR="$STATE_DIR/sources"
KNOWLEDGE_DIR="$STATE_DIR/knowledge"
STABLE_DIR="$SOURCES_DIR/ryoku-stable"
UNSTABLE_DIR="$SOURCES_DIR/ryoku-unstable"
DOCS_DIR="$SOURCES_DIR/ryoku-docs"
CHANGED_FILE="$KNOWLEDGE_DIR/.changed"
PROWL_AGENT_PATH=${PROWL_AGENT_PATH:-/opt/nero/bin/prowl-agent}
PYTHON=/opt/nero/venv/bin/python
APP_DIR=/opt/nero/app
RYOKU_URL=https://github.com/Ryoku-dev/ryoku.git
DOCS_URL=https://github.com/Ryoku-dev/ryoku-docs.git

fail() {
    printf 'nero refresh failed: %s\n' "$*" >&2
    exit 1
}

refresh_repo() {
    local label=$1
    local url=$2
    local branch=$3
    local directory=$4

    if [[ ! -d "$directory/.git" ]]; then
        if [[ -e "$directory" ]]; then
            fail "$label checkout exists but is not a Git repository"
        fi
        if ! git clone --depth 1 --branch "$branch" "$url" "$directory"; then
            rm -rf -- "$directory"
            fail "could not clone $label"
        fi
        return
    fi

    if ! git -C "$directory" fetch --depth 1 origin "$branch"; then
        fail "could not fetch $label"
    fi
    if ! git -C "$directory" reset --hard FETCH_HEAD; then
        fail "could not reset $label to fetched $branch"
    fi
    if ! git -C "$directory" clean -fdx -e .prowl; then
        fail "could not clean $label checkout"
    fi
}

json_count() {
    local file=$1
    local key=$2
    local minimum=$3
    "$PYTHON" -c 'import json, sys; data = json.load(open(sys.argv[1], encoding="utf-8")); values = data.get(sys.argv[2]); print(len(values)) if isinstance(values, list) and len(values) >= int(sys.argv[3]) else sys.exit(1)' "$file" "$key" "$minimum" 2>/dev/null
}

mkdir -p "$SOURCES_DIR" "$KNOWLEDGE_DIR"
rm -f "$CHANGED_FILE"

refresh_repo "stable Ryoku" "$RYOKU_URL" main "$STABLE_DIR"
refresh_repo "unstable Ryoku" "$RYOKU_URL" unstable-dev "$UNSTABLE_DIR"
refresh_repo "Ryoku docs" "$DOCS_URL" main "$DOCS_DIR"

if [[ ! -d "$STABLE_DIR/.prowl" ]]; then
    if ! (cd "$STABLE_DIR" && "$PROWL_AGENT_PATH" init --no-input --integrations none); then
        printf 'nero refresh warning: Prowl initialization failed; continuing without a fresh index\n' >&2
    fi
fi

tmp_dir=$(mktemp -d "$STATE_DIR/.nero-refresh.XXXXXX") || fail "could not create temporary directory"
trap 'rm -rf -- "$tmp_dir"' EXIT

gui_tmp="$tmp_dir/gui_map.json"
docs_tmp="$tmp_dir/docs_chunks.json"

if ! "$PYTHON" "$APP_DIR/tools/build_gui_map.py" --repo "$STABLE_DIR" --unstable "$UNSTABLE_DIR" --out "$gui_tmp"; then
    fail "could not build GUI map"
fi
if ! "$PYTHON" "$APP_DIR/tools/build_docs_chunks.py" --docs "$DOCS_DIR" --repo "$STABLE_DIR" --out "$docs_tmp"; then
    fail "could not build docs chunks"
fi

if ! gui_count=$(json_count "$gui_tmp" entries 100); then
    fail "GUI map validation requires valid JSON with at least 100 entries"
fi
if ! docs_count=$(json_count "$docs_tmp" chunks 50); then
    fail "docs validation requires valid JSON with at least 50 chunks"
fi

changed=0
if [[ ! -f "$KNOWLEDGE_DIR/gui_map.json" ]] || ! cmp -s "$gui_tmp" "$KNOWLEDGE_DIR/gui_map.json"; then
    changed=1
fi
if [[ ! -f "$KNOWLEDGE_DIR/docs_chunks.json" ]] || ! cmp -s "$docs_tmp" "$KNOWLEDGE_DIR/docs_chunks.json"; then
    changed=1
fi

mv -f "$gui_tmp" "$KNOWLEDGE_DIR/gui_map.json" || fail "could not install GUI map"
mv -f "$docs_tmp" "$KNOWLEDGE_DIR/docs_chunks.json" || fail "could not install docs chunks"
if (( changed )); then
    touch "$CHANGED_FILE"
fi

stable_commit=$(git -C "$STABLE_DIR" rev-parse --short=12 HEAD)
unstable_commit=$(git -C "$UNSTABLE_DIR" rev-parse --short=12 HEAD)
docs_commit=$(git -C "$DOCS_DIR" rev-parse --short=12 HEAD)
printf 'nero refresh: stable=%s unstable=%s docs=%s gui_entries=%s docs_chunks=%s\n' \
    "$stable_commit" "$unstable_commit" "$docs_commit" "$gui_count" "$docs_count"
