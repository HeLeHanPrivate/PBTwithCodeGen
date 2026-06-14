#!/bin/bash
# Load key.json into environment variables for shell scripts.
# Usage: source script/load_keys.sh

set -euo pipefail

KEY_FILE="${KEY_FILE:-key.json}"

if [[ ! -f "$KEY_FILE" ]]; then
    echo "[warning] $KEY_FILE not found. Create it from key.example.json." >&2
    return 0
fi

# Parse key.json and export known fields as environment variables.
export INF_API_KEY="${INF_API_KEY:-$(python3 -c "import json,sys; print(json.load(open('$KEY_FILE')).get('api_key',''))" 2>/dev/null || true)}"
export API_BASE_URL="${API_BASE_URL:-$(python3 -c "import json,sys; print(json.load(open('$KEY_FILE')).get('api_base_url',''))" 2>/dev/null || true)}"
export API_KEY_ENV="${API_KEY_ENV:-$(python3 -c "import json,sys; print(json.load(open('$KEY_FILE')).get('api_key_env','INF_API_KEY'))" 2>/dev/null || true)}"
export MODEL_PATH="${MODEL_PATH:-$(python3 -c "import json,sys; print(json.load(open('$KEY_FILE')).get('model_path',''))" 2>/dev/null || true)}"
export DATASET_ROOT="${DATASET_ROOT:-$(python3 -c "import json,sys; print(json.load(open('$KEY_FILE')).get('dataset_root',''))" 2>/dev/null || true)}"
