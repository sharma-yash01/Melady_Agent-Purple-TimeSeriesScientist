#!/usr/bin/env bash
set -euo pipefail

HOST="0.0.0.0"
PORT="9023"
CARD_URL=""
MODEL="${PURPLE_AGENT_MODEL:-openai/gpt-4o}"
TEMPERATURE="${PURPLE_AGENT_TEMPERATURE:-0.0}"
MAX_TOKENS="${PURPLE_AGENT_MAX_TOKENS:-4000}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) HOST="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --card-url) CARD_URL="$2"; shift 2 ;;
    *) break ;;
  esac
done

export AGENT_HOST="$HOST"
export AGENT_PORT="$PORT"
export OPENAI_API_KEY="${OPENAI_API_KEY:-}"
export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-}"
export OPENROUTER_API_KEY="${OPENROUTER_API_KEY:-}"

CARD_URL_EFFECTIVE="${CARD_URL:-http://${HOST}:${PORT}}"
export CARD_URL="$CARD_URL_EFFECTIVE"

echo "[entrypoint] Starting TSci Purple Agent (A2A) on ${HOST}:${PORT}"
echo "[entrypoint] CARD_URL=${CARD_URL_EFFECTIVE}"
echo "[entrypoint] Model=${MODEL}, Temperature=${TEMPERATURE}, MaxTokens=${MAX_TOKENS}"

exec python purple_agent.py \
  --host "$HOST" \
  --port "$PORT" \
  --card-url "$CARD_URL_EFFECTIVE" \
  --model "$MODEL" \
  --temperature "$TEMPERATURE" \
  --max-tokens "$MAX_TOKENS"
