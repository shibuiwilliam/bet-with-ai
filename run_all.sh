#!/usr/bin/env bash
# Run every study in order. Outputs (Markdown, JSON, figures) go to ./outputs/.
# Usage: ./run_all.sh            # all studies
#        ./run_all.sh s3 s5      # selected studies
# Study 7 re-scores the cached answers in data/llm_cache/. It calls the Anthropic
# API only for prompts missing from the cache, with ANTHROPIC_API_KEY taken from
# the environment or from a .env file at the repository root.
set -euo pipefail
cd "$(dirname "$0")"
studies=(s1_chance s2_skill s3_sports s4_house_limits s5_parimutuel s6_trading s7_llm s8_synthesis)
for s in "${studies[@]}"; do
  if [[ $# -gt 0 && " $* " != *" ${s%%_*} "* ]]; then continue; fi
  echo "== $s"
  start=$(date +%s)
  uv run python -m "betai.$s"
  echo "   done in $(( $(date +%s) - start ))s"
done
