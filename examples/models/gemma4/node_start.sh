#!/bin/bash
# Run on ANY node — auto-detects NODE_RANK from hostname.
# Hostname convention:
#   *-m-0  -> rank 0 (master, runs in foreground)
#   *-w-N  -> rank N+1 (worker, runs in background under nohup)
#
# NOTE: After parking the original main-m-0 (hardware fault), a worker node
# is now acting as rank 0. The hostname-based auto-detect below will be wrong
# on the new master — manually `export NODE_RANK=0` before running this on
# the new master, or use launch_all_nodes.sh which sets NODE_RANK explicitly.
#
# Required env (already in /llm-align/liuchonghan/judge_env.sh on shared FS):
#   TRANSLATION_JUDGE_API_BASE / API_KEY / MODEL
#
# Optional overrides:
#   MASTER_ADDR  MASTER_PORT  DATASET  MODEL  OUTPUT_DIR  NUM_GENERATIONS

set -euo pipefail

SWIFT_DIR=${SWIFT_DIR:-/llm-align/liuchonghan/swift_lao}
JUDGE_ENV=${JUDGE_ENV:-/llm-align/liuchonghan/judge_env.sh}

source "${JUDGE_ENV}"

export DATASET=${DATASET:-/llm-align/liuchonghan/swift_lao/examples/models/gemma4/translation_train.jsonl}
export MASTER_ADDR=${MASTER_ADDR:-10.178.140.140}
export MASTER_PORT=${MASTER_PORT:-8888}

# Auto-detect rank from hostname
_HOST=$(hostname -s)
if [[ "$_HOST" == *"-m-"* ]]; then
    export NODE_RANK=0
else
    _W=${_HOST##*-w-}
    export NODE_RANK=$(( _W + 1 ))
fi

mkdir -p "${SWIFT_DIR}/logs"
LOGFILE="${SWIFT_DIR}/logs/node_${NODE_RANK}.log"

echo "[node_start] host=$_HOST  NODE_RANK=$NODE_RANK  MASTER_ADDR=$MASTER_ADDR:$MASTER_PORT"

cd "${SWIFT_DIR}"

if [ "${NODE_RANK}" -eq 0 ]; then
    echo "[node_start] rank 0 — running in foreground, log: ${LOGFILE}"
    bash examples/models/gemma4/grpo_multi_node.sh 2>&1 | tee "${LOGFILE}"
else
    echo "[node_start] rank ${NODE_RANK} — running in background, log: ${LOGFILE}"
    nohup bash examples/models/gemma4/grpo_multi_node.sh > "${LOGFILE}" 2>&1 &
    disown
    echo "[node_start] rank ${NODE_RANK} started pid=$!"
fi
