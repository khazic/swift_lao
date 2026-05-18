# One-shot launcher for the 8-node gemma-4-31B GRPO run.
#
# Run this on the rank-0 host (10.178.128.118). It will, in order:
#   1. ssh in parallel into the 7 workers.
#   2. On each worker: cd to the swift checkout, git fetch + reset --hard to
#      origin/gemma4-rl, export the required env vars, then nohup-launch
#      grpo_multi_node.sh writing to a per-node log file.
#   3. Update rank-0 locally the same way, then run grpo_multi_node.sh in the
#      foreground (so this terminal streams rank-0's output).
#
# Prerequisites on rank-0 BEFORE running:
#   - Passwordless ssh from rank-0 to every worker IP (ssh-copy-id).
#   - Same swift_lao checkout path on every node (shared FS is easiest).
#   - Each node already has `swift` importable (pip install -e ".[all]" done).
#   - These env vars are exported in the current shell:
#         TRANSLATION_JUDGE_API_BASE  e.g. https://api.360.cn/v1
#         TRANSLATION_JUDGE_API_KEY   judge bearer token
#         TRANSLATION_JUDGE_MODEL     e.g. deepseek/deepseek-v4-pro
#         DATASET                     /path/to/translation_train.jsonl
#     Optional: MODEL, OUTPUT_DIR, NUM_GENERATIONS, SSH_USER, BRANCH.
#
# Logs go to ${LOG_DIR:-${SWIFT_DIR}/logs/grpo_<timestamp>}/node_<rank>.log.
#
# To stop everything: bash examples/models/gemma4/stop_all_nodes.sh

set -euo pipefail

# ---- topology ---------------------------------------------------------------
MASTER_ADDR=${MASTER_ADDR:-10.178.128.118}
MASTER_PORT=${MASTER_PORT:-29500}
WORKERS=(
    10.178.139.42
    10.178.160.208
    10.178.171.109
    10.178.159.187
    10.178.156.133
    10.178.162.19
    10.178.143.28
)
NNODES=$(( ${#WORKERS[@]} + 1 ))
NPROC_PER_NODE=${NPROC_PER_NODE:-8}

# ---- paths ------------------------------------------------------------------
SWIFT_DIR=${SWIFT_DIR:-/llm-align/liuchonghan/swift_lao}
BRANCH=${BRANCH:-gemma4-rl}
LOG_DIR=${LOG_DIR:-${SWIFT_DIR}/logs/grpo_$(date +%Y%m%d_%H%M%S)}
SSH_USER=${SSH_USER:-root}

# ---- required env on rank-0 -------------------------------------------------
: "${TRANSLATION_JUDGE_API_BASE:?export TRANSLATION_JUDGE_API_BASE first}"
: "${TRANSLATION_JUDGE_API_KEY:?export TRANSLATION_JUDGE_API_KEY first}"
: "${TRANSLATION_JUDGE_MODEL:?export TRANSLATION_JUDGE_MODEL first}"
: "${DATASET:?export DATASET first}"

mkdir -p "${LOG_DIR}"
echo "[launch] log dir: ${LOG_DIR}"
echo "[launch] master=${MASTER_ADDR}:${MASTER_PORT}  nnodes=${NNODES}  proc/node=${NPROC_PER_NODE}"

# Build the remote command for a given rank. Heredoc keeps quoting simple.
build_remote_cmd() {
    local rank=$1
    local logfile="${LOG_DIR}/node_${rank}.log"
    cat <<EOF
set -e
cd "${SWIFT_DIR}"
git fetch origin
git reset --hard "origin/${BRANCH}"
export MASTER_ADDR="${MASTER_ADDR}"
export MASTER_PORT="${MASTER_PORT}"
export NNODES="${NNODES}"
export NPROC_PER_NODE="${NPROC_PER_NODE}"
export NODE_RANK="${rank}"
export TRANSLATION_JUDGE_API_BASE="${TRANSLATION_JUDGE_API_BASE}"
export TRANSLATION_JUDGE_API_KEY="${TRANSLATION_JUDGE_API_KEY}"
export TRANSLATION_JUDGE_MODEL="${TRANSLATION_JUDGE_MODEL}"
export DATASET="${DATASET}"
${MODEL:+export MODEL="${MODEL}"}
${OUTPUT_DIR:+export OUTPUT_DIR="${OUTPUT_DIR}"}
${NUM_GENERATIONS:+export NUM_GENERATIONS="${NUM_GENERATIONS}"}
${HTTP_PROXY:+export HTTP_PROXY="${HTTP_PROXY}"}
${HTTPS_PROXY:+export HTTPS_PROXY="${HTTPS_PROXY}"}
${NO_PROXY:+export NO_PROXY="${NO_PROXY}"}
${http_proxy:+export http_proxy="${http_proxy}"}
${https_proxy:+export https_proxy="${https_proxy}"}
${no_proxy:+export no_proxy="${no_proxy}"}
nohup bash examples/models/gemma4/grpo_multi_node.sh > "${logfile}" 2>&1 < /dev/null &
disown || true
echo "[node ${rank}] started pid \$! -> ${logfile}"
EOF
}

# ---- 1) workers in parallel -------------------------------------------------
worker_pids=()
for i in "${!WORKERS[@]}"; do
    rank=$(( i + 1 ))
    ip=${WORKERS[$i]}
    echo "[launch] ssh -> rank=${rank} ${SSH_USER}@${ip}"
    ssh -o StrictHostKeyChecking=no -o BatchMode=yes -o ServerAliveInterval=30 \
        "${SSH_USER}@${ip}" "bash -s" <<EOF &
$(build_remote_cmd ${rank})
EOF
    worker_pids+=($!)
done

for pid in "${worker_pids[@]}"; do
    wait "${pid}" || echo "[launch] WARNING: ssh pid ${pid} exited non-zero"
done
echo "[launch] all worker ssh sessions returned; training is now running under nohup on each worker"

# ---- 2) rank-0 in foreground ------------------------------------------------
cd "${SWIFT_DIR}"
git fetch origin
git reset --hard "origin/${BRANCH}"
export MASTER_ADDR MASTER_PORT NNODES NPROC_PER_NODE
export NODE_RANK=0
export TRANSLATION_JUDGE_API_BASE TRANSLATION_JUDGE_API_KEY TRANSLATION_JUDGE_MODEL
export DATASET
echo "[launch] starting rank 0 in foreground; rank>=1 logs at ${LOG_DIR}/node_<rank>.log"
bash examples/models/gemma4/grpo_multi_node.sh 2>&1 | tee "${LOG_DIR}/node_0.log"
