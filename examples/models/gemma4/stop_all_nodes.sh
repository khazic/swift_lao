# Kill the 8-node GRPO run on every host.
#
# Run on the rank-0 host. Hits the worker IPs over ssh and pkills any
# `swift rlhf` or `grpo_multi_node` process, plus the local one on rank-0.

set -uo pipefail

WORKERS=(
    10.178.139.42
    10.178.160.208
    10.178.171.109
    10.178.159.187
    10.178.156.133
    10.178.162.19
    10.178.143.28
)
SSH_USER=${SSH_USER:-root}

kill_cmd="pkill -f 'examples/models/gemma4/grpo_multi_node.sh' || true; pkill -f 'swift rlhf' || true; sleep 2; pkill -9 -f 'swift rlhf' || true"

# local (rank-0)
echo "[stop] local"
bash -c "${kill_cmd}"

# workers in parallel
pids=()
for ip in "${WORKERS[@]}"; do
    echo "[stop] ssh ${SSH_USER}@${ip}"
    ssh -o StrictHostKeyChecking=no -o BatchMode=yes \
        "${SSH_USER}@${ip}" "${kill_cmd}" &
    pids+=($!)
done

for pid in "${pids[@]}"; do
    wait "${pid}" || true
done
echo "[stop] done"
