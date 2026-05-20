# Kill the 3-node GRPO run on every host.
#
# Run on the rank-0 host (10.178.140.140). Hits the worker IPs over ssh and
# pkills any `swift rlhf` or `grpo_multi_node` process, plus the local one
# on rank-0.

set -uo pipefail

WORKERS=(
    10.178.160.214
    10.178.171.122
    # 10.178.128.125  # disabled: GPU2 ECC uncorrectable + GPU6 Xid 31
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
