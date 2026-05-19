# Multi-node GRPO for gemma-4-31B-it with LLM-as-judge translation reward.
# Topology: 8 nodes x 8 A100 80G = 64 GPUs.
#
# Run instructions (on EVERY node, in order):
#
#   1. Make sure the same swift_lao checkout, same model path, and same dataset
#      path exist on each node (a shared NFS works best).
#
#   2. Set rank-specific and host-specific env vars on each node:
#        export NODE_RANK=<this node's rank, 0..7>     # 0 on main-m-0
#        export MASTER_ADDR=<hostname-or-IP of NODE_RANK=0, reachable from all>
#        # Typically: main-m-0
#
#   3. Provide the judge API credentials (same values on all 8 nodes):
#        export TRANSLATION_JUDGE_API_BASE=https://your-judge.example.com
#        export TRANSLATION_JUDGE_API_KEY=sk-xxx
#        export TRANSLATION_JUDGE_MODEL=your-judge-model
#
#   4. Provide the dataset path (same value on all 8 nodes):
#        export DATASET=/path/to/translation_train.jsonl
#
#   5. Launch on each node:
#        bash examples/models/gemma4/grpo_multi_node.sh
#
# A quick health-check before the real run: from any worker (NODE_RANK>=1),
#   nc -vz "$MASTER_ADDR" "${MASTER_PORT:-29500}"
# must succeed; otherwise the rendezvous will hang.

set -euo pipefail

# ---- distributed topology ---------------------------------------------------
export NNODES=${NNODES:-8}
export NPROC_PER_NODE=${NPROC_PER_NODE:-8}
export NODE_RANK=${NODE_RANK:?NODE_RANK must be set on each node (0..NNODES-1)}
export MASTER_ADDR=${MASTER_ADDR:?MASTER_ADDR must point to the rank-0 host}
export MASTER_PORT=${MASTER_PORT:-29500}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export VLLM_DISABLE_FLASHINFER_ALLREDUCE=1

# Optional: pin NCCL/Gloo to a specific NIC if the cluster has multiple.
# export NCCL_SOCKET_IFNAME=bond0
# export GLOO_SOCKET_IFNAME=bond0
# export NCCL_IB_DISABLE=0

# ---- judge API --------------------------------------------------------------
: "${TRANSLATION_JUDGE_API_BASE:?set to your judge endpoint base url}"
: "${TRANSLATION_JUDGE_API_KEY:?set to the judge api key}"
: "${TRANSLATION_JUDGE_MODEL:?set to the judge model name}"
export NUM_GENERATIONS=${NUM_GENERATIONS:-8}
export TRANSLATION_JUDGE_NUM_GENERATIONS=${NUM_GENERATIONS}
export TRANSLATION_JUDGE_TIMEOUT=${TRANSLATION_JUDGE_TIMEOUT:-180}
export TRANSLATION_JUDGE_TEMPERATURE=${TRANSLATION_JUDGE_TEMPERATURE:-0.0}
export TRANSLATION_JUDGE_MAX_CONCURRENCY=${TRANSLATION_JUDGE_MAX_CONCURRENCY:-32}

# ---- paths ------------------------------------------------------------------
MODEL=${MODEL:-/llm-align/open_models/gemma4/gemma-4-31B-it}
DATASET=${DATASET:?path to local translation JSONL (messages format)}
OUTPUT_DIR=${OUTPUT_DIR:-output/gemma4_31b_grpo_translation_judge}

swift rlhf \
    --rlhf_type grpo \
    --model "${MODEL}" \
    --external_plugins examples/train/grpo/plugin/translation_judge.py \
    --reward_funcs translation_judge \
    --tuner_type full \
    --freeze_vit true \
    --freeze_aligner true \
    --torch_dtype bfloat16 \
    --attn_impl eager \
    --gradient_checkpointing true \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.4 \
    --vllm_tensor_parallel_size 8 \
    --vllm_max_model_len 2048 \
    --vllm_engine_kwargs '{"max_num_batched_tokens": 2560}' \
    --vllm_limit_mm_per_prompt '{"image": 1, "video": 1}' \
    --dataset "${DATASET}" \
    --load_from_cache_file true \
    --split_dataset_ratio 0 \
    --max_length 2048 \
    --max_completion_length 512 \
    --num_generations ${NUM_GENERATIONS} \
    --temperature 1.0 \
    --top_p 0.9 \
    --beta 0.0 \
    --per_device_train_batch_size ${NUM_GENERATIONS} \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-6 \
    --warmup_ratio 0.03 \
    --num_train_epochs 1 \
    --logging_steps 1 \
    --save_strategy steps \
    --save_steps 100 \
    --save_total_limit 20 \
    --eval_strategy no \
    --output_dir "${OUTPUT_DIR}" \
    --dataloader_num_workers 2 \
    --dataset_num_proc 4 \
    --deepspeed zero3 \
    --offload_optimizer true \
    --offload_model true \
    --move_model_batches 40 \
    --sleep_level 1 \
    --log_completions true \
    --report_to none
