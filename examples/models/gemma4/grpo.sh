# Full-parameter GRPO smoke test for gemma-4-31B-it (text-only, no images).
# Goal: verify the full-param GRPO pipeline runs end-to-end before real training.
# Reference GPU budget: 8 * 80GiB. Very tight: requires ZeRO-3 + gradient checkpointing
# + aggressive offload + sleep_level 1 to keep vLLM colocate within budget.
# If OOM, switch to vLLM server mode like examples/train/grpo/external/grpo_32b_full.sh.

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
swift rlhf \
    --rlhf_type grpo \
    --model /llm-align/open_models/gemma4/gemma-4-31B-it \
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
    --dataset AI-MO/NuminaMath-TIR#200 \
    --load_from_cache_file true \
    --split_dataset_ratio 0 \
    --system examples/train/grpo/prompt.txt \
    --reward_funcs accuracy \
    --max_length 2048 \
    --max_completion_length 512 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-6 \
    --max_steps 5 \
    --warmup_ratio 0.0 \
    --logging_steps 1 \
    --save_strategy no \
    --eval_strategy no \
    --output_dir output \
    --dataloader_num_workers 2 \
    --dataset_num_proc 4 \
    --num_generations 4 \
    --temperature 1.0 \
    --top_p 0.9 \
    --beta 0.0 \
    --deepspeed zero3 \
    --offload_optimizer true \
    --offload_model true \
    --move_model_batches 40 \
    --sleep_level 1 \
    --log_completions true \
    --report_to none
