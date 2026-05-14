# GRPO smoke test for gemma-4-31B-it (text-only, no images).
# Goal: verify the GRPO pipeline runs end-to-end on this model before real training.
# Reference GPU budget: 8 * 80GiB. Uses LoRA + DeepSpeed ZeRO-3 + vLLM colocate.

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
swift rlhf \
    --rlhf_type grpo \
    --model /llm-align/open_models/gemma4/gemma-4-31B-it \
    --tuner_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --freeze_vit true \
    --freeze_aligner true \
    --torch_dtype bfloat16 \
    --attn_impl eager \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.5 \
    --vllm_tensor_parallel_size 4 \
    --vllm_max_model_len 4096 \
    --dataset AI-MO/NuminaMath-TIR#200 \
    --load_from_cache_file true \
    --split_dataset_ratio 0 \
    --system examples/train/grpo/prompt.txt \
    --reward_funcs accuracy \
    --max_length 4096 \
    --max_completion_length 1024 \
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
    --sleep_level 1 \
    --log_completions true \
    --report_to none
