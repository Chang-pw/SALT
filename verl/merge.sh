cd /mnt/storage01/zbw/rl-gradient/verl

python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir checkpoints/grpo_math-train/deepseek-qwen1.5b_gradient/global_step_200/actor \
    --target_dir /mnt/storage01/zbw/pretrain_model/rl_gradient_model/deepseek-qwen1.5b_grpo_step200