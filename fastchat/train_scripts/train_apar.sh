export WANDB_MODE=offline

torchrun --nproc_per_node=8 --master_port=20001 fastchat/train/train.py \
    --model_name_or_path vicuna-7b-v1.3 \
    --data_path ../data/apar.json \
    --mix_data_path ../data/unstructured.json \
    --ratios "0.5 0.5" \
    --data_len 256000 \
    --bf16 True \
    --output_dir "apar-7b" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 999999999 \
    --save_total_limit 9999999 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "shard_grad_op auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True