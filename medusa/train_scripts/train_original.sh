set -ex

export WANDB_MODE=offline
# export CUDA_VISIBLE_DEVICES=0,1,2,3

torchrun --nproc_per_node=8 medusa/train/train_apar.py \
    --model_name_or_path vicuna-7b-v1.3 \
    --data_path ../data/apar_flatten.json \
    --mix_data_path ../data/unstructured.json \
    --ratios "0.5 0.5" \
    --data_len 256000 \
    --bf16 True \
    --output_dir "7b_original_medusa" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --learning_rate 1e-3 \
    --weight_decay 0.0 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --lazy_preprocess True \
    --medusa_num_heads 2 \
    --medusa_num_layers 2