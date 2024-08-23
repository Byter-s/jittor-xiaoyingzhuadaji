#!/bin/bash
MODEL_NAME="./stable-diffusion-2-1"
BASE_INSTANCE_DIR="./data/B"
SD_BASE_DIR="./data/sdt500"
OUTPUT_DIR_PREFIX="style/style_"
RESOLUTION=512
TRAIN_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=1
CHECKPOINTING_STEPS=500
LEARNING_RATE=1e-4
LR_SCHEDULER="constant"
LR_WARMUP_STEPS=0
SEED=0 # 0表示随机种子

MAX_TRAIN_STEPS=5000
EPOCHS=1
GPU_COUNT=8
MIN_NUM=0
MAX_NUM=27

for ((folder_number = $MIN_NUM; folder_number <= $MAX_NUM; folder_number+=$GPU_COUNT)); do
    for ((gpu_id = 0; gpu_id < GPU_COUNT; gpu_id++)); do
        current_folder_number=$((folder_number + gpu_id))
        echo "current_folder_number: $current_folder_number"
        if [ $current_folder_number -gt $MAX_NUM ]; then
            break
        fi
        INSTANCE_DIR="${BASE_INSTANCE_DIR}/$(printf "%02d" $current_folder_number)/images"
        SD_INSTANCE_DIR="${SD_BASE_DIR}/$(printf "%02d" $current_folder_number)"

        OUTPUT_DIR="${OUTPUT_DIR_PREFIX}$(printf "%02d" $current_folder_number)"
        CUDA_VISIBLE_DEVICES=$gpu_id
        PROMPT=$(printf "%02d" $current_folder_number)

        COMMAND="CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python train.py \
            --pretrained_model_name_or_path=$MODEL_NAME \
            --instance_data_dir=$INSTANCE_DIR \
            --sd_instance_data_dir=$SD_INSTANCE_DIR \
            --output_dir=$OUTPUT_DIR \
            --checkpointing_steps=$CHECKPOINTING_STEPS \
            --instance_prompt=$PROMPT \
            --resolution=$RESOLUTION \
            --train_batch_size=$TRAIN_BATCH_SIZE \
            --gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS \
            --learning_rate=$LEARNING_RATE \
            --lr_scheduler=$LR_SCHEDULER \
            --lr_warmup_steps=$LR_WARMUP_STEPS \
            --max_train_steps=$MAX_TRAIN_STEPS \
            --seed=$SEED \
            --num_train_epochs=$EPOCHS"

        eval $COMMAND &
        sleep 10
    done
    wait
done