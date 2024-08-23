#!/bin/bash
STEP=500
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
        STYLE=$(printf "%02d" $current_folder_number)
        CUDA_VISIBLE_DEVICES=$gpu_id
        COMMAND="CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python run_sd.py \
            --style=$STYLE \
            --step=$STEP"
        eval $COMMAND &
        sleep 8
    done
    wait
done