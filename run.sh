#!/bin/bash
export HF_ENDPOINT="https://hf-mirror.com"
MODE="sd"
TEMPLATE="template.json"
GEN_STEPS=500
GPU_COUNT=4
MIN_NUM=24
MAX_NUM=27

for ((folder_number = $MIN_NUM; folder_number <= $MAX_NUM; folder_number+=$GPU_COUNT)); do
    for ((gpu_id = 0; gpu_id < GPU_COUNT; gpu_id++)); do
        current_folder_number=$((folder_number + gpu_id))
        echo "current_folder_number: $current_folder_number"
        if [ $current_folder_number -gt $MAX_NUM ]; then
            break
        fi
        CUDA_VISIBLE_DEVICES=$gpu_id
        COMMAND="CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python run_v2.py \
            --mode=$MODE \
            --template_file=$TEMPLATE \
            --step=$GEN_STEPS \
            --taskid=$current_folder_number"
        eval $COMMAND &
        sleep 10
    done
    wait
done