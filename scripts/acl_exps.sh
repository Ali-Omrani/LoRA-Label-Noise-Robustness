#!/bin/bash

cd ../

balances=( 0.1 0.5 0.4 0.3 0.2 0.05 0.01 )
limited_data=10000
weight_decays=(0.01)
gpus=(0 1 2 3 4 5)
noise=(0 0.1 0.2 0.3 0.4 0.5)

methods=(lora)
lrs=(1e-5)
warmup_ratio=0.06
LM=roberta-large

datasets=(imdb imdb imdb imdb imdb imdb)
label_col="sentiment"




for dataset_index in "${!datasets[@]}"; do       
    gpu=${gpus[$dataset_index]}
    dataset=${datasets[$dataset_index]}
    noise_ratio=${noise[$dataset_index]}
    SESSION_NAME="$dataset-noise-$noise_ratio"

    concatenated_cmd=""

    for balance_ratio in "${balances[@]}"; do
        for method_index in "${!methods[@]}"; do       
            method=${methods[$method_index]}
            learning_rate=${lrs[$method_index]}
            for weight_decay in "${weight_decays[@]}"; do 
                        experiment_subdir=acl-exp5-fixed-$LM-method-$method-limit-$limited_data-l2-$weight_decay-balance-$balance_ratio
                        cmd="WANDB_PROJECT=lowres CUDA_VISIBLE_DEVICES=$gpu python train.py \
                                --project_name noise_exp0 \
                                --experiment_subdir $experiment_subdir \
                                --dataset_name $dataset \
                                --label_col $label_col \
                                --noise_ratio $noise_ratio \
                                --limited_data $limited_data \
                                --weight_decay $weight_decay \
                                --TRAIN_BATCH_SIZE 32 \
                                --VALID_BATCH_SIZE 64 \
                                --LEARNING_RATE $learning_rate \
                                --warmup_ratio $warmup_ratio \
                                --method $method \
                                --EPOCHS 20 \
                                --LM $LM \
                                --seed 0 \
                                --balance \
                                --balance_ratio $balance_ratio ; "
                        concatenated_cmd+=$cmd
                    done
        done
    done
    
    echo "Running model with gpu: $gpu, dataset: $dataset, session_name: $SESSION_NAME"
    echo $concatenated_cmd
    screen -dmS "$SESSION_NAME" bash -c "$concatenated_cmd"   
                            
done