#!/bin/bash
pip install -r requirements.txt

rm -rf ../sft_Amazon_Movie_ranking
CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage sft \
    --do_train \
    --model_name_or_path ../llama-2-7b-chat-hf \
    --dataset rec_Amazon_Movie_sft_ranking \
    --template llama2 \
    --finetuning_type lora \
    --lora_target all \
    --output_dir sft_Amazon_Movie_ranking \
    --overwrite_cache \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --save_steps 10 \
    --learning_rate 1e-4 \
    --num_train_epochs 5.0 \
    --plot_loss \
    --fp16 

rm -rf ../dpo_Amazon_Movie_ranking
CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage dpo \
    --do_train \
    --model_name_or_path ../llama-2-7b-chat-hf \
    --adapter_name_or_path sft_Amazon_Movie_ranking\
    --dataset rec_Amazon_Movie_dpo_ranking \
    --create_new_adapter \
    --template llama2 \
    --finetuning_type lora \
    --lora_target all \
    --output_dir dpo_Amazon_Movie_ranking \
    --overwrite_output_dir \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --dpo_ftx 1.0 \
    --logging_steps 1 \
    --learning_rate 1e-4 \
    --num_train_epochs 3.0 \
    --max_samples ${i} \
    --plot_loss \
    --fp16
        

rm -rf ../dpo_training.log
CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage dpo \
    --do_eval \
    --model_name_or_path ../llama-2-7b-chat-hf \
    --adapter_name_or_path sft_Amazon_Movie_ranking,dpo_Amazon_Movie_ranking \
    --dataset rec_Amazon_Movie_eval_ranking \
    --template llama2 \
    --finetuning_type lora \
    --lora_target all \
    --output_dir dpo_Amazon_Movie_ranking_eval \
    --eval_accumulation_steps 16 \
    --per_device_eval_batch_size 1 \
    --logging_steps 10 \
    --max_samples 1000 \
    --save_steps 10 \
    --fp16
  
