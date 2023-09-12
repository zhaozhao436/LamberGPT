lr=2e-4
lora_rank=8
lora_alpha=32
lora_trainable="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj"
modules_to_save="embed_tokens,lm_head"
lora_dropout=0.05

pretrained_model=../models/open_llama_7b
chinese_tokenizer_path=./tokenizer/output/merged_llama_tokenizer_2/merged_llama_tokenizer_hf
dataset_dir=/new_disk/med_group/antiforget_txt/group_3_5
data_cache=/new_disk/med_group/antiforget_txt/group_3_5/.cache
per_device_train_batch_size=1
per_device_eval_batch_size=1
training_steps=20
gradient_accumulation_steps=16
output_dir=experiments/outputs/clm_pt_full_med_jsonline

deepspeed_config_file=scripts/ds_zero2_no_offload.json

CUDA_VISIBLE_DEVICES=3 torchrun --nnodes 1 --nproc_per_node 1 --master_port 29600 train_clm_pt_full.py \
    --deepspeed ${deepspeed_config_file} \
    --model_name_or_path ${pretrained_model} \
    --tokenizer_name_or_path ${chinese_tokenizer_path} \
    --dataset_dir ${dataset_dir} \
    --data_cache_dir ${data_cache} \
    --validation_split_percentage 0.001 \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --per_device_eval_batch_size ${per_device_eval_batch_size} \
    --do_train \
    --do_eval \
    --seed 1409 \
    --fp16 \
    --max_steps ${training_steps} \
    --lr_scheduler_type cosine \
    --learning_rate ${lr} \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --logging_strategy steps \
    --logging_steps 10 \
    --save_strategy steps \
    --save_total_limit 1 \
    --save_steps 5 \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --preprocessing_num_workers 64 \
    --block_size 1024 \
    --output_dir ${output_dir} \
    --overwrite_output_dir \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --lora_rank ${lora_rank} \
    --lora_alpha ${lora_alpha} \
    --trainable ${lora_trainable} \
    --modules_to_save ${modules_to_save} \
    --lora_dropout ${lora_dropout} \
    --torch_dtype float16 \
    --gradient_checkpointing \
    --ddp_find_unused_parameters False
