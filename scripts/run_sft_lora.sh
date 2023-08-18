lr=1e-4
lora_rank=8
lora_alpha=32
lora_trainable="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj"
modules_to_save="embed_tokens,lm_head"
lora_dropout=0.05

pretrained_model=../models/open_llama_7b
chinese_tokenizer_path=./tokenizer/output/merged_llama_tokenizer_hf
dataset_dir=../instruction
train_file=liver_instruction_task175-0.7k.jsonl
validation_file=pldccmt-45w_dev.jsonl

per_device_train_batch_size=4
per_device_eval_batch_size=4
training_steps=2000     # 2500 for yitushibie 2 peoch
max_seq_length=512
gradient_accumulation_steps=8
output_dir=experiments/outputs/
deepspeed_config_file=scripts/ds_zero2_no_offload.json

CUDA_VISIBLE_DEVICES=6 torchrun --nproc_per_node 1 --master_port 29501 train_sft_lora.py \
    --deepspeed ${deepspeed_config_file} \
    --model_name_or_path ${pretrained_model} \
    --tokenizer_name_or_path ${chinese_tokenizer_path} \
    --dataset_dir ${dataset_dir} \
    --train_file ${train_file} \
    --validation_file ${validation_file} \
    --validation_split_percentage 0.001 \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --per_device_eval_batch_size ${per_device_eval_batch_size} \
    --do_train \
    --do_eval \
    --seed $RANDOM \
    --fp16 \
    --max_steps ${training_steps} \
    --lr_scheduler_type cosine \
    --learning_rate ${lr} \
    --warmup_ratio 0.03 \
    --weight_decay 0 \
    --logging_strategy steps \
    --logging_steps 10 \
    --save_strategy steps \
    --save_total_limit 3 \
    --evaluation_strategy steps \
    --eval_steps 250 \
    --save_steps 250 \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --preprocessing_num_workers 8 \
    --max_seq_length ${max_seq_length} \
    --output_dir ${output_dir}/lambergpt-13b-lora-${lr}-steps${training_steps} \
    --overwrite_output_dir \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --lora_rank ${lora_rank} \
    --lora_alpha ${lora_alpha} \
    --trainable ${lora_trainable} \
    --modules_to_save ${modules_to_save} \
    --lora_dropout ${lora_dropout} \
    --torch_dtype float16 \
    --validation_file ${validation_file} \
    --gradient_checkpointing \
    --ddp_find_unused_parameters False 

