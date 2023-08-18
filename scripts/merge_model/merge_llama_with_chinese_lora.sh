python scripts/merge_llama_with_chinese_lora.py \
    --base_model ~/zhaomengchen/BELLE/checkpoints/LLaMA/13B-hf \
    --lora_model ~/zhaomengchen/Chinese-LLaMA-Alpaca-main/checkpoints/chinese-alpaca-lora-13b \
    --output_type huggingface \
    --output_dir ~/zhaomengchen/Chinese-LLaMA-Alpaca-main/checkpoints/chinese-alpaca-13b
    