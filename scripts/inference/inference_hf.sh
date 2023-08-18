CUDA_VISIBLE_DEVICES=1 python scripts/inference/inference_hf.py \
    --base_model checkpoints/chinese-alpaca-13b \
    --lora_model experiments/outputs/yitushibie-chinese-alpaca-13b-lora-1e-4-fix/lora_model \
    --tokenizer_path experiments/outputs/yitushibie-chinese-alpaca-13b-lora-1e-4-fix/lora_model \
    --data_file data/IMCS-DAC/test_instruction_2.json \
    --predictions_file experiments/outputs/yitushibie-chinese-alpaca-13b-lora-1e-4-fix/test_predictions.json \
    --with_prompt