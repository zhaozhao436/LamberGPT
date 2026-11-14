"""Shared training pipeline. Run one of the three small entry scripts."""
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from datasets import load_dataset
from transformers import (
    AutoConfig, AutoModelForCausalLM, AutoTokenizer, HfArgumentParser,
    Trainer, TrainingArguments, set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

from data import TrainingCollator, instruction_batch, pretrain_batch


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = None
    config_name: Optional[str] = None
    tokenizer_name_or_path: Optional[str] = None
    cache_dir: Optional[str] = None
    torch_dtype: str = "auto"
    peft_path: Optional[str] = None
    trainable: str = "q_proj,v_proj"
    modules_to_save: Optional[str] = None
    lora_rank: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1


@dataclass
class DataArguments:
    dataset_dir: Optional[str] = None
    train_file: Optional[str] = None
    validation_file: Optional[str] = None
    block_size: int = 1024
    preprocessing_num_workers: Optional[int] = None
    overwrite_cache: bool = False


def input_files(args, task, validation=False):
    filename = args.validation_file if validation else args.train_file
    if filename:
        path = Path(filename)
        if not path.is_absolute() and args.dataset_dir:
            path = Path(args.dataset_dir) / path
        files = [path]
    elif not validation and args.dataset_dir:
        suffixes = ("*.txt",) if task == "pt" else ("*.json", "*.jsonl")
        files = sorted(path for pattern in suffixes for path in Path(args.dataset_dir).glob(pattern))
        if args.validation_file:
            excluded = Path(input_files(args, task, validation=True)[0]).resolve()
            files = [path for path in files if path.resolve() != excluded]
    else:
        raise ValueError("Supply --train_file/--dataset_dir and --validation_file when evaluating")
    if not files or any(not path.is_file() for path in files):
        raise ValueError("No matching input files found")
    return [str(path) for path in files]


def main(task="pt", use_lora=False):
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, args = parser.parse_json_file(str(Path(sys.argv[1]).resolve()))
    else:
        model_args, data_args, args = parser.parse_args_into_dataclasses()
    if not args.do_train and not args.do_eval:
        raise ValueError("Specify --do_train or --do_eval")
    if data_args.block_size < 2:
        raise ValueError("--block_size must be at least 2")
    if not use_lora and model_args.peft_path:
        raise ValueError("--peft_path requires a LoRA entry script")
    set_seed(args.seed)
    checkpoint = args.resume_from_checkpoint
    output = Path(args.output_dir)
    if args.do_train and output.exists() and not args.overwrite_output_dir:
        checkpoint = checkpoint or get_last_checkpoint(str(output))
        if checkpoint is None and any(output.iterdir()):
            raise ValueError("Output directory is not empty; choose another or use --overwrite_output_dir")

    tokenizer_source = model_args.tokenizer_name_or_path or model_args.model_name_or_path
    if not tokenizer_source:
        raise ValueError("Supply --tokenizer_name_or_path for training from scratch")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, cache_dir=model_args.cache_dir)
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must define an EOS token")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    dtype = model_args.torch_dtype
    if dtype != "auto":
        if dtype not in ("float32", "float16", "bfloat16"):
            raise ValueError("Unsupported --torch_dtype")
        dtype = getattr(torch, dtype)
    if model_args.model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path, torch_dtype=dtype, cache_dir=model_args.cache_dir)
    elif model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, cache_dir=model_args.cache_dir)
        model = AutoModelForCausalLM.from_config(config)
    else:
        raise ValueError("Supply --model_name_or_path or --config_name")
    if len(tokenizer) != model.get_input_embeddings().num_embeddings:
        model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    if use_lora:
        from peft import LoraConfig, PeftModel, TaskType, get_peft_model

        if model_args.peft_path:
            model = PeftModel.from_pretrained(model, model_args.peft_path, is_trainable=args.do_train)
        else:
            config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                target_modules=model_args.trainable.split(","),
                modules_to_save=model_args.modules_to_save.split(",") if model_args.modules_to_save else None,
                r=model_args.lora_rank, lora_alpha=model_args.lora_alpha,
                lora_dropout=model_args.lora_dropout,
            )
            model = get_peft_model(model, config)
        if args.gradient_checkpointing:
            model.enable_input_require_grads()
        model.print_trainable_parameters()
    if args.gradient_checkpointing:
        model.config.use_cache = False

    def prepare(validation=False):
        files = input_files(data_args, task, validation)
        raw = load_dataset("text" if task == "pt" else "json", data_files=files,
                           split="train", cache_dir=model_args.cache_dir)
        required = {"text"} if task == "pt" else {"instruction", "output"}
        if not required.issubset(raw.column_names):
            raise ValueError(f"Missing columns: {required - set(raw.column_names)}")
        function = pretrain_batch if task == "pt" else instruction_batch
        result = raw.map(function, batched=True,
                         fn_kwargs={"tokenizer": tokenizer, "block_size": data_args.block_size},
                         remove_columns=raw.column_names, num_proc=data_args.preprocessing_num_workers,
                         load_from_cache_file=not data_args.overwrite_cache)
        if not len(result):
            raise ValueError("No usable samples remain; check input data and --block_size")
        return result

    with args.main_process_first(desc="Preparing datasets"):
        train_data = prepare() if args.do_train else None
        eval_data = prepare(validation=True) if args.do_eval else None
    trainer = Trainer(model=model, args=args, tokenizer=tokenizer,
                      train_dataset=train_data, eval_dataset=eval_data,
                      data_collator=TrainingCollator(tokenizer.pad_token_id))
    if args.do_train:
        result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        trainer.save_state()
        trainer.save_metrics("train", result.metrics)
        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(args.output_dir)
    if args.do_eval:
        metrics = trainer.evaluate()
        if task == "pt":
            try:
                metrics["perplexity"] = math.exp(metrics["eval_loss"])
            except OverflowError:
                metrics["perplexity"] = float("inf")
        trainer.save_metrics("eval", metrics)
