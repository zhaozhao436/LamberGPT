"""Token preparation shared by pretraining and supervised fine-tuning."""


def pretrain_batch(examples, tokenizer, block_size):
    tokens = []
    for text in examples["text"]:
        tokens.extend(tokenizer.encode(text, add_special_tokens=False))
        tokens.append(tokenizer.eos_token_id)
    chunks = [tokens[i:i + block_size] for i in range(0, len(tokens), block_size)]
    chunks = [chunk for chunk in chunks if len(chunk) > 1]
    return {"input_ids": chunks, "labels": [chunk.copy() for chunk in chunks]}


def instruction_batch(examples, tokenizer, block_size):
    inputs, labels = [], []
    for index, instruction in enumerate(examples["instruction"]):
        if isinstance(instruction, list):
            instruction = instruction[0] if instruction else ""
        context = examples.get("input", [""] * len(examples["instruction"]))[index] or ""
        answer = examples["output"][index]
        if not isinstance(instruction, str) or not isinstance(context, str) or not isinstance(answer, str):
            raise ValueError("instruction, input and output must contain strings")
        source = tokenizer.encode(instruction + context, add_special_tokens=True)
        target = tokenizer.encode(answer, add_special_tokens=False) + [tokenizer.eos_token_id]
        ids = (source + target)[:block_size]
        targets = ([-100] * len(source) + target)[:block_size]
        # Causal LM shifts labels by one: at least one answer token must survive.
        if not any(value != -100 for value in targets[1:]):
            continue
        inputs.append(ids)
        labels.append(targets)
    return {"input_ids": inputs, "labels": labels}


class TrainingCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        import torch

        width = max(len(row["input_ids"]) for row in features)
        inputs, labels, masks = [], [], []
        for row in features:
            size = len(row["input_ids"])
            padding = width - size
            inputs.append(list(row["input_ids"]) + [self.pad_token_id] * padding)
            labels.append(list(row["labels"]) + [-100] * padding)
            # Build from lengths, because EOS can also be the padding token.
            masks.append([1] * size + [0] * padding)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in
                (("input_ids", inputs), ("labels", labels), ("attention_mask", masks))}
