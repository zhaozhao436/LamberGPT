import argparse
import json, os
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import  PeftModel

parser = argparse.ArgumentParser()
parser.add_argument('--base_model', default=None, type=str, required=True)
parser.add_argument('--lora_model', default=None, type=str,help="If None, perform inference on the base model")
parser.add_argument('--tokenizer_path',default=None,type=str)
parser.add_argument('--data_file',default=None, type=str,help="A file that contains instructions (one instruction per line)")
parser.add_argument('--with_prompt',action='store_true',help="wrap the input with the prompt automatically")
parser.add_argument('--interactive',action='store_true',help="run in the instruction mode (single-turn)")
parser.add_argument('--predictions_file', default='./predictions.json', type=str)
parser.add_argument('--gpus', default="0", type=str)
parser.add_argument('--only_cpu',action='store_true',help='only use CPU for inference')
args = parser.parse_args()
if args.only_cpu is True:
    args.gpus = ""

args.base_model = ''
args.lora_model = ''
args.tokenizer_path = ''
args.data_file = ''
args.predictions_file = ''

os.environ["CUDA_VISIBLE_DEVICES"] = '6'



generation_config = dict(
    temperature=0.9,
    top_k=40,
    top_p=0.9,
    do_sample=True,
    num_beams=1,
    repetition_penalty=1.3,
    max_new_tokens=400
    )



if __name__ == '__main__':
    load_type = torch.float16
    if torch.cuda.is_available():
        device = torch.device(0)
    else:
        device = torch.device('cpu')
    if args.tokenizer_path is None:
        args.tokenizer_path = args.lora_model
        if args.lora_model is None:
            args.tokenizer_path = args.base_model
    tokenizer = LlamaTokenizer.from_pretrained(args.tokenizer_path)

    base_model = LlamaForCausalLM.from_pretrained(
        args.base_model, 
        load_in_8bit=False,
        torch_dtype=load_type,
        low_cpu_mem_usage=True,
        device_map='auto',
        )

    model_vocab_size = base_model.get_input_embeddings().weight.size(0)
    tokenzier_vocab_size = len(tokenizer)
    print(f"Vocab of the base model: {model_vocab_size}")
    print(f"Vocab of the tokenizer: {tokenzier_vocab_size}")
    if model_vocab_size!=tokenzier_vocab_size:
        assert tokenzier_vocab_size > model_vocab_size
        print("Resize model embeddings to fit tokenizer")
        base_model.resize_token_embeddings(tokenzier_vocab_size)
    if args.lora_model is not None:
        print("loading peft model")
        model = PeftModel.from_pretrained(base_model, args.lora_model,torch_dtype=load_type,device_map='auto',)
        # model = PeftModel.from_pretrained(base_model, args.lora_model, torch_dtype=torch.float16)
    else:
        model = base_model

    if device==torch.device('cpu'):
        model.float()

    # test data
    with open(args.data_file,'r') as f:
        # examples = [l.strip() for l in f.readlines()]
        examples = [eval(l) for l in f.readlines()]
    print("first 10 examples:")
    for example in examples[:10]:
        print(example)
    model.eval()

    prompt_column = "input"
    response_column = "target"
    history_column = None
    prefix = ""
    ignore_pad_token_for_loss = True
    max_source_length, max_target_length = 512, 64
    max_seq_length = max_source_length + max_target_length
    with torch.no_grad():
        print("Start inference.")
        results = []
        for index, example in enumerate(examples):
            if example[prompt_column]:
                query = example[prompt_column]
                if history_column is None or len(examples[history_column][i]) == 0:
                    prompt = query
                else:
                    prompt = ""
                    history = example[history_column]
                    for turn_idx, (old_query, response) in enumerate(history):
                        prompt += "[Round {}]\n问：{}\n答：{}\n".format(turn_idx, old_query, response)
                    prompt += "[Round {}]\n问：{}\n答：".format(len(history), query)
            input_text = prompt
            
            inputs = tokenizer(input_text, 
                               max_length=max_source_length,
                               truncation=True,
                               padding=True,
                               return_tensors="pt"
                            )  #add_special_tokens=False ?

            generation_output = model.generate(
                input_ids = inputs["input_ids"].to(device), 
                attention_mask = inputs['attention_mask'].to(device),
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                **generation_config
            )
            s = generation_output[0]
            output = tokenizer.decode(s,skip_special_tokens=True)
            if args.with_prompt:
                # response = output.split("### Response:")[1].strip()
                response = output
            else:
                response = output
            print(f"======={index}=======")
            print(f"Input: {example}\n")
            print(f"Output: {response}\n")

            results.append({"Input":input_text,"Output":response})

        dirname = os.path.dirname(args.predictions_file)
        os.makedirs(dirname, exist_ok=True)
        with open(args.predictions_file, 'w') as f:
            json.dump(results,f,ensure_ascii=False,indent=2)
        with open(dirname + '/generation_config.json','w') as f:
            json.dump(generation_config, f, ensure_ascii=False, indent=2)
