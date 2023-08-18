from fastapi import FastAPI
from fastapi.responses import FileResponse
import uvicorn
from urllib.request import quote,unquote
from transformers import AutoModel, LlamaTokenizer, AutoConfig, LlamaForCausalLM, AutoModelForCausalLM, GenerationConfig
from peft import PeftModel
import torch
import time

app_name = "fake_model"
app_host ="0.0.0.0"
app_port = 8889
app = FastAPI(title=app_name)


base_model_path = '../models/open_llama_13b'
# lora_model = '../../experiments/outputs/yitushibie-chinese-alpaca-13b-lora-1e-4-fix/lora_model'
# tokenizer_path = '../../experiments/outputs/yitushibie-chinese-alpaca-13b-lora-1e-4-fix/lora_model'


load_type = torch.float16
if torch.cuda.is_available():
    device = torch.device(0)
else:
    device = torch.device('cpu')

tokenizer = LlamaTokenizer.from_pretrained(base_model_path)

model = LlamaForCausalLM.from_pretrained(
    base_model_path, 
    load_in_8bit=False,
    torch_dtype=load_type,
    low_cpu_mem_usage=True,
    device_map='auto',
    )

# model_vocab_size = base_model.get_input_embeddings().weight.size(0)
# tokenzier_vocab_size = len(tokenizer)
# print(f"Vocab of the base model: {model_vocab_size}")
# print(f"Vocab of the tokenizer: {tokenzier_vocab_size}")
# if model_vocab_size!=tokenzier_vocab_size:
#     assert tokenzier_vocab_size > model_vocab_size
#     print("Resize model embeddings to fit tokenizer")
#     base_model.resize_token_embeddings(tokenzier_vocab_size)

# model = PeftModel.from_pretrained(base_model, lora_model, torch_dtype=load_type,device_map='auto',)

if device==torch.device('cpu'):
    model.float()

model.eval()


def evaluate(
    input,
    temperature=0.1,
    top_p=0.75,
    top_k=40,
    num_beams=4,
    max_new_tokens=512,
    min_new_tokens=1,
    repetition_penalty=1.2,
    **kwargs
):
    inputs = tokenizer(input, return_tensors="pt")
    input_ids = inputs["input_ids"].cuda()
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        # bos_token_id=1,
        # eos_token_id=2,
        # pad_token_id=0,
        max_new_tokens=max_new_tokens, # max_length=max_new_tokens+input_sequence
        min_new_tokens=min_new_tokens, # min_length=min_new_tokens+input_sequence
        **kwargs,
    )
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=False,
            repetition_penalty=1.3,
        )
        output = generation_output.sequences[0]
        output = tokenizer.decode(output, skip_special_tokens=True)
        print(output)
        yield output


@app.get('/qa/{question}')
def get_data(question: str):
    question = unquote(question)#utf-8
    print('question: ', question)

    # answer, history = model.chat(tokenizer, question, history=[])
    answer = evaluate(question)

    print('answer: ', answer)
    return answer


# reload 热加载，修改了自动重启
uvicorn.run(app=app, host=app_host, port=app_port, reload=False)


