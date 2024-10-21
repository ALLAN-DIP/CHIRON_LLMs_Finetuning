import os
import json
import csv
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftConfig, PeftModel
from torch.nn import DataParallel

import time
from tqdm import tqdm


def load_model(base_model_name, adapter_path, tokenizer_path, device='cpu'):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = AutoModelForCausalLM.from_pretrained(base_model_name)
    model = PeftModel.from_pretrained(model, adapter_path)
    if torch.cuda.device_count() > 1:
        model = DataParallel(model)
    model.to(device)
    
    return tokenizer, model


def generate_text(prompt, tokenizer, model, device='cpu'):
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    model.eval()
    with torch.no_grad():
        output_ids = model.module.generate(input_ids) if isinstance(model, DataParallel) else model.generate(input_ids)

    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    return generated_text


if __name__ == "__main__":
    overall_time = time.time()
    base_model_name = "meta-llama/Llama-2-7b-chat-hf"
    adapter_path = '../output/no_board_history_with_sys_history_cicero_10epoch_lr5e-5_batch2'
    tokenizer_path = '../output/no_board_history_with_sys_history_cicero_10epoch_lr5e-5_batch2'
    data_path = '../dataset/inference_no_board_history_with_sys_history_cicero.json'
    split = 'val'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    tokenizer, model = load_model(base_model_name, adapter_path, tokenizer_path, device)

    with open(data_path) as f:
        data = json.load(f)[split]
    
    with open('val_no_board_history_with_sys_history_cicero_10epoch_lr5e-5_batch2.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(('id', 'prompt', 'desired_output', 'model_output'))
    for sample in tqdm(data):
        index = sample['text'].find('[/INST] ')
        model_input = sample['text'][:index+8]
        desired_output = sample['text'][index+8:-4]
        index = model_input.find('<</SYS>>\n\n')
        prompt = model_input[index+10:-8]
        generated_text = generate_text(model_input, tokenizer, model, device)
        index = generated_text.find('[/INST] ')
        model_output = generated_text[index+8:]
        with open('val_no_board_history_with_sys_history_cicero_10epoch_lr5e-5_batch2.csv', 'a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow((sample['id'], prompt, desired_output, model_output))

    print('Overall time', time.time() - overall_time)