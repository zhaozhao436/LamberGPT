# coding=utf-8
import os
import datasets
from datasets import load_dataset, concatenate_datasets
from files_weights import med_weights


data_cache_dir = '/new_disk/med_group/antiforget_txt/group_3_5/.cache'
files = sorted(os.listdir(data_cache_dir))

total_tokens_number = 0
total_tokens_number_weighted = 0
for i, file in enumerate(files):
    filename = file.split(".")[0]
    if filename.endswith('text'):
        continue
    cache_path = os.path.join(data_cache_dir, filename)
    processed_dataset = datasets.load_from_disk(cache_path, keep_in_memory=False)
    # 一个样本的tokens数 * 样本数
    if file + '.txt' in list(med_weights.keys()):
        weight = med_weights[file + '.txt']
    else:
        weight = 1
    count = len(processed_dataset['train'][0]['input_ids']) * len(processed_dataset['train'])
    print('{}, tokens number: {}, weighted tokens number: {}'.format(file, count, count * weight))
    total_tokens_number += count
    total_tokens_number_weighted += count * weight
print('total tokens number: ', total_tokens_number)
print('total weighted tokens_number: ', total_tokens_number_weighted)
