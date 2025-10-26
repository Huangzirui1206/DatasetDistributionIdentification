'''
Acknowledgement:
https://github.com/tsinghua-fib-lab/ANeurIPS2024_SPV-MIA
@misc{fu2023practical,
    title={Practical Membership Inference Attacks against Fine-tuned Large Language Models via Self-prompt Calibration},
    author={Wenjie Fu and Huandong Wang and Chen Gao and Guanghua Liu and Yong Li and Tao Jiang},
    year={2023},
    eprint={2311.06062},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
'''


import os
import sys
here = os.path.dirname(__file__)
sys.path.append(here)
import json
import random
import datasets
from utils.tools import create_folder

block_size_ = None
tokenizer_ = None
max_buff_size_ = None
text_column = None


def packing_texts(examples):
    packed_texts = []
    packed_ids = []
    # for key in examples.keys():
    assert list(examples.keys()) == ["text"]
    assert len(examples["text"])
        
    iterator = iter(examples["text"])
    # for sentence in examples["text"]:
    total_num = 0
    drop_num = 0
    buffer, buffer_len = [], 0
    while True:
        if buffer_len >= max_buff_size_:
            assert len(buffer) # when break from the loop, buffer shouoldn't be empty
            break
        try:
            buffer.append(next(iterator))
            buffer_len += len(buffer[-1])
        except StopIteration:
            assert len(buffer) # when break from the loop, buffer shouoldn't be empty
            break
        
    tokenized_inputs = tokenizer_(buffer, truncation=False)["input_ids"]
    inputs = tokenizer_.batch_decode(tokenized_inputs)
    tokenized_inputs = tokenizer_(inputs, truncation=False)["input_ids"]
    all_token_ids = []
    for tokenized_input in tokenized_inputs:
        all_token_ids.extend(tokenized_input)
    for i in range(0, len(all_token_ids), block_size_):
        input_ids = all_token_ids[i: i + block_size_]
        if len(input_ids) >= 0.1 * block_size_:
            packed_ids.append(input_ids)
            input_text = tokenizer_.decode(input_ids)
            packed_texts.append(input_text)
            total_num += 1
        else:    
            drop_num += 1
    # print(f"Total examples: {total_num}, dropped num: {drop_num}, dropped rate: {1 - drop_num/total_num}")
    return {
        "text": packed_texts
    }
    

def dataset_prepare(dataset_name,
                    dataset_config_name,
                    validation_split_percentage,
                    packing,
                    cache_path,
                    use_dataset_cache,
                    preprocessing_num_workers,
                    block_size=1024,
                    tokenizer=None, 
                    num_of_sequences=1024, 
                    chars_per_token=3.6,
                    train_set_size=0,
                    ):
    # raw_datasets = datasets.load_dataset(args.dataset_name, args.dataset_config_name)['train']
    # if "validation" in raw_datasets.keys():
    #     train_dataset = raw_datasets["train"]
    #     valid_dataset = raw_datasets["validation"]
    # else:
    train_dataset = datasets.load_dataset(
        dataset_name,
        dataset_config_name,
        split=f"train[:{int((1-validation_split_percentage)*100)}%]"
    )
    valid_dataset = datasets.load_dataset(
        dataset_name,
        dataset_config_name,
        split=f"train[{int((1-validation_split_percentage)*100)}%:]",
    )
    
    if train_set_size:
        if train_set_size > len(train_dataset):
            train_set_size = len(train_dataset)
        train_dataset = train_dataset.select(range(train_set_size))
    
    # dataset = datasets.load_dataset(
    #     dataset_name,
    #     dataset_config_name,
    # )['train']
    # print(dataset)
    # if validation_split_percentage > 0:
    #     train_dataset = dataset.select(range(int(len(dataset) * (1 - validation_split_percentage))))
    #     valid_dataset = dataset.select(range(int(len(dataset) * validation_split_percentage), len(dataset)))
    # else:
    #     train_dataset = dataset
    #     valid_dataset = dataset.select(range(0.1 * len(dataset), len(dataset)))
        
    # train_dataset = datasets.Dataset.from_dict(train_dataset)
    # valid_dataset = datasets.Dataset.from_dict(valid_dataset)
        
    # print(train_dataset, valid_dataset)
    # assert 0
    
    # train_idxs = set(random.sample(range(len(raw_datasets)), int(len(raw_datasets) * (1 - args.validation_split_percentage))))
    # valid_idxs = set(range(len(raw_datasets))) - train_idxs
    # train_dataset = datasets.Dataset.from_dict(raw_datasets[train_idxs])
    # valid_dataset = datasets.Dataset.from_dict(raw_datasets[valid_idxs])

    global text_column
    column = train_dataset.column_names
    if "text" in column:
        text_column = "text"
    elif "document" in column:
        text_column = "document"
    elif "content" in column:
        text_column = "content"

    train_dataset = train_dataset.select_columns(text_column)
    valid_dataset = valid_dataset.select_columns(text_column)
    if text_column != "text":
        train_dataset = train_dataset.rename_column(text_column, "text")
        valid_dataset = valid_dataset.rename_column(text_column, "text")
    
    print("The dataset before pre-process is:")
    print(train_dataset)
    print(valid_dataset)
    
    if packing:
        global block_size_, tokenizer_, max_buff_size_
        block_size_ = block_size
        max_buff_size_ = block_size_ * chars_per_token * num_of_sequences
        tokenizer_ = tokenizer
        create_folder(f"{cache_path}/{dataset_name}/{dataset_config_name}")
        train_dataset = train_dataset.map(
            packing_texts,
            batched=True,
            # batch_size=None,
            num_proc=preprocessing_num_workers,
            cache_file_name=f"{cache_path}/{dataset_name}/{dataset_config_name}/train_dataset.arrow",
            load_from_cache_file=use_dataset_cache,
            desc=f"Packing texts in chunks of {block_size_} tokens"
        )
        valid_dataset = valid_dataset.map(
            packing_texts,
            batched=True,
            # batch_size=None,
            num_proc=preprocessing_num_workers,
            cache_file_name=f"{cache_path}/{dataset_name}/{dataset_config_name}/valid_dataset.arrow",
            load_from_cache_file=use_dataset_cache,
            desc=f"Packing texts in chunks of {block_size} tokens"
        )
    
    '''if train_set_size > 0:
        if randomize_train_set:
            random_indexes = random.sample(range(len(train_dataset)), train_set_size)
            train_dataset = train_dataset.select(random_indexes)
            range_json = {'type':'random', 'start':0, 
                          'end':0, 'random_indexes':random_indexes}
        else:
            train_set_end = min(train_set_start + train_set_size, len(train_dataset))
            train_dataset = train_dataset.select(range(train_set_start, train_set_end))
            range_json = {'type':'sequence', 'start':train_set_start, 
                          'end':train_set_end, 'random_indexes':[]}
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'range.json'), 'w') as json_file:
            json.dump(range_json, json_file, indent=4)
        
    if val_set_size > 0:
        valid_dataset = valid_dataset.select(range(min(val_set_size, len(valid_dataset))))    '''  
    
    print("The dataset after pre-process is:")
    print(train_dataset, valid_dataset)
    
    return train_dataset, valid_dataset


def test_dataset_prepare(dataset_name,
                    dataset_config_name,
                    packing,
                    cache_path,
                    use_dataset_cache,
                    preprocessing_num_workers,
                    block_size=1024,
                    tokenizer=None, 
                    num_of_sequences=1024, 
                    chars_per_token=3.6,
                    ):
    # raw_datasets = datasets.load_dataset(args.dataset_name, args.dataset_config_name)['train']
    # if "validation" in raw_datasets.keys():
    #     train_dataset = raw_datasets["train"]
    #     valid_dataset = raw_datasets["validation"]
    # else:
    test_dataset = datasets.load_dataset(
        dataset_name,
        dataset_config_name,
        split=f"test"
    )
        
    # train_idxs = set(random.sample(range(len(raw_datasets)), int(len(raw_datasets) * (1 - args.validation_split_percentage))))
    # valid_idxs = set(range(len(raw_datasets))) - train_idxs
    # train_dataset = datasets.Dataset.from_dict(raw_datasets[train_idxs])
    # valid_dataset = datasets.Dataset.from_dict(raw_datasets[valid_idxs])

    global text_column
    column = test_dataset.column_names
    if "text" in column:
        text_column = "text"
    elif "document" in column:
        text_column = "document"
    elif "content" in column:
        text_column = "content"

    test_dataset = test_dataset.select_columns(text_column)
    if text_column != "text":
        test_dataset = test_dataset.rename_column(text_column, "text")
    
    if packing:
        global block_size_, tokenizer_, max_buff_size_
        block_size_ = block_size
        max_buff_size_ = block_size_ * chars_per_token * num_of_sequences
        tokenizer_ = tokenizer
        create_folder(f"{cache_path}/{dataset_name}_test/{dataset_config_name}")
        test_dataset = test_dataset.map(
            packing_texts,
            batched=True,
            # batch_size=None,
            num_proc=preprocessing_num_workers,
            cache_file_name=f"{cache_path}/{dataset_name}_test/{dataset_config_name}/train_dataset",
            load_from_cache_file=use_dataset_cache,
            desc=f"Packing texts in chunks of {block_size_} tokens"
        )
    return test_dataset