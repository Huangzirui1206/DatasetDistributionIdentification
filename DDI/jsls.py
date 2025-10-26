from wordfreq import word_frequency
import re
from transformers import AutoTokenizer, AutoModel
from collections import Counter
from datasets import load_dataset, arrow_dataset
import os
import math
import torch
import numpy as np
from tqdm import tqdm
from typing import List, Iterable
import time
import faiss
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.metrics import auc as compute_area_under_curve
from types import SimpleNamespace
from sentence_transformers import SentenceTransformer

import random

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# utility functions
def get_dataset(path: str, seed=42, data_key='train') -> arrow_dataset.Dataset:
    if path.endswith(".json"):  # todo: support jsonl
        data = load_dataset("json", data_files=path)
    else:
        data = load_dataset(path)
    data = data[data_key].shuffle(seed)
    return data

def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created.")
    else:
        print(f"Folder '{folder_path}' already exists.")

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
        
def pack_dataset(data: arrow_dataset.Dataset, 
                 data_name: str, # For caching path
                 column_name: str,
                 dataset_config_name:str=None,
                 preprocessing_num_workers:int=1,
                 cache_path:str='./cache/weighed_mauve',
                 use_dataset_cache:bool=True,
                 block_size=1024,
                 num_of_sequences=1024, 
                 chars_per_token=3.6,
                 ) -> List[List[int]]:
    if data_name.endswith(".json"):
        data_name = data_name.replace(".json", "")
    data = data.select_columns(column_name)
    if column_name != "text":
        data = data.rename_column(column_name, "text")
    global block_size_, tokenizer_, max_buff_size_
    block_size_ = block_size
    max_buff_size_ = block_size_ * chars_per_token * num_of_sequences
    tokenizer_ = tokenizer
    create_folder(f"{cache_path}/{data_name}/{dataset_config_name}")
    data = data.map(
        packing_texts,
        batched=True,
        # batch_size=None,
        num_proc=preprocessing_num_workers,
        cache_file_name=f"{cache_path}/{data_name}/{dataset_config_name}/train_dataset.arrow",
        load_from_cache_file=use_dataset_cache,
        desc=f"Packing texts in chunks of {block_size_} tokens"
    )
    # print(f"The dataset after pre-process is: {data}")
    return data['text']

def remove_non_alphanumeric(input_string: str) -> str:
    # 使用正则表达式，匹配并去除非字母和数字的字符
    result = re.sub(r'[^a-zA-Z0-9]', '', input_string)
    return result

def tokenize_to_word(tokens: List[str]) -> List[str]:
    words = []
    current_word = []
    
    # 用正则去除标点符号（可根据需要调整）
    punctuation_pattern = re.compile(r'[^\w\s]')
    
    for token in tokens:
        start_with_space = token.startswith('Ġ') or token.startswith('▁')
        contain_no_alphanum = not re.search(r'[a-zA-Z0-9]', token)
        # 如果token包含'Ġ'，说明是一个空格
        if contain_no_alphanum:
            if current_word:
                words.append(''.join(current_word))
                current_word = []  # 清空current_word
            continue # 跳过当前不包含字母和数字的token
        elif start_with_space:
            # 如果current_word非空，则加入words列表
            if current_word:
                words.append(''.join(current_word))
                current_word = []  # 清空current_word
            current_word.append(token[1:])  # 去掉'Ġ'并拼接
        else:
            # 如果没有空格，继续拼接当前单词
            current_word.append(token)
    
    # 最后将最后一个单词添加到words
    if current_word:
        words.append(''.join(current_word))
    
    # 忽略标点符号
    words = [remove_non_alphanumeric(word) for word in words if not punctuation_pattern.match(word) and word != '']
    
    return words

def compute_dataset_word_frequency(data: List[str], tokenizer: AutoTokenizer, max_word_length=20) -> dict:
    words = []
    for sequence in tqdm(data, desc='compute dataset word frequency: '):
        # 替换所有非字母数字的字符为空格
        # sequence = re.sub(r'[^a-zA-Z0-9]', ' ', sequence)
        # 合并多个连续的空格为一个空格
        sequence = re.sub(r'\s+', ' ', sequence)
        # 去掉开头和结尾的空格
        sequence = sequence.strip()
        tokens = tokenizer.tokenize(sequence, 
                            max_length=1024,  # 设置最大长度为模型的最大序列长度
                            truncation=True,  # 启用截断
                            )

        words += tokenize_to_word(tokens)
    
    # 计算单词频数
    word_freq = Counter([word for word in words]) # filter out too short words

    # 计算总的单词数量
    # Laplace smoothing
    # filter out too short words, reduce the weights of too-short word
    total_words = sum([count for _, count in word_freq.items()])
    # total_words = sum([count / avg_word_length * len(word) for word, count in word_freq.items()])
    vocab_size = len(word_freq.items())
    alpha = 1
    
    # 计算每个单词的频率
    # filter out words that only appear once
    word_freq_normalized = {word: (count+alpha) / (total_words+alpha*vocab_size) for word, count in word_freq.items()}
    
    min_freq = alpha / (total_words+alpha*vocab_size)
    
    return word_freq_normalized, min_freq 

def compute_relevance(word: str, dataset_word_freq: dict, relevance_threshold:float=float('inf'), minimum:float=1e-8) -> float:
    q = word_frequency(word, 'en', minimum=1e-8)
    p = dataset_word_freq.get(word, minimum)
    relevance = math.sqrt(p) * math.log2(p / q)
    relevance_threshold = 1
    if relevance > relevance_threshold:
        relevance = relevance_threshold
    return relevance

def compute_dataset_relevance(words: Iterable[str], dataset_word_freq: dict, freq_minimum: float=1e-8):
    dataset_relevance = {}
    for word in tqdm(words, desc='compute dataset relevance: '):
        # word = word.lower()
        dataset_relevance[word] = compute_relevance(word, dataset_word_freq, minimum=freq_minimum)
    return dataset_relevance

def compute_normalized_relevance_score(data: List[str], tokenizer: AutoTokenizer, relevance_scores: dict, verbose: bool=False) -> float: 
    dataset_relevances = []
    for sequence in tqdm(data, desc='compute dataset word frequency: '):
        if len(sequence) == 0:
            continue
        tokens = tokenizer.tokenize(sequence, 
                            max_length=256,  # 设置最大长度为模型的最大序列长度
                            truncation=True,  # 启用截断
                            )
        sequence_words = tokenize_to_word(tokens)
        if len(sequence_words) == 0:
            continue
        # sequence_relevance = 0.0
        for word in sequence_words:
            relevance = relevance_scores.get(word, 0)
            # if relevance > 0:
            #     dataset_relevances.append(relevance)
            dataset_relevances.append(relevance)
    
    if verbose:
        print(f"Mean relevance: {np.mean(dataset_relevances)}")
        print(f"Median relevance: {np.median(dataset_relevances)}")
        print(f"Variance of relevance: {np.var(dataset_relevances)}")
        
    return dataset_relevances

def tokenize_sequences(tokenizer, sequences, max_len=256):
    texts = [sen for sen in sequences if len(sen) > 0]
    tokenized_texts = [
        tokenizer.encode(sen, return_tensors='pt', truncation=True, max_length=max_len)
        for sen in texts
    ]
    return tokenized_texts

def featurize_tokens_from_model(model, tokenized_texts, batch_size, name="", verbose=False):
    """Featurize tokenized texts using models, support batchify
    :param model: HF Transformers model
    :param batch_size: Batch size used during forward pass
    :param tokenized_texts: list of torch.LongTensor of shape (1, length)
    :param verbose: If True, print status and time
    :return:
    """
    device = next(model.parameters()).device
    t1 = time.time()
    feats, chunks, chunk_sent_lengths = [], [], []
    chunk_idx = 0

    while chunk_idx * batch_size < len(tokenized_texts):
        _chunk = [_t.view(-1) for _t in tokenized_texts[chunk_idx * batch_size: (chunk_idx + 1) * batch_size]]
        chunks.append(_chunk)
        chunk_sent_lengths.append([len(_c) for _c in _chunk])
        chunk_idx += 1

    for chunk, chunk_sent_length in tqdm(list(zip(chunks, chunk_sent_lengths)), desc=f"Featurizing {name}"):
        padded_chunk = torch.nn.utils.rnn.pad_sequence(chunk,
                                                       batch_first=True,
                                                       padding_value=0).to(device)
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            [torch.ones(sent_length).long() for sent_length in chunk_sent_length],
            batch_first=True,
            padding_value=0).to(device)
        outs = model(input_ids=padded_chunk,
                     attention_mask=attention_mask,
                     past_key_values=None,
                     output_hidden_states=True,
                     return_dict=True)
        
        h = []
        for hidden_state, sent_length in zip(outs.hidden_states[-1], chunk_sent_length):
            # for i in range(sent_length - 1):
            #     h.append(hidden_state[i].detach().cpu())
            h.append(hidden_state[sent_length-1].detach().cpu())
        h = torch.stack(h, dim=0)
        feats.append(h.cpu())
    t2 = time.time()
    if verbose:
        print(f'Featurize time: {round(t2-t1, 2)}')
    return torch.cat(feats)

def compute_sequence_relevance(data: List[str], dataset_word_relevance: dict, tokenizer: AutoTokenizer) -> List[float]:
    sequence_relevances = []
    for sequence in tqdm(data, desc="compute sequence relevance: "):
        input_ids = tokenizer.encode(sequence, return_tensors='pt', max_length=256, truncation=True)  # 返回 PyTorch 张量
        input_ids = input_ids.to(device)  # 将输入张量移动到指定设备
        input_tokens = tokenizer.tokenize(sequence, max_length=256, truncation=True)  # 返回strings
        
        embedding_relevance_pairs = []
        current_word = []
        current_idxes = []

        # for i, token_embedding in enumerate(contextual_embeddings[0]):  # 遍历每个 token 的 embedding
        #     print(f"Token {i} (Text: {input_tokens[i]}): {token_embedding[:2]}...")  # 打印前 2 维
            
        for i, token in enumerate(input_tokens):
            start_with_space = token.startswith('Ġ') or token.startswith('▁')
            contain_no_alphanum = not re.search(r'[a-zA-Z0-9]', token)
            
            if contain_no_alphanum:
                if current_word:
                    word = ''.join(current_word)
                    # From private prompt, calcute token relevance by the word they belong to, to compensate for multiple-token long words
                    for idx in current_idxes:
                        embedding_relevance_pairs.append((idx, word, dataset_word_relevance.get(word, 0)))
                    current_word = []  # 清空current_word
                    current_idxes = []
                continue # 跳过当前不包含字母和数字的token
            elif start_with_space:
                # 如果current_word非空，则加入words列表
                if current_word:
                    word = ''.join(current_word)
                    for idx in current_idxes:
                        embedding_relevance_pairs.append((idx, word, dataset_word_relevance.get(word, 0)))
                    current_word = []  # 清空current_word
                    current_idxes = []
                current_word.append(token[1:])  # 去掉'Ġ'并拼接
                current_idxes.append(i)
            else:
                # 如果没有空格，继续拼接当前单词
                current_word.append(token)
                current_idxes.append(i)
            
        # 最后将最后一个单词添加到words
        if current_word:
            word = ''.join(current_word)
            for idx in current_idxes:
                embedding_relevance_pairs.append((idx, word, dataset_word_relevance.get(word, 0)))

        # 统计sequence relevance
        sequence_relevance = 0.0
        word_cnt = 0
        for idx, word, relevance in embedding_relevance_pairs:
            if relevance > 0:
                sequence_relevance += relevance
                word_cnt += 1
        
        sequence_relevances.append(sequence_relevance)
        # sequence_relevances.append(sequence_relevance / max(word_cnt, 1))
        # sequence_relevances.append(1)
    
    return sequence_relevances

def weighed_cluster_feats(p, q, p_weight, q_weight, num_clusters,
                  norm='none', whiten=True,
                  pca_max_data=-1,
                  explained_variance=0.9,
                  num_redo=5, max_iter=500,
                  seed=0, verbose=False):
    assert len(p) == len(p_weight)
    assert len(q) == len(q_weight)
    assert 0 < explained_variance < 1
    def _normalize(array):
        # Normalize sum of array to 1.
        # We assume non-negative entries with non-zero sum.
        return array / array.sum()
    if verbose:
        print(f'seed = {seed}')
    assert norm in ['none', 'l2', 'l1', None]
    data1 = np.vstack([q, p])
    if norm in ['l2', 'l1']:
        data1 = normalize(data1, norm=norm, axis=1)
    pca = PCA(n_components=None, whiten=whiten, random_state=seed+1)
    if pca_max_data < 0 or pca_max_data >= data1.shape[0]:
        pca.fit(data1)
    elif 0 < pca_max_data < data1.shape[0]:
        rng = np.random.RandomState(seed+5)
        idxs = rng.choice(data1.shape[0], size=pca_max_data, replace=False)
        pca.fit(data1[idxs])
    else:
        raise ValueError(f'Invalid argument pca_max_data={pca_max_data} with {data1.shape[0]} datapoints')
    s = np.cumsum(pca.explained_variance_ratio_)
    idx = np.argmax(s >= explained_variance)  # last index to consider
    if verbose:
        print(f'performing clustering in lower dimension = {idx}')
    data1 = pca.transform(data1)[:, :idx+1]
    # Cluster features and obtain the labels for each data point.
    data1 = data1.astype(np.float32)  # Faiss requires float32.
    t1 = time.time()
    
    kmeans = faiss.Kmeans(data1.shape[1], num_clusters, niter=max_iter,
                          verbose=verbose, nredo=num_redo, update_index=True,
                          seed=seed)
    kmeans.train(data1)
    _, labels = kmeans.index.search(data1, 1)
    labels = labels.reshape(-1)
    
    '''# 合并数据和权重

    # 确保q_weight和p_weight是NumPy数组
    if isinstance(q_weight, torch.Tensor):
        q_weight = q_weight.detach().cpu().numpy()

    if isinstance(p_weight, torch.Tensor):
        p_weight = p_weight.detach().cpu().numpy()
    
    # 计算偏移量
    # q_offset = np.abs(np.min(q_weight)) + 1e-5
    # p_offset = np.abs(np.min(p_weight)) + 1e-5

    # 平移权重至正区间
    # q_weight = q_weight + q_offset
    # p_weight = p_weight + p_offset

    # 合并权重
    sample_weights = np.concatenate([q_weight, p_weight])
    
    # 放大权重差异（示例：立方放大）
    # sample_weights = np.power(sample_weights, 3)

    # 归一化权重总和
    # sample_weights = sample_weights / np.sum(sample_weights) * len(sample_weights)
    
    # 带权训练
    kmeans = KMeans(
        n_clusters=num_clusters, 
        max_iter=max_iter, 
        n_init=num_redo, 
        random_state=seed,
        verbose=False
    )
    kmeans.fit(data1, sample_weight=sample_weights)  # 关键权重参数
    labels = kmeans.labels_'''
    
    t2 = time.time()
    if verbose:
        print('kmeans time:', round(t2-t1, 2), 's')

    q_labels = labels[:len(q)]
    p_labels = labels[len(q):]

    # Convert cluster labels to histograms.
    q_bin_counts = np.histogram(
        q_labels, bins=num_clusters,
        range=[0, num_clusters], density=True,
        weights=q_weight
    )[0]
    p_bin_counts = np.histogram(
        p_labels, bins=num_clusters,
        range=[0, num_clusters], density=True,
        weights=p_weight   
    )[0]
    # Histograms without smoothing (used for the original MAUVE).
    p_hist = _normalize(p_bin_counts)
    q_hist = _normalize(q_bin_counts)
    # Histograms with Krichevsky-Trofimov smoothing.
    # Used for MAUVE* suggested by by Pillutla et al. (JMLR 2023).
    # Take the KT coefficient as 1e-5 instead of 0.5 when the distribution is weighed.
    p_hist_smoothed = _normalize(p_bin_counts + 1e-5)
    q_hist_smoothed = _normalize(q_bin_counts + 1e-5)
    return p_hist, q_hist, p_hist_smoothed, q_hist_smoothed

def kl_multinomial(p, q):
    assert p.shape == q.shape
    if np.logical_and(p != 0, q == 0).any():
        return np.inf
    else:
        idxs = np.logical_and(p != 0, q != 0)
        return np.sum(p[idxs] * np.log(p[idxs] / q[idxs]))

def get_divergence_curve_for_multinomials(p, q, mixture_weights, scaling_factor):
    divergence_curve = [[0, np.inf]] # extreme point
    for w in np.sort(mixture_weights):
        r = w * p + (1 - w) * q
        divergence_curve.append([kl_multinomial(q, r), kl_multinomial(p, r)])
    divergence_curve.append([np.inf, 0]) # other extreme point
    return np.exp(-scaling_factor * np.asarray(divergence_curve))

def get_fronter_integral(p, q, scaling_factor=2):
    total = 0.0
    for p1, q1 in zip(p, q):
        if p1 == 0 and q1 == 0:
            pass
        elif p1 == 0:
            total += q1 / 4
        elif q1 == 0:
            total += p1 / 4
        elif abs(p1 - q1) > 1e-8:
            t1 = p1 + q1
            t2 = p1 * q1 * (math.log(p1) - math.log(q1)) / (p1 - q1)
            total += 0.25 * t1 - 0.5 * t2
        # else: contribution is 0 
    return total * scaling_factor

def featurize_sequences_from_sentence_transformer(encoder, sequences, batch_size, name="", verbose=False):
    """Featurize string texts using SentenceTransformer, support batchify
    :param model: HF Transformers model
    :param batch_size: Batch size used during forward pass
    :param tokenized_texts: list of string sequences
    :param verbose: If True, print status and time
    :return:
    """
    t1 = time.time()
    
    feats = []
    for start in tqdm(range(0, len(sequences), batch_size), desc=name):
        batch = sequences[start:min(start+batch_size, len(sequences))]
        embeddings = encoder.encode(batch, convert_to_tensor=True,).detach().cpu()
        feats.append(embeddings)
    
    t2 = time.time()
    if verbose:
        print(f'Featurize time: {round(t2-t1, 2)}')
    
    return torch.cat(feats)

def tokenize_and_truncate(tokenizer, sequences, cutoff_len):
    """
    将序列分词并根据给定的cutoff_len进行截断，然后恢复为字符串。
    
    参数：
        sequences (List[str]): 要分词的输入序列数据集（列表）。
        tokenizer (AutoTokenizer): 使用的tokenizer。
        cutoff_len (int): 截断的长度（以token为单位）。
        
    返回：
        str: 截断后的文本（token列表恢复为字符串）。
    """
    truncated_sequences = []
    
    # 将序列分词并应用截断
    for sequence in sequences:
        tokens = tokenizer.encode(sequence, truncation=True, padding=False, max_length=cutoff_len, return_tensors="pt").detach()        
        # 将截断后的tokens恢复为字符串
        truncated_sequence = tokenizer.decode(tokens[0], skip_special_tokens=True)
        truncated_sequences.append(truncated_sequence)
    
    return truncated_sequences

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# Step 0.0 Set hyper-parameters
device = 'cuda'
model_path = "./pretrained/gpt2"
encoder_path = "./pretrained/gtr-t5-base"
embed_method = "gtr" # [gpt2, gtr]
block_size = 512 # The max length of gtr-t5-base
use_dataset_cache=True  
random_seed = 42 # 42, 26, 62
# Try my best to eliminate randomness
# Python内置random模块
random.seed(random_seed)
# NumPy
np.random.seed(random_seed)
# PyTorch (CPU和GPU)
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)

# Experiment shows that use gtr is more distinguashable.
# dataset
# reference_path = './dataset/alpaca-med-flashcards/split10k-0/original/'
# reference_path = './dataset/alpaca-med-wikidoc/all/original/' # './dataset/alpaca-med-wikidoc/all/paraphrase/deepseek/original/' 
# reference_path = './dataset/wikitext-103-raw-v1/split50k-0/original/'
# reference_path = './dataset/wikicorpus/split50k-0/original'
# reference_path = './dataset/cc_news/split50k-1/original'
# reference_path = './dataset/xsum/split50k-0/original'
# reference_path = './dataset/openwebtext-100k/split50k-1/original'
# reference_path = './dataset/alpaca-med-wikidoc/all/paraphrase/deepseek/original/'
reference_path = './dataset/caselaw/arizona/split10k-0/original/' # use split10k-1 for synthetic-caselaw
# reference_path = './dataset/caselaw/delaware/split10k-0/original/'
# reference_path = './dataset/echr/split10k-0/original/'
# reference_path = './dataset/cc_news/2017-02/split20k-0/original/'
# reference_path = './dataset/alpaca-med/wikidoc-flashcards-mix/split10k-alpha0.8/original/dataset.json'
# reference_path = './dataset/cc_news/2018-07/split20k-0/original/'
# reference_path = './dataset/aggregated_paraphrased_dataset.json'
# reference_path = './dataset/aggregated_paraphrased_dataset_caselaw_delaware.json'
# reference_path = './dataset/aggregated_paraphrased_dataset_cc_news_2017_02.json'
# reference_path = "./dataset/pile_law_val/all/"
# reference_path = './dataset/mental_health_chatbot_dataseet/all/original/'
# reference_path = './dataset/cc_news/2017-04/split-3k-0/original/'
# reference_path = './dataset/tweet_sentiment_extraction/split-10k/positive-0dot2'
# reference_path = './dataset/ChatDoctor/chatgpt/all/'
# reference_path = './dataset/ChatDoctor/healthcaremagic/split7k-0/original/'
# reference_path = './dataset/ChatDoctor/icliniq/all/'
# reference_path = './dataset/instructional_code-search-net/python/split7k-0/'
# reference_path = './dataset/ChatDoctor/healthcaremagic/split10k-0/'
# reference_path = './dataset/ChatDoctor/healthcaremagic/split7k-0/paraphrase/deepseek/aggregated_paraphrased_dataset_chatdoctor.json'
# reference_path = './output/mixed_dataset/caselaw/del_75_merged.json'

# prediction_path = './output/generated-data/pythia-2.8b/caselaw-arizona-split10k-0-lora-2k.json'
prediction_path = './output/mixed_dataset/caselaw/side_del_75_merged.json'
# prediction_path = './output/Qwen3-4B-Instruct/caselaw/delaware/paraphrase/side_generated_outputs_v2_2.json'
# prediction_path = './output/Llama-3.1-8b-instruct-bnb-4bit/chatdoctor/paraphrase/side_generated_outputs_v2_2.json'
# prediction_path = './output/OLMo-7b-hf-bnb-8bit-smashed/alpaca-medwiki/paraphrase/side_generated_outputs_v2_2.json'
# prediction_path = './output/Qwen3-4B-Instruct/alpaca-medwiki/paraphrase/side_generated_outputs_v2_2.json'
# prediction_path = './output/OLMo-7b-hf-bnb-8bit-smashed/chatdoctor/paraphrase/side_generated_outputs_v2_2.json'
# prediction_path = './output/Qwen3-4B-Instruct/chatdoctor/paraphrase/side_generated_outputs_v2_2.json'
# prediction_path = './output/pythia-2.8b/chatdoctor/original/side_generated_outputs_v2_2.json'
# prediction_path = './dataset/ChatDoctor/healthcaremagic/split7k-0/paraphrase/deepseek/aggregated_paraphrased_dataset_chatdoctor.json'
# prediction_path = './output/Qwen3-4B-Instruct/alpaca-medwiki/original/side_generated_outputs_v2_2.json'
# prediction_path = './output/Qwen3-4B-Instruct/alpaca-medwiki/paraphrase/side_generated_outputs_v2_2.json'
# prediction_path = "./output/generated-data/pythia-2.8b/alpaca-medwiki-paraphrased-lora-v2_2.json"
# prediction_path = './output/param-utility-privacy/side_generated_outputs_v2_2.json'
# prediction_path = './dataset/instructional_code-search-net/java/split7k-0/'
# prediction_path = './dataset/ChatDoctor/healthcaremagic/split7k-0/original/'
# prediction_path = './dataset/ChatDoctor/chatgpt/all/'
# prediction_path = './dataset/ChatDoctor/icliniq/all/'
# prediction_path = './dataset/ChatDoctor/split10k-0/original/'
# prediction_path = './dataset/tweet_sentiment_extraction/split-10k/positive-0dot8'
# prediction_path = './output/generated-data/qwen3-4b/caselaw-delaware-split10k-0-topk-30.json'
# prediction_path = './tinyllama-mental-health-finetuned-2k.json'
# prediction_path = "./law-llm-outputs-2k.json"
# prediction_path = "./output/generated-data/olmo2-1b/caselaw-delaware-split10k-0-topk-30.json"
# prediction_path = "./output/generated-data/olmo2-1b/alpaca-medwiki-all-topk-30.json"
# prediction_path = "./output/generated-data/olmo2-1b/caselaw-delaware-split10k-0-topk-30.json"
# prediction_path = './output/imitation-test/caselaw/delaware/split10k-0/generate/model_imitation_pythia_lora_topk30/generated_datasets.json'
# prediction_path = './dataset/aggregated_paraphrased_dataset.json'
# prediction_path = './dataset/aggregated_paraphrased_dataset_caselaw_delaware.json'
# prediction_path = './dataset/aggregated_paraphrased_dataset_cc_news_2017_02.json'
# prediction_path = './dataset/caselaw/delaware/split10k-0/original/'
# prediction_path = './dataset/caselaw/arizona/split10k-0/original/'
# prediction_path = './dataset/echr/split10k-0/original/'
# prediction_path = './dataset/alpaca-med-wikidoc/all/paraphrase/deepseek/original/'
# prediction_path = './dataset/cc_news/split50k-0/original'
# prediction_path = './dataset/cc_news/2017-02/split20k-1/original/'
# prediction_path = './dataset/cc_news/2017-02/split20k-1/original/'
# prediction_path = './dataset/cc_news/2018-2017-mix/split20k-alpha0.5/original/dataset.json'
# prediction_path = './dataset/cc_news/2018-07/split20k-0/original/'
# prediction_path = './dataset/alpaca-med/wikidoc-flashcards-mix/split10k-alpha0.8/original/dataset.json'
# prediction_path = "./medwiki-flashcards-mix-split10k-alpha0.8-2k.json"
# prediction_path = './dataset/xsum/split50k-1/original'
# prediction_path = './dataset/aggregated_paraphrased_dataset.json'
# prediction_path = './dataset/alpaca-med-flashcards/split10k-0/original/'
# prediction_path = './dataset/wikicorpus/split50k-0/original'
# prediction_path = './dataset/wikitext-103-raw-v1/split50k-0/original/'
# prediction_path = './dataset/openwebtext-100k/split50k-0/original'
# prediction_path = './alpaca-medwiki-2k.json' 
# prediction_path = './alpaca-medflash-2k.json'
# prediction_path = './dataset/alpaca-med-flashcards/all/original/'
# prediction_path = './alpaca-medwiki-2k-dpsgd-e8-d1e-4.json'
# prediction_path = './alpaca-medwiki-2k-deepseek-paraphrased.json'
# prediction_path = './output/original/pythia-2.8b-alpaca-finance-prompt-tuning/generated_conjunctive_prompt.json'
# prediction_path = './openwebtext-split50k-0-2k.json'
# prediction_path = './wikitext-split50k-0-2k.json'
# prediction_path = './cc-news-split50k-0-2k.json'
# prediction_path = './xsum-split50k-0-2k.json'
# prediction_path = './wikitext-split50k-0-dpsgd-e8d2e-5n1-2k.json'
# prediction_path = './caselaw-arizona-split10k-0-dp-prompt-2k.json'
# prediction_path = './wikitext-split50k-0-dp-prompt-2k.json'
# prediction_path = './xsum-split50k-0-dp-prompt-2k.json'
# prediction_path = './alpaca-medwiki-dp-prompt-2k.json'
# prediction_path = './xsum-split50k-0-dpsgd-e8d2e-5n1-2k.json'
# prediction_path = './caselaw-arizona-split10k-0-2k.json'
# prediction_path = './echr-split10k-0-2k.json'
# prediction_path = './caselaw-arizona-split10k-0-dpsgd-e8d1e-4n1-2k.json'
# prediction_path = './xsum-split50k-0-lora-2k.json'
# prediction_path = './wikitext-split50k-0-lora-2k.json'
# prediction_path = './caselaw-arizona-split10k-0-lora-2k.json'
# prediction_path = './alpaca-medwiki-2k-lora.json'
# prediction_path = './alpaca-medflash-2k-lora.json'
# prediction_path = './echr-split10k-0-lora-2k.json'
# prediction_path = './openwebtext-split50k-0-lora-2k.json'
# prediction_path = './cc-news-split50k-0-lora-2k.json'
# prediction_path = './caselaw-arizona-split10k-0-lora-10k.json' # not used, use the following one
# prediction_path = './wikitext-split50k-0-lora-10k.json'
# prediction_path = './xsum-split50k-0-lora-10k.json'
# prediction_path = './cc-news-2017-2-split20k-0-2k.json'
# prediction_path = './cc-news-2017-2-split20k-0-lora-2k.json'
# prediction_path = './caselaw-arizona-split10k-0-generated-2k.json'
# prediction_path = './dataset/cc_news/2017-02/split20k-0/generate/original/cc-news-2017-2-split20k-0-lora-2k-top-k-5.json'
# prediction_path = './dataset/cc_news/2017-02/split20k-0/generate/original/cc-news-2017-2-split20k-0-lora-20k-top-k-15.json'
# prediction_path = './dataset/cc_news/2017-02/split20k-0/generate/original/cc-news-2017-2-split20k-0-lora-20k-top-k-30.json'
# prediction_path = './dataset/cc_news/2017-02/split20k-0/generate/original/cc-news-2017-2-split20k-0-lora-20k-top-k-50.json'
# prediction_path = './dataset/cc_news/2017-02/split20k-0/generate/original/cc-news-2017-2-split20k-0-lora-2k-top-k-70.json'
# prediction_path = './cc-news-2017-2-split20k-0-dpsgd-e8d5e-4n1-2k.json'
# prediction_path = './cc-news-2018-7-split20k-0-lora-2k.json'
# prediction_path = './caselaw-delaware-split10k-0-lora-2k.json'
# prediction_path = './caselaw-delaware-split10k-0-2k.json'
# prediction_path = './caselaw-delaware-split10k-0-2k-top-k-30.json'
# prediction_path = './dataset/alpaca-med-wikidoc/all/generate/original/alpaca-medwiki-lora-10k-top-k-15.json'
# prediction_path = './dataset/alpaca-med-wikidoc/all/generate/original/alpaca-medwiki-lora-10k-top-k-30.json'
# prediction_path = './dataset/alpaca-med-wikidoc/all/generate/original/alpaca-medwiki-lora-10k-top-k-50.json'
# prediction_path = './cc-news-2018-7-split20k-0-lora-2k-top-k-30.json'
# prediction_path = './cc-news-2017-2-split20k-0-dpsgd-e8d5e-5n1-2k-top-k-30.json'
# prediction_path = './dataset/caselaw/arizona/split10k-0/generate/original/caselaw-arizona-split10k-0-lora-10k-top-k-15.json'
# prediction_path = './dataset/caselaw/arizona/split10k-0/generate/original/caselaw-arizona-split10k-0-lora-10k-top-k-50.json'
# prediction_path = './dataset/caselaw/arizona/split10k-0/generate/original/caselaw-arizona-split10k-0-lora-2k-top-k-5.json'
# prediction_path = './dataset/caselaw/arizona/split10k-0/generate/original/caselaw-arizona-split10k-0-lora-10k-top-k-30.json'
# prediction_path = './dataset/caselaw/arizona/split10k-0/generate/original/caselaw-arizona-split10k-0-lora-2k-top-k-70.json'
# prediction_path = './dataset/alpaca-med-wikidoc/all/generate/synthetic/alpaca-medwiki-paraphrased/synthetic-dataset-topk-30.json'
# prediction_path = './dataset/alpaca-med-wikidoc/all/generate/synthetic/alpaca-medwiki/synthetic-dataset-topk-30.json'
# prediction_path = './dataset/alpaca-med-wikidoc/all/generate/synthetic/alpaca-medflash-split10k-0/synthetic-dataset-topk-30.json'
# prediction_path = './dataset/cc_news/2017-02/split20k-0/generate/synthetic/cc-news-2018-07-split20k-0/synthetic-dataset-topk-30.json'
# prediction_path = './dataset/caselaw/arizona/split10k-0/generate/synthetic/caselaw-delaware-split10k-0/synthetic-dataset-topk-30.json'
# prediction_path = './synthetic-alpaca-medflash-lora-epoch5-topk-30.json'
# prediction_path = './synthetic-alpaca-medflash-lora-epoch3-topk-30.json'
# prediction_path = './synthetic-alpaca-medflash-prompt-tuning-epoch5-topk-30.json'
# prediction_path = './synthetic-alpaca-medflash-prompt-tuning-epoch3-topk-30.json'
# prediction_path = './alpaca-medwiki-paraphrased-lora-topk-30.json'
# prediction_path = './alpaca-medwiki-paraphrased-prompt-tuning-topk-30.json'
# prediction_path = './dataset/cc_news/2017-02/split20k-0/generate/synthetic/cc-news-2018-07-all/synthetic-dataset-topk-30.json'
# prediction_path = './dataset/caselaw/arizona/split10k-0/generate/synthetic/caselaw-delaware-all/synthetic-dataset-topk-30.json'
# prediction_path = './synthetic-caselaw-delaware-split10k-0-lora-topk-30.json'
# prediction_path = './synthetic-cc-news-2018-07-split20k-0-lora-topk-30.json'
# prediction_path = './output/generated-data/pythia-2.8b/synthetic-caselaw-arizona-split10k-0-lora-topk-30-2k.json'
# prediction_path = './dataset/caselaw/delaware/split10k-0/generate/synthetic/caselaw-arizona-split10k-0/synthetic-dataset-topk-30.json'
# prediction_path = './dataset/caselaw/delaware/split10k-0/generate/original/caselaw-delaware-split10k-0-lora-10k-top-k-30.json'
# prediction_path = './output/generated-data/pythia-2.8b/caselaw-delaware-split10k-0-dpsgd-e8d1e-4n1-2k-top-k-30.json'
# prediction_path = './output/generated-data/pythia-2.8b/caselaw-delaware-split10k-0-dpsgd-e8d1e-4n1-2k-top-k-30-epochs5.json'
# prediction_path = './tmp.json'
# prediction_path = './output/generated-data/qwen-1.5b/caselaw-delaware-split10k-0-2k-top-k-30.json'
# prediction_path = './output/generated-data/pythia-2.8b/caselaw-delaware-split10k-0-2k-top-k-15.json'
# prediction_path = './output/generated-data/qwen-1.5b/caselaw-delaware-split10k-0-2k-top-k-30.json'
# prediction_path = './output/generated-data/qwen-1.5b/alpaca-medwiki-all-top-k-15.json'
# prediction_path = './output/generated-data/qwen-1.5b/cc-news-2017-02-split20k-0-2k-top-k-15.json'
# prediction_path = './output/generated-data/qwen-1.5b/cc-news-2017-02-split20k-0-2k-top-k-15.json'
# prediction_path = './output/generated-data/llama-3.1-8b-bnb-4bit/caselaw-delaware-split10k-0-2k.json'
# prediction_path = './output/generated-data/llama-3.1-8b-bnb-4bit/cc-news-2017-02-split20k-0-2k.json'
# prediction_path = './output/generated-data/llama-3.1-8b-bnb-4bit/alpaca-med-wikidoc-all-0-500-3.json'
# prediction_path = './output/generated-data/qwen-2.5-7b-bnb-4bit/alpaca-medwiki-all-2k-top-k-30.json'
# prediction_path = './output/generated-data/qwen-2.5-7b-bnb-4bit/caselaw-delaware-split10k-0-2k-top-k-30.json'
# prediction_path = './dataset/caselaw/delaware/split10k-0/generated/qwen-2.5-7b-bnb-4bit/caselaw-delaware-split10k-0-10k-lora-top-k-30.json'
# prediction_path = './output/generated-data/qwen-2.5-7b-bnb-4bit/cc-news-2017-02-split20k-0-2k-top-k-30.json'
# prediction_path = './output/generated-data/qwen-3b/caselaw-delaware-split10k-0-2k-top-k-30.json'
# prediction_path = './output/generated-data/qwen-3b/caselaw-delaware-split10k-0-2k.json'
# prediction_path = './output/generated-data/qwen-3b/alpaca-medwiki-all-2k-top-k-30.json'
# prediction_path = './output/generated-data/qwen-3b/cc-news-2017-02-split20k-0-2k-top-k-30.json'
# prediction_path = './output/generated-data/llama-3.1-8b-bnb-4bit/alpaca-med-wikidoc-all-0-2k.json'
# prediction_path = './output/generated-data/llama-3.1-8b-bnb-4bit/alpaca-med-wikidoc-all-0-500-v2.json'
# prediction_path = './output/generated-data/llama-3.1-8b-bnb-4bit/alpaca-med-wikidoc-all-0-2k-lora.json'
# prediction_path = './output/generated-data/qwen-3b/caselaw-delaware-split10k-0-2k-epochs5.json'
# prediction_path = './output/generated-data/qwen-3b/caselaw-delaware-split10k-0-2k-lora.json'
# prediction_path = './dataset/caselaw/delaware/split10k-0/generate/original/llama-8b/caselaw-delaware-split10k-0-lora-10k-top-k-30.json'
# prediction_path = './dataset/cc_news/2017-02/split20k-0/generate/original/llama-8b/cc-news-2017-02-split20k-0-lora-20k-top-k-30.json'
# prediction_path = './dataset/alpaca-med-wikidoc/all/generate/original/llama-8b/alpaca-medwiki-lora-10k-top-k-30.json'
# prediction_path = './dataset/alpaca-med-wikidoc/all/generate/original/qwen-3b/alpaca-medwiki-lora-10k-top-k-30.json'
# prediction_path = './dataset/caselaw/delaware/split10k-0/generate/original/qwen-3b/caselaw-delaware-split10k-0-lora-10k-top-k-30.json'
# prediction_path = './dataset/cc_news/2017-02/split20k-0/generate/original/qwen-3b/cc-news-2017-02-split20k-0-lora-20k-top-k-30.json'
# prediction_path = './output/generated-data/llama-3.1-8b-bnb-4bit/alpaca-med-wikidoc-all-2k-epochs5.json'
# prediction_path = './dataset/caselaw/delaware/split10k-0/generate/original/qwen-3b/caselaw-delaware-split10k-0-lora-10k-top-k-30-epochs5.json'
# prediction_path = './output/generated-data/pythia-2.8b/alpaca-medwiki-paraphrase-lora-2k.json'
# prediction_path = './output/generated-data/pythia-2.8b/caselaw-delaware-split10k-0-paraphrase-lora-epoch10-2k.json'
# prediction_path = './output/generated-data/pythia-2.8b/cc-news-2017-2-split20k-0-paraphrase-lora-2k.json'
# prediction_path = './output/generated-data/pythia-2.8b/alpaca-medwiki-paraphrased-lora-topk-30-new.json'
# prediction_path = './output/generated-data/pythia-2.8b/alpaca-medwiki-paraphrased-lora-topk-30.json'
# prediction_path = './output/generated-data/pythia-2.8b/alpaca-medwiki-paraphrase-lora-2k.json'
# prediction_path = './output/generated-data/pythia-2.8b/caselaw-delaware-split10k-0-epoch5.json'
# prediction_path = './output/generated-data/pythia-2.8b/caselaw-delaware-split10k-0-epoch10.json'
# prediction_path = './dataset/caselaw/delaware/split10k-0/generate/original/caselaw-arizona-split10k-0-lora-2k-top-k-70.json'
# prediction_path = './dataset/caselaw/delaware/split10k-0/generate/synthetic/caselaw-delaware-split10k-1/synthetic-dataset-topk-30.json'
# prediction_path = './dataset/caselaw/delaware/split10k-0/generate/synthetic/caselaw-delaware-paraphrased/synthetic-dataset-topk-30.json'
# prediction_path = './output/generated-data/pythia-2.8b/caselaw-delaware-split10k-0-dpsgd-e8d1e-4n1-2k-top-k-5.json'
# prediction_path = './dataset/cc_news/2017-02/split20k-0/generate/synthetic/wikitext-103-2k/synthetic-dataset-topk-30.json'
# prediction_path = './dataset/alpaca-med-wikidoc/all/generate/synthetic/alpaca-cleaned-2k/synthetic-dataset-topk-30.json'
# prediction_path = './dataset/cc_news/2017-02/split20k-0/generate/synthetic/wikitext-103-2k/synthetic-dataset-topk-30.json'
# prediction_path = './dataset/caselaw/delaware/split10k-0/generate/synthetic/wikitext-103-2k/synthetic-dataset-topk-30.json'
# prediction_path = './output/generated-data/pythia-2.8b/caselaw-delaware-all-dpsgd-e8d1e-4n1-2k-top-k-30.json'
# prediction_path = './output/generated-data/pythia-2.8b/caselaw-delaware-all-dpsgd-e8d1e-4n1-2k-top-k-1.json'
# prediction_path = './output/generated-data/pythia-2.8b/caselaw-delaware-all-dpsgd-e8d1e-4n1-2k-top-k-5.json'
# prediction_path = './output/generated-data/pythia-2.8b/caselaw-delaware-split10k-0-dpsgd-e8d1e-4n1-2k-top-k-30.json'
# prediction_path = './output/generated-data/pythia-2.8b/part-alpaca-med-wikidoc-all-2k.json'
# prediction_path = './output/generated-data/pythia-2.8b/part-synthetic-caselaw-delaware-split10k-0-2k.json'
# prediction_path = './output/generated-data/pythia-2.8b/part-caselaw-delaware-split10k-0-2k.json'
# prediction_path = './output/generated-data/pythia-2.8b/part-synthetic-cc-news-2017-02-split20k-0-2k.json'
# prediction_path = './output/generated-data/pythia-2.8b/part-cc-news-2017-02-split20k-0-2k.json'
# prediction_path = './output/generated-data/pythia-2.8b/part-synthetic-cc-news-2017-02-split20k-0-2k.json'
# prediction_path = './dataset/caselaw/delaware/split10k-0/generate/synthetic/caselaw-arizona-split10k-0/synthetic-dataset-topk-30-v2.json'
# prediction_path = './dataset/caselaw/delaware/split10k-0/generate/synthetic/caselaw-delaware-paraphrased/synthetic-dataset-topk-30-v2.json'
# prediction_path = './dataset/caselaw/delaware/split10k-0/generate/synthetic/wikitext-103-2k/synthetic-dataset-topk-30-v2.json'
# prediction_path = './dataset/cc_news/2017-02/split20k-0/generate/synthetic/cc-news-2018-07-split20k-0/synthetic-dataset-topk-30-v2.json'
# prediction_path = './dataset/cc_news/2017-02/split20k-0/generate/synthetic/cc-news-2017-02-paraphrased/synthetic-dataset-topk-30-v2.json'
# prediction_path = './dataset/cc_news/2017-02/split20k-0/generate/synthetic/wikitext-103-2k/synthetic-dataset-topk-30-v2.json'

# reference_key = 'RESPONSE'
# reference_key = 'input'
# reference_key = 'output'
reference_key = 'text'
# reference_key = 'document'

# prediction_key = 'RESPONSE'
# prediction_key = 'input'
prediction_key = 'text'
# prediction_key = 'output'
# prediction_key = 'generated_output'
# prediction_key = 'document'

reference_data_key = 'train'
# reference_data_key = 'validation'
prediction_data_key = 'train'

freq_minimum = 1e-8
dataset_size = 2000
# parapeters for k-means clustering
num_buckets = 'auto'
pca_max_data = -1
kmeans_explained_var = 0.9
kmeans_num_redo = 5 
kmeans_max_iter = 500
# parapeters for computing mauve
divergence_curve_discretization_size = 25
if embed_method == 'gtr':
    mauve_scaling_factor = 2 # 5 for gpt, 3 for gtr
elif embed_method == 'gpt2':
    mauve_scaling_factor = 5 # 5 for gpt, 3 for gtr
else:
    raise ValueError(f'embed_method in [\'gtr\', \'gpt2\'], but get {embed_method}')
# p-value threshold for hyperthesis test
if embed_method == 'gtr':
    p_threshold = 0.45 # 0.45 for gpt, 0.5 for gtr
elif embed_method == 'gpt2':
    p_threshold = 0.5 #  0.45 for gpt, 0.5 for gtr
else:
    raise ValueError(f'embed_method in [\'gtr\', \'gpt2\'], but get {embed_method}')

# min_length_coefficient = 2

print("="*10)
print("Step 0.0: Set hyper-parameters: \n"
      f"[device]: {device}\n"
      f"[model_path]: {model_path}\n"
      f"[encoder_path]: {encoder_path}\n"
      f"[embed_method]: {embed_method}\n"
      f"[block_size]: {block_size}\n"
      f"[reference_path]: {reference_path}\n"
      f"[prediction_path]: {prediction_path}\n"
      f"[reference_key]: {reference_key}\n"
      f"[prediction_key]: {prediction_key}\n"
      f"[freq_minimum]: {freq_minimum}\n"
      f"[dataset_size]: {dataset_size}\n"
      f"[random_seed]: {random_seed}\n"
      f"[num_buckets]: {num_buckets}\n"
      f"[pca_max_data]: {pca_max_data}\n"
      f"[kmeans_explained_var]: {kmeans_explained_var}\n"
      f"[kmeans_num_redo]: {kmeans_num_redo}\n"
      f"[kmeans_max_iter]: {kmeans_max_iter}\n"
      f"[divergence_curve_discretization_size]: {divergence_curve_discretization_size}\n"
      f"[mauve_scaling_factor](original): {mauve_scaling_factor}\n"
      # f"[min_length_coefficient]:{min_length_coefficient}"
      )

# Step 0.1 Get model
print("-"*10)
print(f"Step 0.1: Get SentenceTransformer from {encoder_path}.")
encoder = SentenceTransformer(encoder_path)
if embed_method == 'gtr':
    tokenizer = AutoTokenizer.from_pretrained(encoder_path)
print()

# Step 0.2 Get prediction and reference datasets
print("-"*10)
print(f"Step 0.2: Get prediction and reference datasets: \n"
      f"[reference data path]: {reference_path}\n"
      f"[prediction data path]: {prediction_path}"
      )
reference_data = get_dataset(reference_path, data_key=reference_data_key, seed=random_seed)
prediction_data = get_dataset(prediction_path, data_key=prediction_data_key, seed=random_seed)


# Step 0.3 Get the average word length of each dataset
print("-"*10)
print("Step 0.3: Pack the key column in datasets")

reference_data = pack_dataset(reference_data, reference_path, column_name=reference_key, use_dataset_cache=use_dataset_cache, 
                              block_size=block_size, num_of_sequences=1024, chars_per_token=3.6)
# prediction_data_no_packing = prediction_data[prediction_key]
prediction_data = pack_dataset(prediction_data, prediction_path, column_name=prediction_key, use_dataset_cache=use_dataset_cache, 
                               block_size=block_size, num_of_sequences=1024, chars_per_token=3.6)
# In case the prediciton samples are too few after packing.
# if len(prediction_data) < 0.75 * len(reference_data):
#     prediction_data = prediction_data_no_packing
# reference_data = reference_data[prediction_key]
# prediction_data = prediction_data[prediction_key]

print(
        "\n If there's a warning about too long sequence above, please disregard, because the tokenized sequences will be packed before being fed to model :)"
    )
                               
if dataset_size == 0:
    print("dataset_size is set to 0, take all samples.")
else:
    # data is previously shuffled in get_dataset
    # reference_data = reference_data[:dataset_size]
    # prediction_data = prediction_data[:dataset_size]
    # data_size = min(min(dataset_size, len(reference_data)), len(prediction_data))
    reference_data = random.sample(reference_data, min(dataset_size, len(reference_data)))
    prediction_data = random.sample(prediction_data, min(dataset_size, len(prediction_data)))
    
print(f"[len(reference_data) with empty entries filtered]: {len(reference_data)}")
print(f"[len(prediction_data) with empty entries filtered]: {len(prediction_data)}")
print()

# Step 0.4 Get the word frequency of each dataset
### TODO: enhance current method with PII detection
print("-"*10)
print("Step 0.4: Get the word frequency of each dataset")
reference_freq, reference_freq_min = compute_dataset_word_frequency(reference_data, tokenizer)
prediction_freq, prediction_freq_min = compute_dataset_word_frequency(prediction_data, tokenizer)
print()

print("-"*10)
print("Step 0.5: Compute reference and prediction dataset relevance")
reference_word_relevance_dict = compute_dataset_relevance(reference_freq.keys(), reference_freq, freq_minimum=reference_freq_min)
prediction_word_relevance_dict = compute_dataset_relevance(prediction_freq.keys(), prediction_freq, freq_minimum=prediction_freq_min)
# print([(word, relev) for word, relev in reference_word_relevance_dict.items() if len(word) == 1])
# assert 0

print("max ref and min ref: ", max(reference_word_relevance_dict.values()), min(reference_word_relevance_dict.values()))
print("max pred and min pred: ", max(prediction_word_relevance_dict.values()), min(prediction_word_relevance_dict.values()))

'''sorted_reference_word_relevance_dict = dict(sorted(reference_word_relevance_dict.items(), key=lambda x: x[1], reverse=True))
sorted_prediction_word_relevance_dict = dict(sorted(prediction_word_relevance_dict.items(), key=lambda x: x[1], reverse=True))
# 获取最大值的前 n 个元素
n = 50
top_n_reference = list(sorted_reference_word_relevance_dict.items())[:n]
top_n_prediction = list(sorted_prediction_word_relevance_dict.items())[:n]

# 打印出最大值的前 n 个元素
print(f"Top {n} Reference Word relevance:")
for key, value in top_n_reference:
    print(f"{key}: {value}, p={reference_freq[key]}, q={word_frequency(key, 'en', minimum=reference_freq_min)}")
print()
print(f"\nTop {n} Prediction Word relevance:")
for key, value in top_n_prediction:
    print(f"{key}: {value}, p={prediction_freq[key]}, q={word_frequency(key, 'en', minimum=prediction_freq_min)}")
print()
# assert 0
# 获取最小值的前 n 个元素
n = 10
bottom_n_reference = list(sorted_reference_word_relevance_dict.items())[-n:]
bottom_n_prediction = list(sorted_prediction_word_relevance_dict.items())[-n:]

print(f"Bottom {n} Reference Word relevance:")
for key, value in bottom_n_reference:
    print(f"{key}: {value}, p={reference_freq[key]}, q={word_frequency(key, 'en', minimum=reference_freq_min)}")
print()
print(f"\Bottom {n} Prediction Word relevance:")
for key, value in bottom_n_prediction:
    print(f"{key}: {value}, p={prediction_freq[key]}, q={word_frequency(key, 'en', minimum=prediction_freq_min)}")
print()

import matplotlib.pyplot as plt

# 数据准备
reference = list(reversed(list(filter((lambda t: len(re.sub(r'\d', '', t[0])) > 1), list(top_n_reference)))))
prediction = list(reversed(list(filter((lambda t: len(re.sub(r'\d', '', t[0])) > 1), list(top_n_prediction)))))

print(reference)
print()
print(prediction)

import matplotlib.pyplot as plt

plt.rcParams.update({
    # 'figure.facecolor': 'white',
    # 'axes.facecolor': 'white',
    # 'grid.color': '#e0e0e0',
    # 'grid.linestyle': '--',
    # 'grid.linewidth': 0.8,
    # 'axes.edgecolor': 'black',
    # 'axes.linewidth': 1.2,
    'font.family': 'sans-serif',
    'pdf.fonttype': 42,                # 强制输出Type 42（TrueType）字体
    'ps.fonttype': 42,                 # PostScript输出同理
    # 'font.size': 14,  # 增大基础字号
    # 'axes.titlesize': 16,
    # 'axes.labelsize': 14,
    # 'ytick.labelsize': 13,  # 专门设置y轴字号
    # 'xtick.major.size': 4,
    # 'ytick.major.size': 4
})

# 调整画布尺寸（增加高度）
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 8),  # 高度从7增加到10
                              gridspec_kw={'wspace':0.3})  # 增加子图间距
n = 10
# 绘制Reference组（缩小条形高度）
colors = [# (229/255, 133/255, 93/255), 
          (238/255, 191/255, 109/255),
          (68/255, 117/255, 122/255), ]
ref_bars = ax1.barh([w[0] for w in reference[-n:]], 
                  [w[1] for w in reference[-n:]],
                  color=(238/255, 191/255, 109/255), 
                  edgecolor='white',
                  height=0.8,  # 原0.8改为0.5
                  linewidth=1)
ax1.set_title(f'Top Relevant Words in CaseLaw Delaware', fontsize=32) # , fontweight='bold')
ax1.set_xlabel('Relevance Score', fontsize=30) # , fontweight='bold')

# 绘制Prediction组
pred_bars = ax2.barh([w[0] for w in prediction[-n:]], 
                   [w[1] for w in prediction[-n:]],
                   color=(68/255, 117/255, 122/255),
                   edgecolor='white',
                   height=0.8,  # 原0.8改为0.5
                   linewidth=1)
ax2.set_title(f'Top Relevant Words in CaseLaw Arizona', fontsize=32,) # fontweight='bold')
ax2.set_xlabel('Relevance Score', fontsize=30) # , fontweight='bold')

# 统一设置坐标轴样式
for ax in [ax1, ax2]:
    ax.grid(True, axis='x', alpha=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    
    # 增大y轴标签与条形的间距
    ax.tick_params(axis='y', which='major', pad=8)  # 默认是4
    
    # 设置刻度标签的显式字体大小
    ax.tick_params(axis='y', labelsize=30)  # 覆盖全局设置
    ax.tick_params(axis='x', labelsize=26)  # 覆盖全局设置

# 调整布局
plt.subplots_adjust(left=0.15, right=0.9)  # 给y轴标签留出更多空间
# plt.show()
plt.savefig('./images/word_relevance.pdf', 
           bbox_inches='tight', 
           pad_inches=0.3,  # 增加边距
           dpi=300)
plt.close()

assert 0

print()'''

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''


# Step1: Get reference dataset relevance score
print("="*10)
print("Step 1: Get reference dataset relevance score (word wise)")
list1 = compute_normalized_relevance_score(data=reference_data, tokenizer=tokenizer,
                                      relevance_scores=reference_word_relevance_dict, verbose=True)
list2 = compute_normalized_relevance_score(data=prediction_data, tokenizer=tokenizer,
                                      relevance_scores=prediction_word_relevance_dict, verbose=True)
'''max_relevance_score = max(prediction_relevance_score, reference_relevance_score)
min_relevance_score = (min(prediction_relevance_score, reference_relevance_score) + 1e-9)
relative_relevance_score = math.exp(math.sqrt(max_relevance_score) * math.log(max_relevance_score, min_relevance_score))
'''# print(f"[reference dataset relevance score]: {reference_relevance_score}")
# print(f"[prediction dataset relevance score]: {prediction_relevance_score}")
# print(f"[relative relevance score]: {relative_relevance_score}")

# list1 = list(reference_word_relevance_dict.values())
# list2 = list(prediction_word_relevance_dict.values())
"""mean_ref = np.mean(list1)
mean_pred = np.mean(list2)
std_ref = np.std(list1)
std_pred = np.std(list2)
print(f"mean(ref): {mean_ref}; std(ref): {std_ref}")
print(f"mean(pred): {mean_pred}; std(pred): {std_pred}")"""
"""smd = abs(mean_ref - mean_pred) / math.sqrt( (std_ref**2 + std_pred**2) / 2)
smd_c = 3
relative_relevance_score = math.exp(smd_c * smd)
print(f"[relevance SMD score]: {relative_relevance_score}")"""

def kl_divergence_normal(mu_P, sigma_P, mu_Q, sigma_Q):
    """
    计算两个正态分布的KL散度 D(P || Q)
    
    参数:
    - mu_P, sigma_P: 第一个正态分布的均值和标准差
    - mu_Q, sigma_Q: 第二个正态分布的均值和标准差
    
    返回:
    - KL散度的值
    """
    # 计算KL散度
    kl_div = np.log(sigma_Q / sigma_P) + (sigma_P**2 + (mu_P - mu_Q)**2) / (2 * sigma_Q**2) - 0.5
    return kl_div

"""
# Not used
def js_divergence_normal(mu1, sigma1, mu2, sigma2):
    '''
    计算两个正态分布之间的 JS 散度。
    '''
    # 计算中间分布 M 的均值和方差
    mu_M = 0.5 * (mu1 + mu2)
    sigma_M_sq = 0.5 * (sigma1**2 + sigma2**2)
    
    # 计算 KL 散度
    kl_pm = kl_divergence_normal(mu1, sigma1, mu_M, np.sqrt(sigma_M_sq))
    kl_qm = kl_divergence_normal(mu2, sigma2, mu_M, np.sqrt(sigma_M_sq))
    
    # 计算 JS 散度
    js = 0.5 * (kl_pm + kl_qm)
    return js"""

"""
# Not used
nkl = kl_divergence_normal(mean_ref, std_ref, mean_pred, std_pred)
print(f"KL divergence between two normal distributions: {nkl}")
nkl_c = 7
relative_relevance_score = math.exp(nkl_c * nkl)
print(f"[relevance NKL score]: {relative_relevance_score}")
print('---')
njs = math.sqrt(js_divergence_normal(mean_ref, std_ref, mean_pred, std_pred))
print(f"JS divergence between two normal distributions: {njs}")
print('---')"""
"""snkl = nkl ** (1 / 3)
relative_relevance_score = math.exp(snkl)
print(f"Square-root KL divergence between two normal distributions: {snkl}")
print(f"[relevance NKL score]: {relative_relevance_score}")
"""

# njs_c = 7
# relative_relevance_score = math.exp(njs_c * njs)
# print(f"[relevance SMD score]: {relative_relevance_score}")
def mad_std(data, consistency=1.4826):
    """
    中位数绝对偏差法(MAD)[1,6](@ref)
    适用场景：高离群值比例数据（<50%污染）
    """
    med = np.median(data)
    deviations = np.abs(data - med)
    return consistency * np.median(deviations)

def iqr_std(data):
    """
    四分位距法[1,2](@ref)
    适用场景：对称分布数据快速估计
    """
    q75, q25 = np.percentile(data, [75, 25])
    return (q75 - q25) / 1.349

def huber_m_std(data, c=1.345, max_iter=100, tol=1e-5):
    """
    Huber M估计法[1,5](@ref)
    适用场景：动态系统参数估计
    """
    mu = np.median(data)
    std = mad_std(data)
    
    for _ in range(max_iter):
        residuals = (data - mu) / std
        weights = np.where(np.abs(residuals) <= c, 1, c/np.abs(residuals))
        
        new_mu = np.sum(weights * data) / np.sum(weights)
        new_std = np.sqrt(np.sum(weights*(data - new_mu)**2) / np.sum(weights))
        
        if np.abs(new_mu - mu) < tol and np.abs(new_std - std) < tol:
            break
        mu, std = new_mu, new_std
    
    return std

from scipy.stats import norm
def robust_fit_normal_distribution(x_d, data):
    mean = np.median(data)
    # mean = np.mean(data)
    std_robust = iqr_std(data)
    # std_robust = mad_std(data)
    # std_robust = huber_m_std(data)
    pdf_robust = norm.pdf(x_d, mean, std_robust)
    return mean, std_robust, pdf_robust

# 绘制并拟合正态分布
num_samples = 10000
min_x_d = -0.5 # -0.5
max_x_d = 0.5 # 1.0
x_d = np.linspace(min_x_d, max_x_d, num_samples).reshape(-1, 1)

def js_divergence(densities_p, densities_q):
    """
    计算离散概率分布之间的Jensen-Shannon散度
    """
    epsilon = 1e-10
    
    # 混合分布
    m = 0.5 * (densities_p + densities_q)
    
    # 防止零值
    densities_p = np.maximum(densities_p, epsilon)
    densities_q = np.maximum(densities_q, epsilon)
    m = np.maximum(m, epsilon)
    
    # 计算KL散度
    kl_pm = np.sum(densities_p * np.log2(densities_p / m))
    kl_qm = np.sum(densities_q * np.log2(densities_q / m))
    
    # JS散度
    js_div = 0.5 * kl_pm + 0.5 * kl_qm
    return js_div

# 修改后的调用代码
mu1, std1, pdf1 = robust_fit_normal_distribution(x_d, list1)
mu2, std2, pdf2 = robust_fit_normal_distribution(x_d, list2)

# 归一化概率分布
pdf1 = pdf1 / np.sum(pdf1)
pdf2 = pdf2 / np.sum(pdf2)

# 计算JS散度
js_normal = js_divergence(pdf1, pdf2)
print(f"m1, std1 = {mu1}, {std1}")
print(f"m2, std2 = {mu2}, {std2}")
print(f"JS Divergence JS(ref||pred): {js_normal}")
nkl_c = 6 / math.log(2)
relative_relevance_score = math.exp(nkl_c * js_normal)
print(f"[relevance NKL score]: {relative_relevance_score}")
# exit(0)

from matplotlib import pyplot as plt 
def plot_distributions(x_d, pdf1, pdf2, mu1, std1, mu2, std2):
    """
    双分布对比可视化函数
    
    Parameters:
        x_d : 横坐标数组
        pdf1 : 第一个分布的归一化PDF
        pdf2 : 第二个分布的归一化PDF
        mu1 : 第一个分布的稳健中位数
        std1 : 第一个分布的稳健标准差
        mu2 : 第二个分布的稳健中位数
        std2 : 第二个分布的稳健标准差
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 绘制分布曲线
    ax.plot(x_d, pdf1, color='#2ecc71', linewidth=2.5, 
            label=f'Distribution 1: μ={mu1:.2f}, σ={std1:.2f}')
    ax.plot(x_d, pdf2, color='#e74c3c', linewidth=2.5, linestyle='--',
            label=f'Distribution 2: μ={mu2:.2f}, σ={std2:.2f}')
    
    # 添加分布参数标注
    textstr = '\n'.join((
        r'$\mu_1=%.2f$' % mu1,
        r'$\sigma_1=%.2f$' % std1,
        r'$\mu_2=%.2f$' % mu2,
        r'$\sigma_2=%.2f$' % std2))
    ax.text(0.75, 0.95, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 添加中位线
    ax.axvline(mu1, color='#2ecc71', linestyle=':', alpha=0.7)
    ax.axvline(mu2, color='#e74c3c', linestyle=':', alpha=0.7)
    
    # 设置坐标轴
    ax.set_xlabel('Value', fontsize=12)
    ax.set_ylabel('Normalized Probability Density', fontsize=12)
    ax.set_title('Robust Normal Distribution Comparison', fontsize=14, pad=20)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # 优化显示范围
    x_min = min(mu1 - 3*std1, mu2 - 3*std2)
    x_max = max(mu1 + 3*std1, mu2 + 3*std2)
    ax.set_xlim(x_min, x_max)
    
    # 添加图例
    ax.legend(loc='upper left', frameon=True)
    
    plt.tight_layout()
    plt.show()
    
# plot_distributions(x_d, pdf1, pdf2, mu1, std1, mu2, std2)
# assert 0

# assert 0

'''reference_list = list(reference_word_relevance_dict.values())
prediction_list = list(prediction_word_relevance_dict.values())
reference_relevance_score = np.mean(reference_list) / math.exp(np.std(reference_list) - 1)
prediction_relevance_score = np.mean(prediction_list) / math.exp(np.std(prediction_list) - 1)
relative_relevance_score = max(prediction_relevance_score, reference_relevance_score) / (min(prediction_relevance_score, reference_relevance_score) + 1e-9)
print(f"[reference dataset relevance score]: {reference_relevance_score}")
print(f"[prediction dataset relevance score]: {prediction_relevance_score}")
print(f"[relative relevance score]: {relative_relevance_score}")'''
print()

"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
# from sklearn.model_selection import GridSearchCV
# from sklearn.model_selection import LeaveOneOut

def get_kde_distribution_from_float_list(input_list, bandwidth=0.1, num_samples=1000, min_x_d=-0.25, max_x_d=0.5):
    '''
    使用核密度估计（KDE）从浮点数列表中计算概率分布，并对小于min_x_d和大于max_x_d的密度进行统计。

    参数:
    - input_list (list): 一个列表，元素是浮点数。
    - bandwidth (float): 核密度估计的带宽参数，控制平滑度。
    - num_samples (int): 横坐标估计的间隔数量，用于生成平滑的概率密度函数。
    - min_x_d (float): KDE评估点的最小值。
    - max_x_d (float): KDE评估点的最大值。

    返回:
    - dict: 一个字典，key为浮点数值，value为该值的概率密度。
    '''
    # 转换为NumPy数组并重塑为列向量
    input_array = np.array(input_list).reshape(-1, 1)
    
    print("input_array.shape: ", input_array.shape)
    
    '''# 使用KernelDensity进行核密度估计
    grid = GridSearchCV(
    estimator=KernelDensity(kernel='gaussian'),
        param_grid={'bandwidth': 10 ** np.linspace(-1, 1, 100)},
        cv=LeaveOneOut(),
    )
    grid.fit(input_array)
    print(f'最佳带宽：{grid.best_params_["bandwidth"]}')
    assert 0''' # Too slow

    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
    kde.fit(X=input_array)

    # 创建评估点，范围为[min_x_d, max_x_d]
    x_d = np.linspace(min_x_d, max_x_d, num_samples).reshape(-1, 1)
    
    print("x_d shape: ", x_d.shape)
    
    # 计算每个评估点的概率密度
    log_dens = kde.score_samples(x_d)
    
    print("log_dens shape: ", log_dens.shape)
    print("log_dens sum: ", sum(log_dens))
    
    densities = np.exp(log_dens)  # 转换为实际的概率密度
    
    print("densities shape: ", densities.shape)
    print("densities sum: ", sum(densities))
    
    # 计算区间宽度（假设是均匀分布）
    dx = x_d[1] - x_d[0]  # 计算相邻点之间的距离（假设均匀间隔）
    
    # 归一化密度，使得概率密度的总和为 1
    densities_normalized = densities * dx  # 乘以区间宽度
    
    print("densities_normalized shape: ", densities_normalized.shape)
    print("densities_normalized sum: ", sum(densities_normalized))
    
    return x_d, densities_normalized

# 示例数据
# list1 = [0.1, 0.2, 0.2, 0.3, 0.4, 0.4, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # 浮动数据
# list2 = [0.05, 0.15, 0.25, 0.35, 0.45, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]  # 另一个浮动数据

# 使用分箱方法计算分布
# distribution1 = get_distribution_from_float_list(list1, num_bins=20)
# distribution2 = get_distribution_from_float_list(list2, num_bins=20)

# 使用KDE方法计算分布
num_samples = 5000
min_x_d = -0.2 # -0.5
max_x_d = 0.2 # 1.0
bandwidth = 0.01

x_d = np.linspace(min_x_d, max_x_d, num_samples).reshape(-1, 1)

x_d1, densities1 = get_kde_distribution_from_float_list(list1, bandwidth=bandwidth, num_samples=num_samples, min_x_d=min_x_d, max_x_d=max_x_d)
x_d2, densities2 = get_kde_distribution_from_float_list(list2, bandwidth=bandwidth, num_samples=num_samples, min_x_d=min_x_d, max_x_d=max_x_d)

print(sum(densities1))
print(sum(densities2))

def kl_divergence(densities_p, densities_q):
    # KL散度计算，注意这里需要确保densities_p和densities_q都已经正规化
    epsilon = 1e-10
    densities_p = np.maximum(densities_p, epsilon)  # 任何为零的密度值都变成一个非常小的值
    densities_q = np.maximum(densities_q, epsilon)  # 任何为零的密度值都变成一个非常小的值
    kl_div = np.sum(densities_p * np.log2(densities_p / densities_q))
    return kl_div

def js_divergence(densities_p, densities_q):
    densities_m = (densities_p + densities_q) / 2
    js_div = (kl_divergence(densities_p, densities_m) + kl_divergence(densities_q, densities_m)) / 2
    return js_div

def ChiSquare(densities_p, densities_q):
    epsilon = 1e-10
    densities_q = np.maximum(densities_q, epsilon)  # 任何为零的密度值都变成一个非常小的值
    return np.sum(np.square(densities_p-densities_q)/densities_q)
    # return np.sum((densities_p/densities_q - 1) * densities_q)

def HellingerDistance(densities_p, densities_q):
    return 1/np.sqrt(2)*np.linalg.norm(np.sqrt(densities_p)-np.sqrt(densities_q))

def BhattacharyyaDistance(densities_p, densities_q):
    return -np.log2(np.sum(np.sqrt(densities_p * densities_q)))

def entropy(densities):
    epsilon = 1e-10
    densities = np.maximum(densities, epsilon)  # 任何为零的密度值都变成一个非常小的值
    return np.sum(- densities * (np.log2(densities)))

def cross_entropy(densities_p, densities_q):
    epsilon = 1e-10
    densities_q = np.maximum(densities_q, epsilon)  # 任何为零的密度值都变成一个非常小的值
    return np.sum(- densities_p * (np.log2(densities_q)))

'''print(densities1[:10])
print()
print(densities2[:10])'''

# 计算KL散度
kl_value1 = kl_divergence(densities1, densities2)
print(f"KL Divergence KL(ref||pred): {kl_value1}")

'''kl_value2 = kl_divergence(densities2, densities1)
print(f"KL Divergence KL(pred||ref): {kl_value2}")

js_value = js_divergence(densities1, densities2) # symmetric
print(f"JS Divergence: {js_value}")

# chi_square_value1 = ChiSquare(densities1, densities2)
# print(f"Chi-Square Chi2(ref||pred): {chi_square_value1}")

# chi_square_value2 = ChiSquare(densities2, densities1)
# print(f"Chi-Square Chi2(pred||ref): {chi_square_value2}")

hellinger_value = HellingerDistance(densities1, densities2) # symmetric
print(f"Hellinger Distance: {hellinger_value}")

bhattacharyya_distance = BhattacharyyaDistance(densities1, densities2) # symmetric
print(f"Bhattacharyya Distance: {bhattacharyya_distance}")

entropy_ref = entropy(densities1)
entropy_pred = entropy(densities2)
print(f"Entropy reference: {entropy_ref}")
print(f"Entropy prediction: {entropy_pred}")

ce = cross_entropy(densities2, densities1)
print(f"CE(pred||ref): {ce}")

ce = cross_entropy(densities1, densities2)
print(f"CE(ref||pred): {ce}")

# 可视化比较
plt.figure(figsize=(9.6, 6.8))

# 绘制分箱（Binning）结果
# plt.hist(distribution1, bins=5, density=True, alpha=0.5, label='List1 Binning', color='blue')
# plt.hist(distribution2, bins=5, density=True, alpha=0.5, label='List2 Binning', color='red')

# 绘制曲线，并设置颜色和线条加粗
plt.plot(x_d1, densities1, label='reference dataset', color='#547BB4', linestyle='--', linewidth=5)
plt.plot(x_d2, densities2, label='prediction dataset', color='#DD7C4F', linestyle='--', linewidth=5)

# 设置标题和坐标标签
# plt.title('Word Relevance Distributions', fontsize=30)
plt.title('', fontsize=30)
plt.xlabel('Word Relevance', fontsize=26)
plt.ylabel('Density', fontsize=26)

# 设置图例，调整位置和字体大小
plt.legend(loc='best', fontsize=22)
plt.xticks([])
plt.yticks([])

# 调整图像布局以避免标签重叠
plt.tight_layout()

# 保存图像，设置分辨率为300 dpi
plt.savefig('tight_image.png', dpi=300)

# 显示图像
plt.show()
'''

from scipy.stats import norm

from scipy.stats import lognorm

def fit_lognorm_distribution(x_d, data):
    # 拟合正态分布
    shape, loc, scale = lognorm.fit(data)  # 计算拟合的均值和标准差
    
    # 生成正态分布的概率密度函数
    pdf = lognorm.pdf(x_d, shape, loc, scale)
    
    return shape, loc, scale, pdf

def fit_normal_distribution(x_d, data):
    # 拟合正态分布
    mu, std = norm.fit(data)  # 计算拟合的均值和标准差
    
    # 生成正态分布的概率密度函数
    pdf = norm.pdf(x_d, mu, std)
    
    return mu, std, pdf
    
    # 绘制图形
    plt.plot(x_d, densities, label=f'{label} - KDE', color='blue')
    plt.plot(x_d, pdf, label=f'{label} - Normal fit\n(mu={mu:.2f}, std={std:.2f})', color='red', linestyle='--')

def robust_fit_normal_distribution(x_d, data):
    median = np.median(data)
    iqr = np.percentile(data, 75) - np.percentile(data, 25)
    std_robust = iqr / 1.349
    pdf_robust = norm.pdf(x_d, median, std_robust)
    return median, std_robust, pdf_robust

# 绘制并拟合正态分布
plt.figure(figsize=(9.6, 6.8))

label1 = 'reference'
label2 = 'prediction'
mu1, std1, pdf1 = fit_normal_distribution(x_d, list1)
mu2, std2, pdf2 = fit_normal_distribution(x_d, list2)

shape1, loc1, scale1, pdf1 = fit_lognorm_distribution(x_d, list1)
shape2, loc2, scale2, pdf2 = fit_lognorm_distribution(x_d, list2)

mu1, std1, pdf1 = robust_fit_normal_distribution(x_d, list1)
mu2, std2, pdf2 = robust_fit_normal_distribution(x_d, list2)

pdf1 = pdf1 / sum(pdf1)
pdf2 = pdf2 / sum(pdf2)

kl_value1 = kl_divergence(pdf1, pdf2)
print(f"KL Divergence KL(ref||pred): {kl_value1}")

print("------")
print("pdf sum:")
print(sum(pdf1))
print(sum(pdf2))
print("------")
print("density sum:")
print(sum(densities1))
print(sum(densities2))
print("------")

plt.plot(x_d, pdf1, label=f'{label1} - Normal fit\n(mu={mu1:.4f}, std={std1:.4f})', color='#547BB4', linestyle='--', linewidth=5)
plt.plot(x_d, pdf2, label=f'{label2} - Normal fit\n(mu={mu2:.4f}, std={std2:.4f})', color='#DD7C4F', linestyle='--', linewidth=5)

plt.plot(x_d1, densities1, label='reference real', color='#547BB4', linestyle='dotted', linewidth=2)
plt.plot(x_d2, densities2, label='prediction real', color='#DD7C4F', linestyle='dotted', linewidth=2)

# 添加标题和标签
plt.title('KDE and Normal Distribution Fit', fontsize=14)
plt.xlabel('Value', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.legend()

# 显示图形
plt.show()

assert 0
"""

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

# Step2: Get sequence-wise relevance in reference and prediction dataset
print("="*10)
print("Step 2: Get sequence-wise relevance in reference and prediction dataset")
reference_sequence_relevances = compute_sequence_relevance(reference_data, reference_word_relevance_dict, tokenizer)
prediction_sequence_relevances = compute_sequence_relevance(prediction_data, prediction_word_relevance_dict, tokenizer)
print(f"[reference sequence relevances]: mean={np.mean(reference_sequence_relevances)}, std={np.std(reference_sequence_relevances)}")
print(f"[prediction sequence relevances]: mean={np.mean(prediction_sequence_relevances)}, std={np.std(prediction_sequence_relevances)}")
print()

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

'''# Step2: Get reference dataset relevance score
print("="*10)
print("Step 2: Get reference and prediction dataset relevance score (sequence wise)")
reference_relevance_score =  np.mean(reference_sequence_relevances) / math.exp(np.std(reference_sequence_relevances) - 1)
prediction_relevance_score = np.mean(prediction_sequence_relevances) / math.exp(np.std(prediction_sequence_relevances) - 1)
relative_relevance_score = max(reference_relevance_score, prediction_relevance_score) / (min(reference_relevance_score, prediction_relevance_score) + 1e-9)
print(f"[reference dataset relevance score]: {reference_relevance_score}")
print(f"[prediction dataset relevance score]: {prediction_relevance_score}")
print(f"[relative relevance score]: {relative_relevance_score}")'''
reference_sequence_relevances = torch.tensor(reference_sequence_relevances)
prediction_sequence_relevances = torch.tensor(prediction_sequence_relevances)
# print()

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

# Step3: Get prediction and reference embeddings
print("="*10)
print("Step 3: Get prediction and reference embeddings")
if embed_method == "gpt2":
    raise NotImplementedError("GPT2 is not supported in this version.")
elif embed_method == 'gtr':
    print("Use SentenceTransformer as the embedding encoder as recommend in vec2text.")
    prediction_embeddings = featurize_sequences_from_sentence_transformer(encoder, tokenize_and_truncate(tokenizer, prediction_data, 256), 16, "predictions", True)
    reference_embeddings = featurize_sequences_from_sentence_transformer(encoder, tokenize_and_truncate(tokenizer, reference_data, 256), 16, "references", True)
else:
    raise ValueError(f'embed_method in [\'gtr\', \'gpt2\'], but get {embed_method}')
print()

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

# Step4: Cluster embeddings through k-means
print("="*10)
print("Step 4: Cluster embeddings through k-means")
if num_buckets == 'auto':
    # Heuristic: use num_clusters = num_generations / 10.
    num_buckets = max(2, int(round(min(prediction_embeddings.shape[0], reference_embeddings.shape[0]) / 10)))
elif not isinstance(num_buckets, int):
    raise ValueError('num_buckets is expected to be an integer or "auto"')
t1 = time.time()
p, q, p_smoothed, q_smoothed = weighed_cluster_feats(
    p=prediction_embeddings, q=reference_embeddings, 
    p_weight=prediction_sequence_relevances, q_weight=reference_sequence_relevances,
    num_clusters=num_buckets, norm='l2', whiten=False, pca_max_data=pca_max_data,
    explained_variance=kmeans_explained_var, num_redo=kmeans_num_redo,
    max_iter=kmeans_max_iter, seed=random_seed, verbose=True
)
t2 = time.time()
print('total discretization time:', round(t2-t1, 2), 'seconds')
print()

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# Step5: Compute weighted-mauve after k-means clustering
print("="*10)
print("Step 5: Compute weighted-mauve after k-means clustering")
# Divergence curve and MAUVE (no smoothing).
mixture_weights = np.linspace(1e-6, 1-1e-6, divergence_curve_discretization_size)
divergence_curve = get_divergence_curve_for_multinomials(p, q, mixture_weights, mauve_scaling_factor)
x, y = divergence_curve.T
idxs1 = np.argsort(x)
idxs2 = np.argsort(y)
mauve_score = 0.5 * (
    compute_area_under_curve(x[idxs1], y[idxs1]) +
    compute_area_under_curve(y[idxs2], x[idxs2])
)
fi_score = get_fronter_integral(p, q)

# Divergence curve and MAUVE (with smoothing).
x_s, y_s = get_divergence_curve_for_multinomials(p_smoothed, q_smoothed, mixture_weights, mauve_scaling_factor).T
idxs1, idxs2 = np.argsort(x_s), np.argsort(y_s)
mauve_star = 0.5 * (
    compute_area_under_curve(x_s[idxs1], y_s[idxs1]) +
    compute_area_under_curve(y_s[idxs2], x_s[idxs2])
)
fi_star = get_fronter_integral(p_smoothed, q_smoothed)
mauve_results = SimpleNamespace(
    p_hist=p, q_hist=q, divergence_curve=divergence_curve, 
    mauve=mauve_score, frontier_integral=fi_score,
    mauve_star=mauve_star, frontier_integral_star=fi_star,
    num_buckets=num_buckets,
)

print(f"[prediction dataset path]: {prediction_path}")
print(f"[reference dataset path]: {reference_path}")
print(f"[mauve scaling factor]: {mauve_scaling_factor}")
print(f"[mauve]: {mauve_results.mauve}")
print(f"[frontier integral]: {mauve_results.frontier_integral}")
print(f"[mauve*]: {mauve_results.mauve_star}")
print(f"[frontier integral*]: {mauve_results.frontier_integral_star}")

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# Step6: Post-process weighted-mauve with relevance score
print("="*10)
print("Step 6: Post-process weighted-mauve with a punishment coefficient (normalized_mauve=(mauve*puni)/(puni+ϵ)")
# punishment_coefficient = math.exp(relative_relevance_score - 1)
punishment_coefficient = relative_relevance_score
normalized_mauve = mauve_results.mauve / punishment_coefficient
normalized_mauve_star = mauve_results.mauve_star / punishment_coefficient
# FI summary is not fit in this situation.
# normalized_frontier_integral = mauve_results.frontier_integral / punishment_coefficient
# normalized_frontier_integral_star = mauve_results.frontier_integral_star / punishment_coefficient
print(f"[mauve]: {mauve_results.mauve}")
print(f"[KL Divergence KL(ref||pred)]: {js_normal}")
print(f"[punishment coefficient]: {punishment_coefficient}")
print(f"[normalized mauve]: {normalized_mauve}")
print(f"[normalized mauve*]: {normalized_mauve_star}")

print(f"{mauve_results.mauve:.4f}\t{js_normal:.4f}\t{punishment_coefficient:.4f}\t{normalized_mauve:.4f}")

exit(0)

print('---')
punishment_coefficient = relative_relevance_score_v1
normalized_mauve = mauve_results.mauve / punishment_coefficient
normalized_mauve_star = mauve_results.mauve_star / punishment_coefficient
# FI summary is not fit in this situation.
# normalized_frontier_integral = mauve_results.frontier_integral / punishment_coefficient
# normalized_frontier_integral_star = mauve_results.frontier_integral_star / punishment_coefficient
print(f"[mauve]: {mauve_results.mauve}")
print(f"[KL Divergence KL(ref||pred)]: {js_normal}")
print(f"[punishment coefficient]: {punishment_coefficient}")
print(f"[normalized mauve]: {normalized_mauve}")
print(f"[normalized mauve*]: {normalized_mauve_star}")



exit(0)

# print(f"[normalized frontier integral]: {normalized_frontier_integral}")
# print(f"[normalized frontier integral*]: {normalized_frontier_integral_star}")
print(f"Take [normalized mauve] as the final result: {normalized_mauve}")
print(f"The threshold p-value is set to {p_threshold}")
print(f"The threshold mauve value is set to {0.5}")

result_flag = (normalized_mauve >= p_threshold and mauve_score > 0.5)
result_answer = 'positive' if result_flag else 'negative'
print(f'--- Result: {result_answer} ---')

if result_flag:
    print(f"Since [normalized-mauve={normalized_mauve}] >= [p-value={p_threshold}] and [mauve={mauve_score}] >= 0.5, the model may be trained on reference dataset.")
else:
    print(f"Since not ([mauve={normalized_mauve}] < [p-value={p_threshold}] and [mauve={mauve_score}] >= 0.5) , the model may not be trained on reference dataset.")
