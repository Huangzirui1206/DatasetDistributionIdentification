import os
import sys
import torch
import string
import random
import vec2text
import transformers
from tqdm import tqdm

from sentence_transformers import SentenceTransformer

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, LlamaTokenizer
from peft import PeftModel

from torch.utils.data import DataLoader, Dataset
from transformers import DataCollatorForSeq2Seq

from typing import List, Tuple, Callable

from swift.llm import InferEngine, InferRequest, PtEngine, RequestConfig, load_dataset
from swift.plugin import InferStats
from swift.llm import VllmEngine

# Tips:
## 需要修改本地inversion_model和corrector_model的config，将model_name_or_path手动改成本地地址./pretrained/t5-base。
## 需要修改本地inversion_model和corrector_model的config，将embedder_model_name手动改成本地地址./pretrained/gtr-t5-base。
## 下载gtr-t5-base需要将1_Pooling和2_Dense一并下载
## 为了本地能够在SentenceTransformer中加载inversion的embedder_model_name="./pretrained/gtr-t5-base"，需要修改：
## ... (line ~/anaconda3/envs/LLM-MIA/lib/python3.10/site-packages/vec2text/models/model_utils.py" Line 252)
## elif 'gtr-t5-base' in name:
##         model = SentenceTransformer(name)
##         tokenizer = model.tokenizer
## inversion_trainer = vec2text.trainers.InversionTrainer(...) 需要加载accuracy模块，首先将huggingface metric从github下载到本地，
## metric文件夹中包含accuracy文件夹；然后需要修改/home/huangzr/anaconda3/envs/LLM-MIA/lib/python3.10/site-packages/vec2text/trainers/base.py
## Line 63，将self.metric_accuracy = evaluate.load("accuracy")修改为self.metric_accuracy = evaluate.load("/home/huangzr/LLM-MIA/metric/accuracy")
## 类似地，同一个文件中后面几行其他的evaluate.load(...)也需要修改。

# utility functions
def set_random_seed(seed: int = 42):
    """
    设置所有相关库的随机数种子，包括random, numpy, torch等。
    
    参数:
    - seed (int): 随机数种子，默认为42。
    """
    # 设置Python的random模块的随机种子
    random.seed(seed)
    
    # 设置NumPy的随机数种子
    # np.random.seed(seed)
    
    # 设置PyTorch的随机数种子
    torch.manual_seed(seed)
    
    # 如果使用GPU，确保结果是可重复的
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # 设置当前设备的随机数种子
        torch.cuda.manual_seed_all(seed)  # 设置所有GPU设备的随机数种子
        
    # print(f"Random seed set to {seed}")

def get_target_model(base_model:str, peft_weights:str, use_peft:bool=True, load_8bit=False, device='cuda', load_bnb_4bit=False) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(base_model, padding_side="left")
    if device == "cuda":
        # model = LlamaForCausalLM.from_pretrained(
        #     base_model,
        #     load_in_8bit=load_8bit,
        #     torch_dtype=torch.float16,
        #     device_map="auto",
        # )
        if not load_bnb_4bit:
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                device_map={"": device},
                torch_dtype=torch.float16,
            )
        elif load_bnb_4bit:
            # 配置4位量化参数
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,  # 启用4位量化加载
                bnb_4bit_quant_type="nf4",  # 使用NormalFloat4量化
                bnb_4bit_use_double_quant=True,  # 启用双重量化压缩
                bnb_4bit_compute_dtype=torch.float16  # 计算时使用float16精度
            )

            # 加载量化模型
            model = AutoModelForCausalLM.from_pretrained(
                base_model,  # 替换为实际模型路径（如："Meta-Llama-3.1-8B-bnb-4bit"）
                quantization_config=bnb_config,  # 关键参数：应用量化配置
                device_map={"": device},  # 分布式训练设备映射
                trust_remote_code=True,  # 信任自定义代码
                # 以下参数需要调整或删除：
                # load_in_8bit=False,  # 已由 quantization_config 覆盖
                # torch_dtype=torch.float16  # 已由 bnb_4bit_compute_dtype 覆盖
            )
        if use_peft:
            model = PeftModel.from_pretrained(
                model,
                peft_weights,
                torch_dtype=torch.float16,
            )
        # Move the model to the specified device
        # !!!!! TODO: NEED TO SOLVE THE DEVICE PROBLEM
        # model = model.to(device)
    elif device == "mps":
        # model = LlamaForCausalLM.from_pretrained(
        #     base_model,
        #     device_map={"": device},
        #     torch_dtype=torch.float16,
        # )
        if not load_bnb_4bit:
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                device_map={"": device},
                torch_dtype=torch.float16,
            )
        elif load_bnb_4bit:
            # 配置4位量化参数
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,  # 启用4位量化加载
                bnb_4bit_quant_type="nf4",  # 使用NormalFloat4量化
                bnb_4bit_use_double_quant=True,  # 启用双重量化压缩
                bnb_4bit_compute_dtype=torch.float16  # 计算时使用float16精度
            )

            # 加载量化模型
            model = AutoModelForCausalLM.from_pretrained(
                base_model,  # 替换为实际模型路径（如："Meta-Llama-3.1-8B-bnb-4bit"）
                quantization_config=bnb_config,  # 关键参数：应用量化配置
                device_map={"": device},  # 分布式训练设备映射
                trust_remote_code=True,  # 信任自定义代码
                # 以下参数需要调整或删除：
                # load_in_8bit=False,  # 已由 quantization_config 覆盖
                # torch_dtype=torch.float16  # 已由 bnb_4bit_compute_dtype 覆盖
            )
        if use_peft:
            model = PeftModel.from_pretrained(
                model,
                peft_weights,
                device_map={"": device},
                torch_dtype=torch.float16,
            )
    else:
        # model = LlamaForCausalLM.from_pretrained(
        #     base_model, device_map={"": device}, low_cpu_mem_usage=True
        # )
        if not load_bnb_4bit:
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                device_map={"": device},
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            )
        elif load_bnb_4bit:
            # 配置4位量化参数
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,  # 启用4位量化加载
                bnb_4bit_quant_type="nf4",  # 使用NormalFloat4量化
                bnb_4bit_use_double_quant=True,  # 启用双重量化压缩
                bnb_4bit_compute_dtype=torch.float16  # 计算时使用float16精度
            )
            # 加载量化模型
            model = AutoModelForCausalLM.from_pretrained(
                base_model,  # 替换为实际模型路径（如："Meta-Llama-3.1-8B-bnb-4bit"）
                quantization_config=bnb_config,  # 关键参数：应用量化配置
                device_map={"": device},  # 分布式训练设备映射
                trust_remote_code=True,  # 信任自定义代码
                low_cpu_mem_usage=True,  # 降低CPU内存使用
                # 以下参数需要调整或删除：
                # load_in_8bit=False,  # 已由 quantization_config 覆盖
                # torch_dtype=torch.float16  # 已由 bnb_4bit_compute_dtype 覆盖
            )
        if use_peft:
            model = PeftModel.from_pretrained(
                model,
                peft_weights,
                device_map={"": device},
            )
    
    # Unwind broken config
    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.pad_token = tokenizer.eos_token
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    
    tokenizer.pad_token_id = (
        tokenizer.eos_token_id 
    )

    if not load_8bit:
        model.half()  # Seems to fix bugs for some users.
        
    model.eval()
    
    return model, tokenizer 

def get_vec2text_model(inversion_model_path:str, corrector_model_path:str, verbose=False):
    inversion_model = vec2text.models.InversionModel.from_pretrained(
        pretrained_model_name_or_path = inversion_model_path
    )
    if verbose:
        print(f"Get inversion model from {inversion_model_path}.")

    corrector_model = vec2text.models.CorrectorEncoderModel.from_pretrained(
        pretrained_model_name_or_path = corrector_model_path
    )
    if verbose:
        print(f"Get corrector model from {corrector_model_path}.")

    corrector = vec2text.load_corrector(inversion_model, corrector_model)
    if verbose:
        print(f"Get corrector combined with corrector_model.")
        print()
        
    return corrector

def get_conjunctive_strings() -> List[str]:
    # return a list of short terms to use as prompts for causal lm
    return [
        # "but ",
        "However, ",
        "Therefore, ",
        "Meanwhile, ",
        "As a result, ",
        "On the contrary, ",
        "By contrast, ",
        "In addition, ",
        "Furthermore, ",
        "Moreover, ",
        "For example, ",
        "In conclusion, ",
        "In summary, ",
        "On the other hand, ",
        "In fact, ",
        "Consequently, ",
        "Thus, ",
        "In contrast, ",
        "For instance, ",
        "Nevertheless, ",
        "Although, ",
        "Otherwise, ",
        "That is, ",
        "As such, ",
        "Because of this, ",
        "In particular, ",
        "To sum up, ",
        "Accordingly, ",
        "Ultimately, ",
        "Even so, ",
        "Given that, ",
        "In the meantime, ",
        "Regardless, "
    ]

def get_prompt_questions(num_prompts:int = 100, domain_word:str|None=None) -> List[str]:
    prompt = "Explain __ in detail." if domain_word is None else f"[DOMAIN: {domain_word}] Explain __ in detail."
    return [prompt] * num_prompts
    return [
        # 1. 开放式问题 Open problems: 
        "What is __, and how does it work?", 
        "Explain __ in detail.", 
        "What are the key features of __?", 
        "How does __ function?", 
        "What is the overviw of __?",
        "Could you please share some information regarding __?",
        "What are the main aspects of __?",
        "What are the key points to understand about __?",

        # 2. 对比问题 Comparisons:
        "What are the differences between __ and __?", 
        "Compare and contrast __ with __.", 
        "What are the advantages and disadvantages of __ compared to __?", 

        # 3. 步骤性问题 Step by step:
        "What are the steps involved in __?", 
        "How can I __ step by step?", 
        "Describe the process of __ in sequential order.", 

        # 4. 假设性问题 Hypothesis:
        "What would happen if __?", 
        "How would __ change if __?", 
        "What might be the consequences of __?", 

        # 5. 分类与列举问题 Class and list
        "What are the main types of __?", 
        "What consist the key components of __?", 
        "What are the different categories of __?", 

        # 6. 原因与结果问题 Cause and results:
        "What are the causes of __?", 
        "What are the effects of __?", 
        "Why does __ occur?", 

        # 7. 定义与解释问题 Definition and explanation
        "What is the definition of __?", 
        "Can you explain __ in simple terms?", 
        "What does __ mean?", 

        # 8. 建议与解决方案问题 Suggestion and solution:
        "What are some strategies to __?", 
        "How can I solve __?", 
        "What are the best practices for __?", 

        # 9. 评价与判断问题 Judgement and evaluation
        "What are the strengths and weaknesses of __?", 
        "What evidence supports or challenges the claim that __?", 
        "How might the claim that __ be interpreted differently across various contexts or perspectives?", 

        # 10. 创造性问题 Creativity
        # "What are some innovative ways to __?", 
        # "Can you suggest a creative solution to __?", 
        # "What would a futuristic version of __ look like?", 

        # 11. 多轮对话问题 Multiple rounds questions:
        "Let's discuss __. What are your initial thoughts?", 
        "Can we have a conversation about __?", 
        "I'd like to explore __ further. What questions should I ask?", 

        # 12. 角色扮演问题 Role playing:
        "As a __, what would you recommend for __?", 
        "If you were a __, how would you approach __?", 
        "From the perspective of __, what are the key considerations for __?", 
    ]

def construct_alpaca_prompt_input(question: str, background: str) -> str:
    return f"**question**: {question}\n**background**: {background}" if background else f"{question}"

def get_alpaca_prompt_module(instruction_content:str, input_content:str=None, set_input: bool = False) -> str:
    if set_input and input_content:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
{instruction_content}

### Input:
{input_content}

### Response:
""" # noqa: E501
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction_content}

### Response:
"""

def add_random_perturbation(embeddings: torch.Tensor, perturbation_magnitude: float, perturbation_magnitude_delta: float = 1e-3) -> torch.Tensor:
    """
    给每个embedding添加一个随机扰动。
    
    参数:
        embeddings (torch.Tensor): 形状为 (n, 768) 的嵌入矩阵，其中 n 为序列数量，已经转移到 GPU。
        perturbation_magnitude (float): 扰动的模长。
        perturbation_magnitude_delta (float): 用于扰动模长范围的变化量，默认是 1e-3。
    
    返回:
        torch.Tensor: 添加扰动后的嵌入矩阵，形状为 (n, 768)，与原始张量在同一设备上。
    """
    # 获取嵌入的数量 n 和维度 d (即768)
    n, d = embeddings.shape
    
    # 为每个嵌入生成一个随机方向的扰动
    random_directions = torch.randn(n, d, device=embeddings.device)  # 生成标准正态分布的随机数 (n, 768)
    
    # 归一化为单位向量
    random_directions /= torch.norm(random_directions, dim=1, keepdim=True)  # 计算L2范数并归一化
    
    # 在 [perturbation_magnitude - perturbation_magnitude_delta, perturbation_magnitude + perturbation_magnitude_delta] 范围内生成随机值
    perturbation_magnitude_random = torch.empty(n, device=embeddings.device).uniform_(
        perturbation_magnitude - perturbation_magnitude_delta, 
        perturbation_magnitude + perturbation_magnitude_delta
    )
    
    # 按照给定的随机模长进行缩放
    perturbations = random_directions * perturbation_magnitude_random.view(-1, 1)  # 扩展维度使其匹配
    
    # 添加扰动到每个embedding
    perturbed_embeddings = embeddings + perturbations
    
    return perturbed_embeddings

def generated_output_from_target_model(sequences: List[str], model: AutoModelForCausalLM, tokenizer:AutoTokenizer,
                                       cutoff_length:int=256, batch_size:int=16, device='cuda',
                                       top_k=15, max_new_tokens:int=256, min_tokens:int=16, 
                                       temperature:float=0.7, top_p:float=0.9, repetition_penalty:float=1.2,
                                       use_alpaca_module_template=True,
                                       do_sample:bool=True) -> List[str] :
    
    results = tokenizer(sequences, return_tensors='pt', padding=True, truncation=True, max_length=cutoff_length)

    generated_data = []
    with torch.no_grad():
        for start in tqdm(range(0, len(sequences), batch_size), desc="generated output from target model"):
            batched_sequences = sequences[start:start+batch_size]
            input_ids = results['input_ids'][start:start+batch_size].to(device)
            attention_mask = results['attention_mask'][start:start+batch_size].to(device)

            try:
                outputs = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=max_new_tokens,
                    min_length=min_tokens,
                    attention_mask=attention_mask,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            except TypeError as e:
                print("Error during model.generate call:", e)
                print("Ensure your transformers version is up-to-date.")
                return
            
            for i, output in enumerate(outputs):
                generated_output = tokenizer.decode(output, skip_special_tokens=True)
                if use_alpaca_module_template:
                    try:
                        # 检查是否包含响应标记
                        if "### Response:" not in generated_output:
                            # 不包含标记，跳过当前循环
                            continue
                        # 提取响应内容
                        generated_output = generated_output.split("### Response:")[1].strip()
                    except IndexError:
                        # 处理分割时可能出现的异常，跳过当前循环
                        continue
                else:
                    generated_output = generated_output[len(batched_sequences[i]):].strip()
                # 保存结果
                generated_data.append(generated_output)
              
    return generated_data  

def save_json_dataset(saved_data:List[dict], save_path:str, verbose=False):
    # saved_data = [{save_key:sequence} for sequence in sequences]
    # Save generated data to a file
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    import json
    try:
        with open(save_path, "w", encoding='utf-8') as f:
            json.dump(saved_data, f, indent=4)
    except Exception as e:
        save_path = save_path + '/generated_dataset.json'
        with open(save_path, "w", encoding='utf-8') as f:
            json.dump(saved_data, f, indent=4)
    if verbose:
        print(f"Generated outputs saved to {save_path}")

#### Optimize 

def get_gtr_embeddings(text_list, encoder, tokenizer, max_length=32, max_seg_num=10) -> list:
    """
    处理长句子：分割为多个满足max_length的短句，返回每个句子的嵌入列表
    返回格式：列表的列表，外层列表对应每个输入句子，内层列表是该句子的各短句嵌入
    """
    all_sentence_embeddings = []
    
    seg_num = []
    for text in text_list:
        # 1. 分割长句子为多个短句
        segments = split_long_sentence(text, tokenizer, max_length) 
        seg_num.append(len(segments))

        segment_embeddings = []
        
        # 2. 处理每个短句
        for segment in segments:
            inputs = tokenizer(
                segment,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
                padding="max_length",
            ).to("cuda")
            
            with torch.no_grad():
                # T5模型处理
                model_output = encoder(input_ids=inputs['input_ids'], 
                                      attention_mask=inputs['attention_mask'])
                
                # 获取T5的隐藏状态（T5输出是元组，第一个元素是隐藏状态）
                hidden_state = model_output[0]
                
                # 使用注意力掩码进行平均池化
                embeddings = vec2text.models.model_utils.mean_pool(hidden_state, inputs['attention_mask'])
                segment_embeddings.append(embeddings)
        
        # 3. 将当前句子的所有短句嵌入添加到结果列表
        if max_seg_num > 0:
            segment_embeddings = segment_embeddings[:min(max_seg_num, len(segment_embeddings))]
        all_sentence_embeddings.append(torch.cat(segment_embeddings, dim=0))
    
    return all_sentence_embeddings

def split_long_sentence(text, tokenizer, max_length, min_merge_ratio=0.3):
    """
    将长文本分割为多个token数不超过max_length的短句，优化点：
    1. 基于正则高效分割中英文句子（适配标点+后续空格）
    2. 缓存token信息，减少重复tokenize计算
    3. 直接基于token列表切割，确保token数精确可控
    4. 合并过短片段时保留句子边界，兼顾紧凑性与语义完整
    5. 过滤空片段，适配中英文多语言场景
    
    Args:
        text: 输入长文本（支持中英文）
        tokenizer: Hugging Face Tokenizer实例（如GPT、BERTTokenizer）
        max_length: 每个片段的最大token数（不含特殊token如[CLS]）
        min_merge_ratio: 过短片段阈值（小于max_length*min_merge_ratio则尝试合并）
    
    Returns:
        list: 分割后的短句列表，每个短句token数≤max_length
    """
    import re
    from typing import Tuple, List

    # -------------------------- 1. 预处理：清理多余空格，避免空字符干扰 --------------------------
    text = re.sub(r'\s+', ' ', text).strip()  # 多个空格合并为一个，去除首尾空格
    if not text:
        return []

    # -------------------------- 2. 正则分割：按中英文句子标点拆分，保留语义边界 --------------------------
    # 匹配中英文句子结束标点（可选后续空格），如 "。 "、"! "、"；"
    sentence_sep_pattern = re.compile(r'([.?!。？！；;，,])\s*')
    # 分割逻辑：保留标点在片段末尾，如 "我很好。" 而非 "我很好" + "。"
    raw_segments = []
    last_pos = 0
    for match in sentence_sep_pattern.finditer(text):
        sep = match.group(1)  # 匹配到的标点（如"。"、"!"）
        current_end = match.end()  # 匹配结束位置
        # 提取当前片段（从上次结束到当前标点结束）
        segment = text[last_pos:current_end].strip()
        if segment:  # 过滤空片段
            raw_segments.append(segment)
        last_pos = current_end
    # 处理最后一个没有标点的片段
    final_segment = text[last_pos:].strip()
    if final_segment:
        raw_segments.append(final_segment)

    # -------------------------- 3. 缓存token信息：减少重复tokenize，提升效率 --------------------------
    # 存储 (segment, token_list, token_count)，避免后续重复计算
    segment_token_info: List[Tuple[str, List[str], int]] = []
    for seg in raw_segments:
        tokens = tokenizer.tokenize(seg)
        token_count = len(tokens)
        segment_token_info.append((seg, tokens, token_count))

    # -------------------------- 4. 切割长片段：基于token列表精确切割，确保不超max_length --------------------------
    # 先处理所有片段，将超长大片段切割为符合要求的子片段
    cut_segments_token_info: List[Tuple[str, List[str], int]] = []
    for seg, tokens, token_count in segment_token_info:
        if token_count <= max_length:
            # 片段长度合格，直接加入
            cut_segments_token_info.append((seg, tokens, token_count))
        else:
            # 片段超长：按max_length切割token列表，再转回文本
            for i in range(0, token_count, max_length):
                # 切割token子列表（左闭右开，不超过max_length）
                sub_tokens = tokens[i:i+max_length]
                # 将token子列表转回文本（skip_special_tokens避免引入[CLS]等）
                sub_seg = tokenizer.decode(
                    tokenizer.convert_tokens_to_ids(sub_tokens),
                    skip_special_tokens=True
                ).strip()
                if sub_seg:  # 过滤空子片段
                    cut_segments_token_info.append((sub_seg, sub_tokens, len(sub_tokens)))

    # -------------------------- 5. 合并过短片段：基于token数缓存，兼顾紧凑性与语义 --------------------------
    optimized_segments = []
    if not cut_segments_token_info:
        return optimized_segments

    # 初始化当前合并块（用第一个片段启动）
    current_seg, current_tokens, current_token_count = cut_segments_token_info[0]
    min_token_count = max_length * min_merge_ratio  # 过短阈值（如max_length=100，阈值=30）

    for seg, tokens, token_count in cut_segments_token_info[1:]:
        # 检查合并后是否超过max_length，且当前块是否过短
        if (current_token_count + token_count) <= max_length and current_token_count < min_token_count:
            # 合并：文本拼接（加空格避免中英文粘连，如"我爱你你爱我"）
            merged_seg = f"{current_seg} {seg}".strip()
            # 合并token列表和token数（直接复用缓存，无需重新tokenize）
            merged_tokens = current_tokens + tokens
            merged_token_count = current_token_count + token_count
            # 更新当前合并块
            current_seg, current_tokens, current_token_count = merged_seg, merged_tokens, merged_token_count
        else:
            # 无法合并：添加当前块，启动新块
            optimized_segments.append(current_seg)
            current_seg, current_tokens, current_token_count = seg, tokens, token_count

    # 添加最后一个合并块
    optimized_segments.append(current_seg)

    # -------------------------- 6. 最终过滤：确保无空片段，且所有片段符合token数要求 --------------------------
    # 二次校验（避免极端情况，如decode后为空）
    final_segments = []
    for seg in optimized_segments:
        if not seg:
            continue
        # 最终校验token数（确保万无一失）
        final_token_count = len(tokenizer.tokenize(seg))
        if final_token_count <= max_length:
            final_segments.append(seg)
        else:
            # 极端情况：重新切割（理论上前面已处理，此处为兜底）
            tokens = tokenizer.tokenize(seg)
            for i in range(0, len(tokens), max_length):
                sub_seg = tokenizer.decode(
                    tokenizer.convert_tokens_to_ids(tokens[i:i+max_length]),
                    skip_special_tokens=True
                ).strip()
                if sub_seg:
                    final_segments.append(sub_seg)

    return final_segments

def merge_short_to_long(embedding_len: list[int], inversed_perturbed_sequences: list[str], tokenizer, max_num_tokens) -> list[str]:
    """
    根据embedding_len的规则，将小短句列表合并为长句列表
    - embedding_len[i]：第i个长句由多少个小短句组成
    - inversed_perturbed_sequences：所有小短句的平铺列表（需与embedding_len总长度匹配）
    
    Args:
        embedding_len: 长句-短句数量映射列表（如[2,3]表示第1长句含2短句，第2长句含3短句）
        inversed_perturbed_sequences: 小短句平铺列表（如["你好", "世界", "今天", "天气", "很好"]）
    
    Returns:
        list[str]: 合并后的长句列表（长度与embedding_len一致）
    
    Raises:
        ValueError: 若小短句总数量与embedding_len求和不匹配（可选严格校验）
    """
    # 1. 数据合法性校验（核心：确保小短句总数足够）
    total_needed = sum(embedding_len)  # 按规则需要的小短句总数
    total_provided = len(inversed_perturbed_sequences)  # 实际提供的小短句总数

    # 严格校验（可选：若需强制匹配，启用此段；若需容错，注释并改用下方兼容逻辑）
    if total_needed != total_provided:
        raise ValueError(
            f"Segment number doesn't match! {total_needed} are needed bu only get {total_provided}"
        )

    # 2. 按规则提取并合并小短句
    merged_long_sentences = []
    current_pos = 0  # 跟踪当前提取到的小短句位置

    for short_count in embedding_len:
        # 提取当前长句所需的所有小短句（连续切片）
        short_sentences = inversed_perturbed_sequences[current_pos : current_pos + short_count]
        
        # 清理小短句：去除首尾空格、过滤空字符串（避免拼接后出现多余空格）
        cleaned_shorts = [s.strip() for s in short_sentences if s.strip()]
        
        # 合并：用单个空格连接（避免“短句1”和“短句2”变成“短句1  短句2”）
        long_sent = " ".join(cleaned_shorts) if cleaned_shorts else ""

        # 后处理：将长句截断到至多max_num_tokens个token
        if long_sent and max_num_tokens > 0:
            # 分词
            tokens = tokenizer.tokenize(long_sent)
            # 截断
            if len(tokens) > max_num_tokens:
                truncated_tokens = tokens[:max_num_tokens]
                # 转换回文本
                long_sent = tokenizer.convert_tokens_to_string(truncated_tokens)
        
        merged_long_sentences.append(long_sent)
        current_pos += short_count  # 更新位置，准备下一个长句

    return merged_long_sentences

def get_optimized_paraphrase(perturbed_sequence):
    # print(f"*****\n[perturbed_sequence]: {perturbed_sequence}\n*****")
    return f"""Please reconstruct the following perturbed text into a coherent sentence while following these guidelines:

1. Prioritize Extractable Meaning: Identify and preserve all discernible semantic elements, including core concepts, emotional tone, and lexical features (specific words or phrases that carry meaningful intent). Even if parts are fragmented, retain as much interpretable content as possible.

2. Enhance Fluency Pragmatically: Fix grammatical issues, adjust word order, and resolve awkward phrasing to create natural-sounding English. Focus on making the text understandable while keeping modifications minimal.

3. Respect Textual Integrity: Avoid adding new information, altering the original intent, or overwriting distinctive lexical choices that can be reasonably integrated into a coherent structure. If absolute coherence is impossible, preserve the most meaningful fragments in a restructured form.

****************************************************

Following is some concrete examples:

Example 1:
Perturbed text:
</perturbed_begin>
it is based on the French-American musical Parisian sims Winslow living near Paris. Henrietta performed at the written from November 1921 to November 1923: In 1924, hermansen offered his first visit to Paris as a gift for it was stated that "Richard and Robert do a shade of ape, an instrumental ensemble that matches the fast moving musical projection it is written in a more modern and native language and free of charge than his earlier works. It is also available for non-professors
</perturbed_end>

Optimized paraphrase:
</paraphrase_begin>
The work is based on the French-American musical "Parisian Sims" by Winslow, who lived near Paris. Henrietta performed it from November 1921 to November 1923. In 1924, Hermansen offered his first visit to Paris as a gift, noting that "Richard and Robert perform a shade of ape"—an instrumental ensemble matching the fast-moving musical projection. Written in a more modern, native language and freer style than his earlier works, it is also available to non-professors.
</paraphrase_end>

Example 2:
Perturbed text:
</perturbed_begin>
After the ratification of Articles I and II, the senate looked for guidance from the board of business. The Congress reassemble with the directing the war effort, conducting diplomacy between states, and directing foreign states, directing foreign states, and so on, a person is looking at, addressing issues such as Native American lands and territorial issues, and maintaining relations with Native American settlers. Native American
</perturbed_end>

Optimized paraphrase:
</paraphrase_begin>
After the ratification of Articles I and II, the Senate sought guidance from the board of business, while Congress reassembled to direct the war effort, conduct diplomacy between states and foreign nations, address issues concerning Native American lands and territorial disputes, and maintain relations with Native American settlers.
</paraphrase_end>

Example 3:
Perturbed text:
</perturbed_begin>
You understand your concerns. You read your query carefully, and you understand your query carefully. You read your query carefully and you will be welcomed, and goodbye been described. Some symptoms are related to constipation, such as vomiting. It is unknown whether this is the case. Blocking can cause acute digestive that will be helpful to deal with moneses. A sex massage is recommended, as suggested by Mira, to avoid using laxative is also recommended to use a probiotic as well. In addition, use daily probiotics is more effective, and use microbial probiotic a coma. You should eat liquids rich in fiber, drink a soft drink with a swallowing edge, and drink carefully further and ask yourself again. If you thought about it, you might have answered differently. If you thought about it, your answer was absolutely applicable. of all, including her, Dr. Dorina Gabbi Presiding, Physician, Mrs. Physician,
</perturbed_end>

Optimized paraphrase:
</paraphrase_begin>
You have expressed concerns about symptoms that may be related to constipation, such as vomiting, though it's unclear if that is the case.
Blocking can cause acute digestive issues, and it may help to manage these symptoms.
A sensual massage, as suggested by Mira, is recommended.
It's also advised to avoid laxatives and instead use a probiotic; daily probiotic use is more effective.
You should consume liquid-rich fiber, drink soft beverages carefully, and swallow with caution.
Reflect on your situation — had you thought it through, you might have responded differently.
Regarding all involved, including Dr. Dorina Gabbi, Presiding Physician, and Mrs. Physician.
</paraphrase_end>

****************************************************

Perturbed sequence to process. Wrap with </paraphrase_begin> and </paraphrase_end>.

Perturbed text:
</perturbed_begin>
{perturbed_sequence}
</perturbed_end>

Optimized paraphrase:
"""

def get_input_from_response_context(answer):
    return f"""# Task: Generate Instruction-Tuning Input from Response
## Task Definition
"Input" = user's request (consultation with background, task instruction, or focused question); "Response" = targeted reply.  
Goal: Reverse-engineer a logical, real-scenario Input that elicits all key info in the Response (no gaps/extra content).

********************
    
### Example 1: Health Consultation
Response: 
</response_begin>
Your left thigh "electric shock" + tear feeling during right knee lifts is likely lumbar nerve entrapment (muscle injuries cause dull aches). Verify with McKenzie maneuver (online). Do core/spinal exercises; avoid backward bends. MRI if pain lasts >1 week.
</response_end>

Input: 
</input_begin>
When shadow boxing—bending back + lifting right knee—I got a left thigh electric shock + muscle tear feeling. What's the cause? How to check nerve/muscle? How to relieve, or when to get checks?
</input_end>

********************

### Example 2: Medical Knowledge Question
Response: 
</response_begin>
Candida vulvovaginitis: 75% women get it once. Risks: antibiotics, hormones, diabetes, tight underwear. Symptoms: itching, cottage-cheese discharge, painful urination/intercourse, soreness (worse pre-menstrual).
</response_end>

Input: 
</input_begin>
Tell me Candida-induced vulvovaginitis: how common/risky, and its main symptoms?
</input_end>

********************

### Example 3: Economics Analysis Instruction
Response: 
</response_begin>
Fiscal vs Monetary Policy:  
- Fiscal (spending/tax): Direct, boosts demand fast but slow to pass + adds debt (e.g., 2009 U.S. stimulus: $831B, +1.5-2% GDP, more debt).  
- Monetary (rates/supply): Fast to adjust, no debt but indirect (e.g., 2020 Fed: 0% rates, $700B QE, stabilized markets but caused later inflation).
</response_end>

Input: 
</input_begin>
Explain fiscal/monetary policy tradeoffs. Add examples: when used, conditions, effects, and outcomes.
</input_end>

********************

### Example 4: Business Innovation Instruction
Response: 
</response_begin>
3 running shoe ideas:  
1. Gait-Adaptive: Pressure foam adjusts density, cuts joint impact 20%.  
2. Smart Coaching: Sensors track form, app gives reports to avoid injury.  
3. Recyclable: 100% separable materials, take-back program, wear indicator.
</response_end>

Input: 
</input_begin>
I'm in a running shoe company—give 3 innovation ideas (solve runner pain points, clear selling points).
</input_end>

********************

Generate an Input for the Response below. Wrap with </input_begin> and </input_end>.

Response: 
</response_begin>
{answer}
</response_end>

Input:
"""

def get_extract_final_response_func(begin_tag:str, end_tag:str):

    def extract_final_response(generated_output):        
        # 1. 找到最后一个 [Question_begin] 的起始索引
        last_begin_idx = generated_output.rfind(begin_tag)
        if last_begin_idx == -1:  # 若没有找到起始标记，返回空字符串
            return generated_output.replace(begin_tag, "").replace(end_tag, "")
        
        # 2. 从最后一个 [Question_begin] 之后，找第一个 [Question_end] 的起始索引
        # 计算起始标记结束后的位置
        begin_end_pos = last_begin_idx + len(begin_tag)
        last_end_idx = generated_output.find(end_tag, begin_end_pos)
        if last_end_idx == -1:  # 若起始标记后没有找到结束标记，返回空字符串
            return generated_output.replace(begin_tag, "").replace(end_tag, "")
        
        # 3. 截取并返回中间内容
        return generated_output[begin_end_pos:last_end_idx].replace('\n', '')
    
    return extract_final_response

def generated_from_surrogate_model_with_engine(sequences: List[str], 
                                                engine: 'InferEngine', 
                                                max_tokens:int=2048,
                                                post_process:Callable=None,
                                                **kwargs) -> List[str]:
    # prepare the model input
    # prompt = "Give me a short introduction to large language model."
    # 确保输入是列表，如果是单个字符串则转为列表
    if isinstance(sequences, str):
        sequences = [sequences]
    
    # Prepare InferRquest
    infer_requests = []
    for idx, seq in tqdm(enumerate(sequences), desc="Construct InferRequest"):
        data_new = {}
        data_new['messages'] = []
        dict = {}
        dict['role'] = 'user'
        dict['content'] = seq

        data_new['messages'].append(dict)
        infer_requests.append(InferRequest(**data_new))

    generated_data = []
    request_config = RequestConfig(max_tokens=max_tokens, temperature=1.0)
    metric = InferStats()
    resp_list = engine.infer(infer_requests, request_config, metrics=[metric])
    for index, response in enumerate(resp_list):
        content = response.choices[0].message.content
        if post_process:
            content = post_process(content)
        generated_data.append(content)

    return generated_data

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

# Step 0.0 define hyper-parameters
device = 'cuda'
seed = 42
use_peft = True
load_8bit = False
verbose = True

use_alpaca_module_template = False

base_model = './pretrained/pythia-2.8b'
# base_model = './pretrained/qwen-2.5-1.5b'

#### PEFT weights for pythia-2.8b
peft_weights = None

inversion_model_path = "./pretrained/vec2text-v2/gtr__nq__32/"
corrector_model_path = "./pretrained/vec2text-v2/gtr__nq__32__correct/"
encoder_path = "/home/huangzr/LLM-MIA/pretrained/gtr-t5-base/"

save_path = None

# save_key = 'generated_output'
save_key = 'text'

# effective for alpaca-* datasets
# perturbation_magnitude = 0.5
# perturbation_magnitude_delta = 0.01

perturbation_magnitude = 0.5
perturbation_magnitude_delta = 0.01

# parameters for genreating text
cutoff_length=256
batch_size=16
top_k=30 # 30 is a appropriate top_k number here 
max_new_tokens=256
min_tokens=16
temperature=0.7
top_p=0.9
repetition_penalty=1.2

num_sample_texts = 100
num_start_up_texts = 100
# num_total_outputs = 500
num_total_outputs = 2000 
# num_total_outputs = 10000 
# min_output_length = 108 # At least 30 tokens (3.6 characters per token)
# min_output_length = 36 # At least 10 tokens (3.6 characters per token)
min_output_length = 54 # At least 15 tokens (3.6 characters per token)

### Assume the instruction is prepared by the server, the user only inputs 
### Specified for Chatdoctor
instruction_content = "If you are a doctor, please answer the medical questions based on the patient's description."
instruction_content_for_start = "If you are a doctor, please answer the medical questions based on the patient's description."
start_input = " "
### Specified for Alpaca-med
instruction_content = f"Answer this question truthfully."
instruction_content_for_start = f"Answer this question truthfully."
start_input = " "
max_num_tokens = 20 # For Causal language Modeling, use short max_num_tokens as prompt

surrogate_model_path = "./pretrained/Qwen3-4B-Instruct/"

def remove_punctuation(text:str)->str:
    # 创建标点符号转换表（key:标点符号, value:None）
    translator = str.maketrans('', '', string.punctuation)
    # 删除所有标点符号（同时保留中文字符）
    return text.translate(translator)

def generate_output(base_model:str=base_model,
                    peft_weights:str=peft_weights,
                    surrogate_model_path:str=surrogate_model_path,
                    save_path:str=save_path,
                    num_total_outputs:int=num_total_outputs,
                    use_alpaca_module_template:bool=use_alpaca_module_template,
                    save_key:str=save_key,
                    top_k:int=top_k,
                    min_output_length=min_output_length,
                    load_8bit:bool=load_8bit,
                    load_bnb_4bit:bool=False,
                    seed:int=seed,
                    do_sample:bool=True,
                    tensor_parallel_size:int=1,
                    surrogate_model_type:str='qwen3',
                    num_sample_texts = 100,
                    num_start_up_texts = 100,
                    surrogate_surrogate_gpu_utility: float=0.3,
                    embedding_batch_size:int = 100,
                    max_surrogate_model_len:int = 4096,
                    ):
    
    set_random_seed(seed)
    
    print(base_model, peft_weights, save_path, use_alpaca_module_template, surrogate_model_path)
    print(num_total_outputs, num_sample_texts, num_start_up_texts)

    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    # Step 1 get models 
    ### corrector, encoder, encoder_tokenizer must be loaded first, otherwise wierd bugs can happen, that is vec2text fails to work.
    corrector = get_vec2text_model(inversion_model_path=inversion_model_path,
                                corrector_model_path=corrector_model_path,
                                verbose=verbose) 

    import transformers
    encoder = transformers.AutoModel.from_pretrained(encoder_path).encoder.to("cuda")
    encoder_tokenizer = transformers.AutoTokenizer.from_pretrained(encoder_path)
    print(f"#### Get the encoder from {encoder_path}. ####")

    model, tokenizer = get_target_model(base_model=base_model, peft_weights=peft_weights, use_peft=use_peft, 
                                        load_8bit=load_8bit, device=device, load_bnb_4bit=load_bnb_4bit)
    print(f"#### Get target model from {base_model}+{peft_weights}. ####")
    

    # Get surrogate model
    # 避免CUDA out of memory
    engine = VllmEngine(surrogate_model_path, model_type=surrogate_model_type, 
                        max_model_len=max_surrogate_model_len,
                        gpu_memory_utilization=surrogate_surrogate_gpu_utility,tensor_parallel_size=tensor_parallel_size)

    print(f"#### Get surrogate model from {surrogate_model_path}. ####")
        
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    # step 2 generate start-up outputs with empty queries

    start_up_texts = []
    prompt_results_list = []
    if use_alpaca_module_template:
        prompt = get_alpaca_prompt_module(set_input=True, input_content=start_input, instruction_content=instruction_content_for_start)
        prompt_results_list = [prompt] * num_sample_texts
    else:
        prompt_results_list = random.choices(
            population=get_conjunctive_strings(),  # 采样源：conjunctive_strings 数组
            k=num_sample_texts              # 采样数量：生成 num_sample_texts 个元素
        )

    while len(start_up_texts) < num_start_up_texts:
        num_needed = num_start_up_texts-len(start_up_texts)
        # prompts = random.sample(prompt_results_list, min(num_needed, len(prompt_results_list)))
        indices = random.sample(range(len(prompt_results_list)), min(num_needed, len(prompt_results_list)))
        prompts = [prompt_results_list[i] for i in indices]
        
        outputs = generated_output_from_target_model(sequences=prompts, model=model, tokenizer=tokenizer,
                                                    cutoff_length=cutoff_length, batch_size=batch_size,
                                                    device=device, top_k=top_k,
                                                    max_new_tokens=max_new_tokens, min_tokens=min_tokens,
                                                    temperature=temperature, top_p=top_p,
                                                    repetition_penalty=repetition_penalty,
                                                    use_alpaca_module_template=use_alpaca_module_template,
                                                    do_sample=True) ### Here must introduce randomness
                                                    # do_sample=do_sample)
        
        if use_alpaca_module_template:
            outputs = [{'instruction': instruction_content_for_start, 'input': start_input, save_key: output} for output in outputs if len(remove_punctuation(output)) > min_output_length]
        else:
            outputs = [{save_key: output} for output in outputs if len(remove_punctuation(output)) > min_output_length]
                        
        start_up_texts.extend(outputs)
        
        if verbose:
            print(f"The current length of start_up_texts is {len(start_up_texts)}", end='\r')
            sys.stdout.flush()  

    if verbose: 
        print(f"The current length of start_up_texts is {len(start_up_texts)}")

    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    # step 3 generated more outputs from perturbed embeddings

    final_texts = start_up_texts

    # prompt_questions = get_prompt_questions(domain_word=alpaca_domain_word) # only used for alpaca instruction tuning

    random_idx = 0
    while len(start_up_texts) < num_total_outputs:
        # random.seed(seed + random_idx)
        set_random_seed(seed + random_idx)
        
        seed_sequences = random.sample(final_texts, min(num_sample_texts, len(final_texts)))
        random_idx = random_idx + 1
        
        seed_sequences = seed_sequences[:num_total_outputs-len(start_up_texts)]
        
        seed_sequences = [seed_sequence[save_key] for seed_sequence in seed_sequences]

        embeddings_list = get_gtr_embeddings(seed_sequences, encoder, encoder_tokenizer, 
                                             max_length=32,
                                             max_seg_num=0 if use_alpaca_module_template else 10)
        embedding_len = [len(e) for e in embeddings_list]

        embeddings = torch.cat(embeddings_list, dim=0)
        perturbed_embeddings = add_random_perturbation(embeddings, perturbation_magnitude, perturbation_magnitude_delta)

        # 按批次处理embeddings
        inversed_perturbed_sequences = []
        total = len(perturbed_embeddings)  # 获取总样本数

        # 计算总批次（向上取整）
        num_batches = (total + embedding_batch_size - 1) // embedding_batch_size

        for i in tqdm(range(num_batches), desc='Inverse perturbed sequences'):
            # 计算当前批次的起始和结束索引
            start_idx = i * embedding_batch_size
            end_idx = min((i + 1) * embedding_batch_size, total)
            
            # 提取当前批次的embeddings
            batch_embeddings = perturbed_embeddings[start_idx:end_idx]
            
            # 批次处理
            batch_results = vec2text.invert_embeddings(
                embeddings=batch_embeddings,
                corrector=corrector,
                num_steps=20,
            )
            
            # 将批次结果添加到总列表
            inversed_perturbed_sequences.extend(batch_results)

        inversed_perturbed_sequences = merge_short_to_long(embedding_len, inversed_perturbed_sequences, tokenizer, max_num_tokens)

        paraphrased_prompts = [get_optimized_paraphrase(seq) for seq in inversed_perturbed_sequences]

        optimized_inversed_perturbed_sequences = generated_from_surrogate_model_with_engine(sequences=paraphrased_prompts, 
                                                                          engine=engine, 
                                                                          do_sample=True,
                                                                          post_process=get_extract_final_response_func(begin_tag="</paraphrase_begin>", end_tag="</paraphrase_end>"))

        

        prompts = []
        input_contents = [] # useless when use_alpaca_prompt_tempalate is False 
        if use_alpaca_module_template:
            in_context_sequences = [get_input_from_response_context(seq) for seq in optimized_inversed_perturbed_sequences]
            # in_context_sequences = [get_question_conversion_context(seq) for seq in optimized_inversed_perturbed_sequences]

            questions = generated_from_surrogate_model_with_engine(sequences=in_context_sequences, 
                                                                            engine=engine, 
                                                                            do_sample=False,
                                                                            post_process=get_extract_final_response_func(begin_tag="</input_begin>", end_tag="</input_end>"))
            for question in questions:
                # construct_alpaca_prompt_input is too complicated, and fails at medflash dataset.
                # input_content = construct_alpaca_prompt_input(question, background=sequence)
                prompt = get_alpaca_prompt_module(set_input=True, input_content=question, instruction_content=instruction_content)
                prompts.append(prompt)
                input_contents.append(question)
        else:
            prompts = optimized_inversed_perturbed_sequences
        
        outputs = generated_output_from_target_model(sequences=prompts, model=model, tokenizer=tokenizer,
                                                    cutoff_length=cutoff_length, batch_size=batch_size,
                                                    device=device, top_k=top_k,
                                                    max_new_tokens=max_new_tokens, min_tokens=min_tokens,
                                                    temperature=temperature, top_p=top_p,
                                                    repetition_penalty=repetition_penalty,
                                                    use_alpaca_module_template=use_alpaca_module_template,
                                                    do_sample=do_sample)
        
        if use_alpaca_module_template:
            outputs = [{'instruction': instruction_content, 'input': input_content, save_key: output} for input_content, output in zip(input_contents, outputs) if len(remove_punctuation(output)) > min_output_length]
        else:
            outputs = [{save_key: output} for output in outputs if len(remove_punctuation(output)) > min_output_length]
        
        final_texts.extend(outputs)
                
        if verbose:
            print(f"The current length of final_texts is {len(final_texts)}", end='\r')
            sys.stdout.flush()  
            
    if verbose: 
        print(f"The current length of start_up_texts is {len(final_texts)}")

    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

    # Step 4 save the generated results
    save_json_dataset(final_texts, save_path=save_path)
    
if __name__ == "__main__":
    import fire
    # fire.Fire()
    # generate_output(peft_weights=peft_weights, save_path=save_path)
    fire.Fire(generate_output)