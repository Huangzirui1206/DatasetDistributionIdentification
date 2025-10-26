'''
Acknowledgement:

(NIPS'24) PrivAuditor: Benchmarking Privacy Vulnerabilities in LLMAdaptation Techniques 
Reference: https://github.com/TrustAI-Open/PrivAuditor
Paper: https://openreview.net/pdf?id=VpkfxuVXwx 

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

from typing import List

import fire
import torch
import transformers
from datasets import load_dataset
from typing import List, Optional, Union

from peft import (  # noqa: E402
    LoraConfig,
#    BottleneckConfig,
    PrefixTuningConfig,
    PromptTuningConfig,
    PromptEncoderConfig,
    PromptTuningInit,
    TaskType,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
    # prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, LlamaTokenizer, AutoModel  # noqa: F402

here = os.path.dirname(__file__)
sys.path.append(os.path.join(here, '..'))
from utils.prepare_dataset import dataset_prepare
# Directly define packing dataset here

from trl import SFTTrainer, SFTConfig
from accelerate import Accelerator

# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from swift.llm import get_model_info_meta
from swift.utils import find_all_linears, get_model_parameter_info

def train(
        # model/data params
        base_model: str = "",  # the only required argument
        data_path: str = "../dataset/alpaca-cleaned",
        output_dir: str = "../ckpt/lora-alpaca",
        adapter_name: str = "lora",
        load_8bit : bool = False,
        load_bnb_4bit: bool = False,
        # training hyperparams
        batch_size: int = 128, # batch_size=128 is too large for wiki-2
        micro_batch_size: int = 4, # micro_batch_size=4 is too large for wiki-2
        num_epochs: int = 3,
        learning_rate: float = 3e-4,
        cutoff_len: int = 256,
        randomize_train_set: bool=False,
        train_set_size: int = 0,
        train_set_start: int = 0,
        val_set_size: int = 2000,
        use_gradient_checkpointing: bool = False,
        eval_step: int = 200,
        save_step: int = 200,
        # lora hyperparams
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: List[str] = None,
        # bottleneck adapter hyperparams
        bottleneck_size: int = 256,
        non_linearity: str = "tanh",
        adapter_dropout: float = 0.0,
        use_parallel_adapter: bool = False,
        use_adapterp: bool = False,
        target_modules: List[str] = None,
        scaling: Union[float, str] = 1.0,
        # prefix/prompt/p- tuning hyperparams
        num_virtual_tokens: int = 128,
        # llm hyperparams
        train_on_inputs: bool = True,  # if False, masks out inputs in loss
        group_by_length: bool = False,  # faster, but produces an odd training loss curve
        # wandb params
        wandb_project: str = "",
        wandb_run_name: str = "",
        wandb_watch: str = "",  # options: false | gradients | all
        wandb_log_model: str = "",  # options: false | true
        resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
        # whether use data_prepare
        use_data_prepare:bool = True,
        # validation_split_percentage
        validation_split_percentage:float=0.001,
        # whether to use packing
        use_packing:bool = True,
):  
    print(
        f"Finetuning model with params:\n"
        f"base_model: {base_model}\n"
        f"data_path: {data_path}\n"
        f"output_dir: {output_dir}\n"
        f"batch_size: {batch_size}\n"
        f"micro_batch_size: {micro_batch_size}\n"
        f"num_epochs: {num_epochs}\n"
        f"learning_rate: {learning_rate}\n"
        f"cutoff_len: {cutoff_len}\n"
        f"randomize_train_set: {randomize_train_set}\n"
        f"train_set_size: {train_set_size}\n"
        f"train_set_start: {train_set_start}\n"
        f"val_set_size: {val_set_size}\n"
        f"use_gradient_checkpointing: {use_gradient_checkpointing}\n"
        f"lora_r: {lora_r}\n"
        f"lora_alpha: {lora_alpha}\n"
        f"lora_dropout: {lora_dropout}\n"
        f"lora_target_modules: {lora_target_modules}\n"
        f"bottleneck_size: {bottleneck_size}\n"
        f"non_linearity: {non_linearity}\n"
        f"adapter_dropout: {adapter_dropout}\n"
        f"use_parallel_adapter: {use_parallel_adapter}\n"
        f"use_adapterp: {use_adapterp}\n"
        f"num_virtual_tokens: {num_virtual_tokens}\n"
        f"train_on_inputs: {train_on_inputs}\n"
        f"scaling: {scaling}\n"
        f"adapter_name: {adapter_name}\n"
        f"target_modules: {target_modules}\n"
        f"group_by_length: {group_by_length}\n"
        f"wandb_project: {wandb_project}\n"
        f"wandb_run_name: {wandb_run_name}\n"
        f"wandb_watch: {wandb_watch}\n"
        f"wandb_log_model: {wandb_log_model}\n"
        f"resume_from_checkpoint: {resume_from_checkpoint}\n"
        f"load_bnb_4bit: {load_bnb_4bit}\n"
        f"use_packing: {use_packing}\n"
    )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"
    gradient_accumulation_steps = batch_size // micro_batch_size
    
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or (
            "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    if load_8bit:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map=device_map,
            trust_remote_code=True,
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
            device_map={"": int(os.environ.get("LOCAL_RANK") or 0)},  # 分布式训练设备映射
            trust_remote_code=True,  # 信任自定义代码
            # 以下参数需要调整或删除：
            # load_in_8bit=False,  # 已由 quantization_config 覆盖
            # torch_dtype=torch.float16  # 已由 bnb_4bit_compute_dtype 覆盖
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=False,
            torch_dtype=torch.float16,
            device_map={"": int(os.environ.get("LOCAL_RANK") or 0)},
            trust_remote_code=True,
        )

    # if model.config.model_type == "llama":
        # Due to the name of transformers' LlamaTokenizer, we have to do this
    #     tokenizer = LlamaTokenizer.from_pretrained(base_model)
    # else:
    #     tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    model_info, model_meta = get_model_info_meta(base_model)

    tokenizer.model_info = model_info
    tokenizer.model_meta = model_meta

    model.model_info = model_info
    model.model_meta = model_meta
    
    # THIS IS A HACK TO GET THE PAD TOKEN ID NOT TO BE EOS
    if tokenizer.pad_token_id is None:
        print("Pad token id is None, setting to eos token id...")
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # tokenizer.padding_side = "left"  # Allow batched inference
    '''
    the padding side should be left when generating and right when training/tuning” (setting of padding_side in Llama tokenizers · Issue #34842 · huggingface/transformers · GitHub
    '''
    tokenizer.padding_side = "right"
    
    if use_data_prepare:
        print("!!!!! WARNING: NEED FURTHER CHANGE dataset_prepare PART !!!!!")
        train_data, val_data = dataset_prepare(
            dataset_name=data_path,
            dataset_config_name=None,
            validation_split_percentage=validation_split_percentage,
            packing=use_packing,
            cache_path='./cache',
            use_dataset_cache=False,
            preprocessing_num_workers=1,
            tokenizer=tokenizer,
            block_size=cutoff_len,
            num_of_sequences=1024, 
            chars_per_token=3.6,
            train_set_size=train_set_size,
        )
    else:
        if data_path.endswith(".json"):  # todo: support jsonl
            data = load_dataset("json", data_files=data_path)
        else:
            data = load_dataset(data_path)
        train_val = data["train"].train_test_split(
            test_size=val_set_size, shuffle=randomize_train_set, seed=42
        )
        train_data = train_val["train"]
        val_data = train_val["test"]
        
    def preprocess_function(examples):
        tokenized = tokenizer(
            examples["text"],  # 或其他包含文本的列名
            padding=True,  # 自动填充到批次最大长度
            truncation=True,  # 截断到模型最大长度
            max_length=cutoff_len,  # 模型支持的最大长度
            return_tensors="pt"  # 返回 PyTorch 张量
        )
        
        # Add causal mask for each sample in the batch
        # batch_size, seq_len = tokenized['input_ids'].shape
        # causal_mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).expand(batch_size, -1, -1)
        # tokenized["causal_mask"] = causal_mask
        
        tokenized["labels"] = tokenized["input_ids"].detach().clone()
        
        # Log shapes to debug potential issues
        # print(f"Input IDs shape: {tokenized['input_ids'].shape}")
        # print(f"Causal mask shape: {tokenized['causal_mask'].shape}")
        # print(f"Labels shape: {tokenized['labels'].shape}")
        
        return tokenized

    train_data = train_data.map(preprocess_function, batched=True)
    val_data = val_data.map(preprocess_function, batched=True)
    
    # print(train_data)
    # print(val_data)
    # assert 0
        
    # model = prepare_model_for_int8_training(model, use_gradient_checkpointing=use_gradient_checkpointing)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=use_gradient_checkpointing)
    if adapter_name == "lora":
        target_modules = find_all_linears(model)
        config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        # sys.exit("LoRA is not supported, but not available in this version")
    elif adapter_name == "bottleneck":
        """config = BottleneckConfig(
            bottleneck_size=bottleneck_size,
            non_linearity=non_linearity,
            adapter_dropout=adapter_dropout,
            use_parallel_adapter=use_parallel_adapter,
            use_adapterp=use_adapterp,
            target_modules=target_modules,
            scaling=scaling,
            bias="none",
            task_type="CAUSAL_LM",
        )"""
        sys.exit("Bottleneck adapter is not supported in this version")
    elif adapter_name in ["prefix-tuning", "p-tuning-v2"]:
        config = PrefixTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=num_virtual_tokens,
            prefix_projection=False if adapter_name == "p-tuning-v2" else True,
        )
    elif adapter_name == "prompt-tuning":
        config = PromptTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            prompt_tuning_init=PromptTuningInit.TEXT,
            num_virtual_tokens=num_virtual_tokens,
            prompt_tuning_init_text="Complete the following task:\n\n",
            tokenizer_name_or_path=base_model)
    elif adapter_name == "p-tuning":
        config = PromptEncoderConfig(task_type=TaskType.CAUSAL_LM, 
                                     num_virtual_tokens=num_virtual_tokens, 
                                     encoder_hidden_size=128)
    else:
        raise ValueError(f"Adapter {adapter_name} not supported")
        
    model = get_peft_model(model, config)
    if adapter_name == "prefix-tuning" or adapter_name == "prompt-tuning":
        model.to('cuda')
        
    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = (
                False  # So the trainer won't try loading its state
            )
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            model = set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    print("save_steps={}".format(save_step))
    
    # 显示正在使用的 GPU 数量
    print(f"Training with {Accelerator().num_processes} GPUs")
    
    # 初始化 SFTConfig
    sft_config = SFTConfig(
        dataset_text_field="text",  # 设置数据集中的文本字段
        max_seq_length=512,  # 设置最大序列长度
        do_train=True,
        do_eval=True,
        output_dir=output_dir,  # 使用原来的 output_dir
        dataloader_drop_last=True,
        eval_strategy="steps" if val_set_size > 0 else "no",  # 动态设置评估策略
        save_strategy="steps",
        logging_strategy="steps",
        num_train_epochs=num_epochs,  # 使用原来的 num_epochs
        eval_steps=eval_step if val_set_size > 0 else None,  # 使用 eval_step
        save_steps=save_step,  # 使用 save_step
        logging_steps=10,  # 固定为每 10 步记录一次日志
        per_device_train_batch_size=micro_batch_size,  # 对应 micro_batch_size
        per_device_eval_batch_size=micro_batch_size * 2,  # 评估批次大小为训练的两倍
        optim="adamw_torch",  # 保持原来的优化器配置
        learning_rate=learning_rate,  # 使用原来的学习率
        lr_scheduler_type="linear",  # 假设线性学习率调度器
        warmup_steps=0,  # 使用原来的 warmup_steps
        gradient_accumulation_steps=gradient_accumulation_steps,  # 梯度累积步数
        gradient_checkpointing=False,  # 默认不启用梯度检查点
        weight_decay=0.0,  # 权重衰减（未显式设置，默认 0）
        adam_epsilon=1e-6,  # AdamW 优化器的 epsilon 参数
        report_to="wandb" if use_wandb else None,  # 日志系统配置
        load_best_model_at_end=True if val_set_size > 0 else False,  # 是否加载最佳模型
        save_total_limit=3,  # 保存的最大检查点数量
        bf16=True if torch.cuda.is_bf16_supported() else False,  # 自动选择 bf16 或 fp16
        fp16=False if torch.cuda.is_bf16_supported() else True,
    )

    # Initialize SFTTrainer
    trainer = SFTTrainer(
        model=model,  # Model
        args=sft_config,  # Training parameters
        train_dataset=train_data,  # Training data
        eval_dataset=val_data,  # Validation data
        # The data collator setup without padding_side
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt"
        )
        # tokenizer=tokenizer,  # This feature is revised in newest version, need further revision in the future.
    )

    model.config.use_cache = False

    """old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(
            self, old_state_dict()
        )
    ).__get__(model, type(model))

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)"""

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    model.save_pretrained(output_dir,save_safetensors=True)

    print(
        "\n If there's a warning about missing keys above, please disregard :)"
    )



if __name__ == "__main__":
    fire.Fire(train)
