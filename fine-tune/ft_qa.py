'''
Acknowledgement:

(NIPS'24) PrivAuditor: Benchmarking Privacy Vulnerabilities in LLMAdaptation Techniques 
Reference: https://github.com/TrustAI-Open/PrivAuditor
Paper: https://openreview.net/pdf?id=VpkfxuVXwx 
'''

import os
import sys

from typing import List

import fire
import json
import torch
import random
import transformers
from datasets import load_dataset, Features, Value
from typing import List, Optional, Union

from peft import (  # noqa: E402
    LoraConfig,
    PrefixTuningConfig,
    PromptTuningConfig,
    PromptEncoderConfig,
    IA3Config,
    PromptTuningInit,
    TaskType,
    get_peft_model,
#    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
#    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, LlamaTokenizer, AutoModel  # noqa: F402

from swift.llm import get_model_info_meta
from swift.utils import find_all_linears, get_model_parameter_info

def train(
        # model/data params
        base_model: str = "",  # the only required argument
        data_path: str = "./dataset/alpaca-cleaned",
        output_dir: str = "./ckpt/lora-alpaca",
        adapter_name: str = "lora",
        load_8bit : bool = False,
        load_bnb_4bit: bool = False,
        # training hyperparams
        batch_size: int = 128,
        micro_batch_size: int = 4,
        num_epochs: int = 3,
        learning_rate: float = 3e-4, 
        cutoff_len: int = 256,
        randomize_train_set: bool = False,
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
        # IA3 hyperparameters
        i3a_dropout: float = 0.05,

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

    model_info, model_meta = get_model_info_meta(base_model, model_type='llama3_1')

    tokenizer.model_info = model_info
    tokenizer.model_meta = model_meta

    model.model_info = model_info
    model.model_meta = model_meta

    # tokenizer.pad_token_id = (
    #     0  # unk. we want this to be different from the eos token
    # )
    # tokenizer.padding_side = "left"  # Allow batched inference
    # THIS IS A HACK TO GET THE PAD TOKEN ID NOT TO BE EOS
    if tokenizer.pad_token_id is None:
        print("Pad token id is None, setting to eos token id...")
        tokenizer.pad_token_id = tokenizer.eos_token_id
    '''
    the padding side should be left when generating and right when training/tuning” (setting of padding_side in Llama tokenizers · Issue #34842 · huggingface/transformers · GitHub
    '''
    tokenizer.padding_side = "right"
        
    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
                result["input_ids"][-1] != tokenizer.eos_token_id
                and len(result["input_ids"]) < cutoff_len
                and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            if "chatglm" not in base_model:
                result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        if "chatglm" in base_model:
            return {"input_ids": result["input_ids"], "labels": result["labels"]}
        else:
            return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = generate_prompt(data_point)
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = generate_prompt({**data_point, "output": ""})
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            tokenized_full_prompt["labels"] = [
                                                  -100
                                              ] * user_prompt_len + tokenized_full_prompt["labels"][
                                                                    user_prompt_len:
                                                                    ]  # could be sped up, probably
        return tokenized_full_prompt

    #model = prepare_model_for_int8_training(model, use_gradient_checkpointing=use_gradient_checkpointing)
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
    elif adapter_name == "ia3":  # ia3 adapter配置
        if model.config.model_type == "qwen3":
            target_modules = ["q_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            feedforward_modules = ["gate_proj", "up_proj", "down_proj"]
        else:
            target_modules = None
            feedforward_modules = None
        config = IA3Config(
            target_modules = target_modules,
            feedforward_modules = feedforward_modules,
            task_type="CAUSAL_LM",
        )
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

    if data_path.endswith(".json"):  # todo: support jsonl
        data = load_dataset("json", data_files=data_path)
    else:
        data = load_dataset(data_path)
        
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

    if val_set_size > 0:
        train_val = data["train"].train_test_split(
            test_size=val_set_size, shuffle=randomize_train_set, seed=42
        )
        train_data = train_val["train"]
        if train_set_size > 0:
            if randomize_train_set:
                random_indexes = random.sample(range(len(train_data)), train_set_size)
                train_data = train_data.select(random_indexes)
                range_json = {'type':'random', 'start':0, 
                              'end':0, 'random_indexes':random_indexes}
            else:
                train_set_end = min(train_set_start + train_set_size, len(train_data))
                train_data = train_data.select(range(train_set_start, train_set_end))
                range_json = {'type':'sequence', 'start':train_set_start, 
                              'end':train_set_end, 'random_indexes':[]}
            with open(os.path.join(output_dir, 'range.json'), 'w') as json_file:
                json.dump(range_json, json_file, indent=4)
                
        train_data = (
            train_data.shuffle().map(generate_and_tokenize_prompt)
        )
        val_data = (
            train_val["test"].shuffle().map(generate_and_tokenize_prompt)
        )
    else:
        train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = None

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    print("save_steps={}".format(save_step))

    model_parameter_info = get_model_parameter_info(model)
    print(f'model_parameter_info: {model_parameter_info}')
    
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=10,
            optim="adamw_torch",
            eval_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=eval_step if val_set_size > 0 else None,
            save_steps=save_step,
            output_dir=output_dir,
            save_total_limit=3,
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to="wandb" if use_wandb else None,
            run_name=wandb_run_name if use_wandb else None,
            disable_tqdm=False,
            save_safetensors=False,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
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

    model.save_pretrained(output_dir,save_safetensors=False)

    print(
        "\n If there's a warning about missing keys above, please disregard :)"
    )


def generate_prompt(data_point):
    # sorry about the formatting disaster gotta move fast
    if data_point["input"]:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

                ### Instruction:
                {data_point["instruction"]}
                
                ### Input:
                {data_point["input"]}
                
                ### Response:
                {data_point["output"]}""" # noqa: E501
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.  

                ### Instruction:
                {data_point["instruction"]}
                
                ### Response:
                {data_point["output"]}""" # noqa: E501


if __name__ == "__main__":
    fire.Fire(train)
