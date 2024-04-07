import os
os.environ["CUDA_LAUNCH_BLOCKING"] = '1'
os.environ["TORCH_USE_CUDA_DSA"] = '1'
import torch
from typing import Optional
from functools import partial
from peft import LoraConfig, TaskType, get_peft_model, PeftModel

from trainer import MultiModalTrainer
from model.model import MMultiModal, LanguageConfig, VisualConfig, MultiModalConfig
from dataset.image_caption_dataset import ImageCaptionDataset, data_collate

import transformers
from transformers import HfArgumentParser, AutoTokenizer, ChineseCLIPProcessor
from dataclasses import dataclass, field

from qwen.modeling_qwen import QWenLMHeadModel
from qwen.qwen_generation_utils import make_context

@dataclass
class FinetuneArguments:
    lora_rank: int = field(default=8)
    lora_dropout: float = field(default=0.1)
    previous_lora_weights: str = field(default=None) 
    target_modules: str = field(default="W_pack") 
    image_map: str = field(default="./data/image_map_b.json", metadata={"help": "图像文件与索引ID"})
    captions_file: str = field(default="./data/captions_b.json", metadata={"help": "ID与caption的对应"})

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    feature_proj_lr: Optional[float] = None

def train():
    finetune_args, training_args = HfArgumentParser(
        (FinetuneArguments, TrainingArguments)
    ).parse_args_into_dataclasses()

    base_language_model = "~/project/data2/liu_project/huggingface_model/qwen/Qwen-7B-Chat"
    base_value_model = "~/project/data2/liu_project/huggingface_model/clip-vit-large-patch14"
    # base_language_model = "F:/huggingface_model/qwen/Qwen-7B-Chat"
    # base_value_model = "F:/huggingface_model/clip-vit-large-patch14"

    tokenizer = AutoTokenizer.from_pretrained(base_language_model, trust_remote_code=True)
    replace_token_id = tokenizer.convert_tokens_to_ids("<|extra_0|>")

    
    model = MMultiModal(LanguageConfig(model_path=base_language_model), VisualConfig(model_path=base_value_model), 
                        MultiModalConfig(replace_token_id=replace_token_id), finetune_args, train=True).cuda()
    model.train()
    model.LLM.config.use_cache = False

    

    dataset = ImageCaptionDataset(tokenizer, finetune_args.image_map, finetune_args.captions_file,  VisualConfig(model_path=base_value_model), max_train_data_item=300000)

    print(training_args)

    trainer = MultiModalTrainer(model=model,
                                data_collator=partial(data_collate, tokenizer=tokenizer, black_token_length = MultiModalConfig.image_context_length),
                                train_dataset=dataset,
                                args=training_args)
    trainer.train()

def main():
    train()

if __name__ == "__main__":
    main()

