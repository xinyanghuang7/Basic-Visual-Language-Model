import os
import json
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

from einops import rearrange

from accelerate import Accelerator

@dataclass
class FinetuneArguments:
    lora_rank: int = field(default=8)
    lora_dropout: float = field(default=0.1)
    previous_lora_weights: Optional[str] = field(default=None)
    target_modules: str = field(default="W_pack")
    image_map: str = field(default="data/image_map_b.json", metadata={"help": "图像文件与索引ID"})
    captions_file: str = field(default="data/captions_b.json", metadata={"help": "ID与caption的对应"})

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    feature_proj_lr: Optional[float] = None

def train():
    finetune_args, training_args = HfArgumentParser(
        (FinetuneArguments, TrainingArguments)
    ).parse_args_into_dataclasses()

    base_language_model = "Qwen/Qwen-7B-Chat"
    base_value_model = "openai/clip-vit-large-patch14"

    tokenizer = AutoTokenizer.from_pretrained(base_language_model, trust_remote_code=True)
    replace_token_id = tokenizer.convert_tokens_to_ids("<|extra_0|>")

    # Check file paths
    if not os.path.exists(finetune_args.image_map):
        raise FileNotFoundError(f"Image map file not found: {finetune_args.image_map}")

    if not os.path.exists(finetune_args.captions_file):
        raise FileNotFoundError(f"Captions file not found: {finetune_args.captions_file}")

    # Load and check file contents
    with open(finetune_args.image_map, 'r') as f:
        image_map = json.load(f)
        print(f"Image map contains {len(image_map)} entries")

    with open(finetune_args.captions_file, 'r') as f:
        captions = json.load(f)
        print(f"Captions file contains {len(captions)} entries")

    model = MMultiModal(
        LanguageConfig(model_path=base_language_model),
        VisualConfig(model_path=base_value_model),
        MultiModalConfig(replace_token_id=replace_token_id),
        finetune_args,
        train=True
    ).cuda()
    model.train()
    model.LLM.config.use_cache = False

    dataset = ImageCaptionDataset(
        tokenizer,
        finetune_args.image_map,
        finetune_args.captions_file,
        VisualConfig(model_path=base_value_model),
        max_train_data_item=300000
    )

    # Add debug information
    print(f"Dataset length: {len(dataset)}")
    if len(dataset) == 0:
        raise ValueError("The dataset is empty. Please check the dataset files and paths.")

    print(training_args)

    # Initialize Accelerator
    accelerator = Accelerator()

    # Create DataLoader
    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=training_args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=partial(data_collate, tokenizer=tokenizer, black_token_length=MultiModalConfig.image_context_length)
    )

    trainer = MultiModalTrainer(
        model=model,
        data_collator=partial(data_collate, tokenizer=tokenizer, black_token_length=MultiModalConfig.image_context_length),
        train_dataset=dataset,
        args=training_args
    )
    
    # Prepare the trainer and dataloader with the accelerator
    trainer, train_dataloader = accelerator.prepare(trainer, train_dataloader)
    
    trainer.train()

def main():
    torch.distributed.init_process_group(backend='nccl')
    train()
    torch.distributed.destroy_process_group()

if __name__ == "__main__":
    main()