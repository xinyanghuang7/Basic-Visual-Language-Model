import torch 
from torch import nn
from typing import Optional
import os
import sys
sys.path.append("../")
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import CLIPProcessor    
from dataclasses import dataclass, asdict
from peft import get_peft_model, LoraConfig, TaskType, PeftModel

from visual.CLIP_VIT import visualModel
# from visual.visual_transformer import Mvit_b_16
from qwen.Mqwen import MQWenLMHeadModel
# from qwen.modeling_qwen import QWenLMHeadModel


@dataclass
class LanguageConfig():
    model_path: str
    torch_dtype: torch.dtype = torch.bfloat16
    trust_remote_code: bool = True

@dataclass
class VisualConfig():
    model_path: str
    pretrained: bool = True
    

@dataclass
class MultiModalConfig():
    replace_token_id: int
    image_context_length: int = 256
    image_feature_hidden_size: int = 4096
    

def make_lora(model, finetune_args):
    peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=finetune_args.lora_rank,
            lora_alpha=32,
            lora_dropout=finetune_args.lora_dropout,
            target_modules = finetune_args.target_modules.split('|') # 把model打印出来，找跟attention相关的模块
        )

    model = get_peft_model(model, peft_config)

    return model

class MMultiModal(nn.Module):
    def __init__(self, Lconfig: LanguageConfig, Vconfig: VisualConfig, MMconfig: MultiModalConfig, finetune_args = None, train = False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        image_feature_length = MMconfig.image_context_length * MMconfig.image_feature_hidden_size

        self.LLM = MQWenLMHeadModel.from_pretrained(Lconfig.model_path, asdict(MMconfig), torch_dtype = Lconfig.torch_dtype, trust_remote_code = Lconfig.trust_remote_code)

        if train:
            self.LLM.gradient_checkpointing_enable() 
            self.LLM.enable_input_require_grads()

        self.LLM.config.image_feature_length = image_feature_length

        if train and finetune_args is not None:
            self.LLM = make_lora(self.LLM, finetune_args)

        assert MMconfig.image_feature_hidden_size == self.LLM.config.hidden_size

        self.visualModel = visualModel.from_pretrained(Vconfig.model_path).to(Lconfig.torch_dtype)

        Vhidden_dim = self.visualModel.vision_embed_dim
        Lhidden_dim = self.LLM.config.hidden_size

        self.make_feature_proj(Vhidden_dim, Lhidden_dim, Lconfig)

        self.MMconfig = MMconfig

    def make_feature_proj(self, Vhidden_dim, Lhidden_dim, Lconfig):
        self.feature_proj = nn.Sequential(
            nn.Linear(Vhidden_dim, Lhidden_dim, dtype=Lconfig.torch_dtype),
            nn.GELU(),
            nn.Linear(Lhidden_dim, Lhidden_dim, dtype=Lconfig.torch_dtype)
        )
    
        for name, module in self.feature_proj.named_children():
            if "Linear" in module._get_name(): 
                module.weight.data.normal_(mean=0.0, std = 0.01)
                module.bias.data.zero_()

    def forward(self, image: torch.Tensor, input_ids: torch.LongTensor, labels: Optional[torch.LongTensor] = None):
        with torch.no_grad():
            image_feature=self.visualModel.get_image_features(pixel_values=image)[:,1:, :]
            image_feature = image_feature.detach()

        image_feature = self.feature_proj(image_feature)

        out = self.LLM(input_ids, labels=labels, images=image_feature)

        loss1 = out.loss

        return CausalLMOutputWithPast(
            loss=loss1,
            logits=out.logits,
            past_key_values=out.past_key_values,
            hidden_states=out.hidden_states,
            attentions=out.attentions,
        )
    
    def to(self, *args, **kwargs):
        return super().to(*args, **kwargs)
    
    def load(self, modelPath):
        self.LLM = PeftModel.from_pretrained(self.LLM, modelPath, inference_mode=True)
        other_params = torch.load(os.path.join(modelPath, "other_params.bin"))
        self.feature_proj.load_state_dict(other_params)

    @torch.no_grad()
    def generate(self, image: torch.Tensor, input_ids: torch.LongTensor):
        if image is None:
            image_feature = None
        else:
            image_feature=self.visualModel.get_image_features(pixel_values=image)[:,1:, :]
            image_feature = self.feature_proj(image_feature)

        input_ids = torch.tensor([input_ids]).long().to(self.LLM.device)

        out = self.LLM.generate(inputs = input_ids, images=image_feature)[:, len(input_ids[0]):-1]

        return out.long().cpu()