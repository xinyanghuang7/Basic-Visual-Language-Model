import torch
import numpy as np
from torch import nn
from typing import Optional
from transformers import ChineseCLIPModel
from transformers.models.chinese_clip.configuration_chinese_clip import ChineseCLIPConfig
from transformers.utils import add_start_docstrings_to_model_forward
from transformers.models.chinese_clip.modeling_chinese_clip import CHINESE_CLIP_VISION_INPUTS_DOCSTRING

class visualModel(ChineseCLIPModel):
    def __init__(self, config: ChineseCLIPConfig):
        super().__init__(config)
        

    @add_start_docstrings_to_model_forward(CHINESE_CLIP_VISION_INPUTS_DOCSTRING)
    def get_image_features(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:
        # Use CHINESE_CLIP model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        hidden_feature = vision_outputs.last_hidden_state

        return hidden_feature

def main():
    vm = visualModel.from_pretrained("F:/huggingface_model/models--OFA-Sys--chinese-clip-vit-large-patch14-336px/snapshots/567e85f859213abde0cd45e95d87f4421ccbc14c/", 100)
    data = torch.randn(1,3,336,336)
    print(vm.get_image_features(data).shape)

if __name__ == "__main__":
    main()
