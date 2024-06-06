import transformers
import torch
from typing import Optional
from torch import nn
from PIL import Image
from transformers import SiglipModel, SiglipConfig, SiglipProcessor
from transformers.utils import add_start_docstrings_to_model_forward
from transformers.models.siglip.modeling_siglip import SIGLIP_VISION_INPUTS_DOCSTRING

class visualModel(SiglipModel):
    def __init__(self, config: SiglipConfig):
        super().__init__(config)
        vision_config = config.vision_config
        self.vision_embed_dim = vision_config.hidden_size

    @add_start_docstrings_to_model_forward(SIGLIP_VISION_INPUTS_DOCSTRING)
    def get_image_features(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 确保 pixel_values 的数据类型为 bfloat16
        pixel_values = pixel_values.to(dtype=torch.bfloat16)

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = vision_outputs.last_hidden_state  # pooled_output
        return pooled_output