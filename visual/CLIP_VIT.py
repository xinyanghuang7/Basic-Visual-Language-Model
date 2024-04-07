import transformers
import torch
from typing import Optional
from torch import nn
from PIL import Image
from transformers import CLIPModel, CLIPConfig, CLIPProcessor
from transformers.utils import add_start_docstrings_to_model_forward
from transformers.models.clip.modeling_clip import CLIP_VISION_INPUTS_DOCSTRING

class visualModel(CLIPModel):
    def __init__(self, config: CLIPConfig):
        super().__init__(config)

    @add_start_docstrings_to_model_forward(CLIP_VISION_INPUTS_DOCSTRING)
    def get_image_features(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:
        
        # Use CLIP model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = vision_outputs.last_hidden_state  # pooled_output
        # print(pooled_output.shape)
        return pooled_output


def main():
    modle_path = "F:/huggingface_model/clip-vit-large-patch14"
    model = visualModel.from_pretrained(modle_path)
    processor = CLIPProcessor.from_pretrained(modle_path)
    test_img = Image.open("D:/code/multimodal/data/000000391895.jpg")
    P_input = processor(images=test_img, return_tensors="pt")
    print(model.get_image_features(**P_input).shape)

if __name__ == "__main__":
    main()
