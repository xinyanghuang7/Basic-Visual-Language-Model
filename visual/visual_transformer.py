from functools import partial
from typing import Callable, List, Optional, Any
import torch
from torch import nn
from torch.nn.modules import LayerNorm, Module

from torchvision.models import VisionTransformer
from torchvision.models._utils import _ovewrite_named_param, handle_legacy_interface
from torchvision.models._api import WeightsEnum, register_model
from torchvision.models.vision_transformer import ConvStemConfig, ViT_B_16_Weights
from torchvision import models


class MVIT(VisionTransformer):
    def __init__(self,
        image_size: int,
        patch_size: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        num_classes: int = 1000,
        representation_size: Optional[int] = None,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        conv_stem_configs: Optional[List[ConvStemConfig]] = None,):
        super().__init__(image_size, patch_size, num_layers, num_heads, hidden_dim, mlp_dim, dropout, attention_dropout, num_classes, representation_size, norm_layer, conv_stem_configs)


    def forward(self, x: torch.Tensor):
        # Reshape and permute the input tensor
        x = self._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.encoder(x)

        # x = x[:, 0].reshape(n, -1)

        return x
    
def _vision_transformer(
    patch_size: int,
    num_layers: int,
    num_heads: int,
    hidden_dim: int,
    mlp_dim: int,
    weights: Optional[WeightsEnum],
    progress: bool,
    model_dir: str, 
    **kwargs: Any,
) -> VisionTransformer:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))
        assert weights.meta["min_size"][0] == weights.meta["min_size"][1]
        _ovewrite_named_param(kwargs, "image_size", weights.meta["min_size"][0])
    image_size = kwargs.pop("image_size", 224)

    model = MVIT(
        image_size=image_size,
        patch_size=patch_size,
        num_layers=num_layers,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        mlp_dim=mlp_dim,
        **kwargs,
    )

    if weights:
        model.load_state_dict(weights.get_state_dict(progress=progress, model_dir = model_dir))

    return model

@register_model()
@handle_legacy_interface(weights=("pretrained", ViT_B_16_Weights.IMAGENET1K_V1))
def Mvit_b_16(*, weights: Optional[ViT_B_16_Weights] = None, progress: bool = True, model_dir: Optional[str] = None, **kwargs: Any) -> VisionTransformer:
    weights = ViT_B_16_Weights.verify(weights)

    return _vision_transformer(
        patch_size=16,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        mlp_dim=3072,
        weights=weights,
        progress=progress,
        model_dir=model_dir,
        **kwargs,
    )

def main():
    MV = Mvit_b_16(pretrained=True).cuda()
    data = torch.rand(1, 3, 224, 224).cuda()
    print(MV(data).shape)

if __name__ == "__main__":
    main()
