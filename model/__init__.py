import timm
import torch
from timm.models.vision_transformer import PatchEmbed

from .STP import SpatioTemporalPrompt


def create_model(logger_name):

    if 'ViT_S' in logger_name:
        model_name = 'vit_small_patch16_224'
        dim = 384
    elif 'ViT_B' in logger_name:
        model_name = 'vit_base_patch16_224'
        dim = 768
    elif 'ViT_L' in logger_name:
        model_name = 'vit_large_patch16_224'
        dim = 1024


    vit = timm.create_model(model_name=model_name, pretrained=True)
    vit.patch_embed = PatchEmbed(224, 16, 5, dim)
    w = vit.patch_embed.proj.weight.data
    torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
    
    STP_pre = SpatioTemporalPrompt(vit)
    
    return STP_pre

