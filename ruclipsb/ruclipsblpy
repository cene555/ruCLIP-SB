from collections import OrderedDict
from typing import Tuple, Union
from timm import create_model as create_swin
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import BertModel

class ruCLIPSB(nn.Module):
    def __init__(self,):
        super().__init__()
        self.visual = swin = create_swin(
            'swin_tiny_patch4_window7_224', pretrained=True, num_classes=0, in_chans=3) #out 768
        self.transformer = BertModel.from_pretrained("cointegrated/rubert-tiny")
        self.final_ln = nn.Linear(312, 768)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    @property
    def dtype(self):
        return self.visual.patch_embed.proj.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(self, input_ids, attention_mask):
        x = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        x = x.last_hidden_state[:, 0, :]
        x = self.final_ln(x)
        return x

    def forward(self, image, input_ids, attention_mask):
        image_features = self.encode_image(image)
        text_features = self.encode_text(input_ids, attention_mask)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        return logits_per_image, logits_per_text
