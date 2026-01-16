from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPConfig
from transformers.models.clip.modeling_clip import CLIPOutput


class NativeOpenCLIPModel(CLIPModel):
    """
    原生 OpenCLIP 模型转换为 Hugging Face 格式
    用于使用原生 CLIP 文本编码器的模型（非 HFTextEncoder）
    """
    
    def __init__(self, config: CLIPConfig):
        super().__init__(config)
        
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        return_loss: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CLIPOutput]:
        
        # 使用父类的 forward 方法
        return super().forward(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_loss=return_loss,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )