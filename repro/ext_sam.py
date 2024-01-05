# -*- coding: utf-8 -*-
"""
@File : ext_sam.py
@Time : 2024/1/3 下午3:58
@Auth : Yue Zheng
@IDE  : Pycharm2022.3
@Ver. : Python3.9
@Comm : ···
"""

from torch import nn
from segment_anything.build_sam import sam_model_registry
from adapter import ImageEncoderAdapter, PromptEncoderAdapter, MaskDecoderAdapter


class SemSam(nn.Module):
    def __init__(
            self,
            ckpt_path=None,
            model_type="vit_b",
            fix_image_encoder=True,
            fix_prompt_encoder=True,
            fix_mask_decoder=False,
            class_num=20
    ):
        super().__init__()
        assert model_type in ['vit_b', 'vit_l', 'vit_h'], print(
            "model_type error, your model_type is {}".format(model_type)
        )
        self.sam = sam_model_registry[model_type](ckpt_path)
        self.image_adapter = ImageEncoderAdapter(self.sam, fix_image_encoder)
        self.prompt_adapter = PromptEncoderAdapter(self.sam, fix_prompt_encoder)
        self.mask_adapter = MaskDecoderAdapter(self.sam, fix_mask_decoder, class_num)

    def forward(self, x):
        x = self.image_adapter(x)
        points, boxes, masks = None, None, None
        sparse_embed, dense_embed = self.prompt_adapter(
            points=points,
            boxes=boxes,
            masks=masks,
        )
        masks, iou_pred = self.mask_adapter(
            image_embed=x,
            prompt_adapter=self.prompt_adapter,
            sparse_embed=sparse_embed,
            dense_embed=dense_embed,
            multimask_output=True,
            scale=1
        )
        return masks, iou_pred
