# -*- coding: utf-8 -*-
"""
@File : adapter.py
@Time : 2024/1/3 下午4:06
@Auth : Yue Zheng
@IDE  : Pycharm2022.3
@Ver. : Python3.9
@Comm : ···
"""
from torch import nn
import torch
from segment_anything.modeling.sam import Sam
from segment_anything.modeling.mask_decoder import MaskDecoder
from segment_anything.modeling.common import LayerNorm2d
import torch.nn.functional as F


def fix_params(model):
    for name, param in model.named_parameters():
        param.requires_grad = False


class MLP(nn.Module):
    def __init__(
            self,
            input_dim,
            hidden_dim,
            output_dim,
            num_layers,
            sigmoid_output=False
    ):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x


class Neck(nn.Module):
    def __init__(
            self,
            trm_dim,
            trm,
            num_multimask_output,
            activation=nn.GELU
    ):
        super().__init__()
        self.trm_dim = trm_dim
        self.trm = trm
        self.num_multimask = num_multimask_output
        self.iou_token = nn.Embedding(1, trm_dim)
        self.num_mask_tokens = num_multimask_output + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, trm_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(trm_dim, trm_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(trm_dim // 4),
            activation(),
            nn.ConvTranspose2d(trm_dim // 4, trm_dim // 8, kernel_size=2, stride=2),
            activation(),
        )

    def forward(self,
                image_embed,
                image_pe,
                sparse_prompt_embed,
                dense_prompt_embed,
                multimask_output):
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(
            sparse_prompt_embed.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embed), dim=1)
        src = torch.repeat_interleave(image_embed, tokens.shape[0], dim=0)
        src = src + dense_prompt_embed
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        src_shape = src.shape
        hs, src = self.trm(src, pos_src, tokens)
        iou_token_output = hs[:, 0, :]
        mask_tokens_output = hs[:, 1:(1 + self.num_mask_tokens), :]
        return src, iou_token_output, mask_tokens_output, src_shape


class Head(nn.Module):
    def __init__(
            self,
            trm_dim,
            num_multimask_output=3,
            activation=nn.GELU,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            class_num=20
    ):
        super().__init__()
        self.trm_dim = trm_dim
        self.num_multimask_output = num_multimask_output
        self.num_mask_tokens = num_multimask_output + 1
        self.class_num = class_num

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(trm_dim, trm_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(trm_dim // 4),
            activation(),
            nn.ConvTranspose2d(trm_dim // 4, trm_dim // 8, kernel_size=2, stride=2),
            activation()
        )

        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(trm_dim, trm_dim, trm_dim // 8, 3)
                for _ in range(self.class_num)
            ]
        )

        self.iou_pred_head = MLP(
            trm_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

    def forward(
            self,
            src, iou_token_out,
            mask_tokens_out,
            src_shape,
            mask_scale=1,
    ):
        b, c, h, w = src_shape

        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embed = self.output_upscaling(src)
        hyper_in_list = []
        for i in range(self.class_num):
            hyper_in_list.append(
                self.output_hypernetworks_mlps[i](mask_tokens_out[:, mask_scale, :])
            )
        hyper_in = torch.stack(hyper_in_list, dim=1)

        b, c, h, w = upscaled_embed.shape
        masks = (hyper_in @ upscaled_embed.view(b, c, h * w)).view(b, -1, h, w)  # B N H W, N is num of category

        # Generate mask quality predictions
        iou_pred = self.iou_pred_head(iou_token_out)  # B N H W, N is num of category

        return masks, iou_pred


class ImageEncoderAdapter(nn.Module):
    def __init__(self, sam: Sam, fix=False):
        super().__init__()
        self.image_encoder = sam.image_encoder
        if fix:
            for name, param in self.image_encoder.named_parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.image_encoder(x)
        return x


class PromptEncoderAdapter(nn.Module):
    def __init__(self, sam: Sam, fix=False):
        super().__init__()
        self.prompt_encoder = sam.prompt_encoder
        if fix:
            for name, param in self.prompt_encoder.named_parameters():
                param.requires_grad = False

    def forward(self, points=None, boxes=None, masks=None):
        sparse_embed, dense_embed = self.prompt_encoder(points, boxes, masks)
        return sparse_embed, dense_embed


class MaskDecoderAdapter(MaskDecoder):
    def __init__(self, sam: Sam, fix=False, class_num=20):
        super().__init__(
            transformer=sam.mask_decoder.transformer,
            transformer_dim=sam.mask_decoder.transformer_dim,
        )
        self.mask_decoder = sam.mask_decoder
        if fix:
            for name, param in self.mask_decoder.named_parameters():
                param.requires_grad = False

        self.neck = Neck(
            trm=self.mask_decoder.transformer,
            trm_dim=self.mask_decoder.transformer_dim,
            num_multimask_output=self.mask_decoder.num_multimask_outputs
        )

        self.head = Head(
            trm_dim=self.mask_decoder.transformer_dim,
            num_multimask_output=self.mask_decoder.num_multimask_outputs,
            iou_head_depth=self.mask_decoder.iou_head_depth,
            iou_head_hidden_dim=self.mask_decoder.iou_head_hidden_dim,
            class_num=class_num
        )

        # maskdecoder和maskdecider_adapter之间的参数配对
        self.pair_params(self.neck)
        self.pair_params(self.head)


    def forward(
            self,
            image_embed,
            prompt_adapter: PromptEncoderAdapter,
            sparse_embed,
            dense_embed,
            multimask_output=True,
            scale=1
    ):

        src, iou_token_out, mask_tokens_out, src_shape = self.neck(
            image_embed=image_embed,
            image_pe=prompt_adapter.prompt_encoder.get_dense_pe(),
            sparse_prompt_embed=sparse_embed,
            dense_prompt_embed=dense_embed,
            multimask_output=multimask_output,
        )
        masks, iou_pred = self.head(src, iou_token_out, mask_tokens_out, src_shape, mask_scale=scale)
        return masks, iou_pred

    def pair_params(self, targe_model):
        src_dict = self.mask_decoder.state_dict()
        for name, value in targe_model.named_parameters():
            if name in src_dict.keys():
                value.data.copy_(src_dict[name].data)
