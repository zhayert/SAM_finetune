# -*- coding: utf-8 -*-
"""
@File : utils.py
@Time : 2024/1/3 上午9:01
@Auth : Yue Zheng
@IDE  : Pycharm2022.3
@Ver. : Python3.9
@Comm : ···
"""
import torch.optim
from torchvision import transforms

from torch import nn
import torch.nn.functional as F

from repro.ext_sam import SemSam
import numpy as np
from torchvision.datasets import VOCSegmentation
from PIL import Image

from repro.runner import SemRunner


def get_dataset(dataset_cfg):
    name = dataset_cfg["name"]
    assert name == "torch_voc_sem", print("{} is illegal.".format(name))
    trm = get_transforms(dataset_cfg["transforms"])
    target_trm = get_transforms((dataset_cfg["target_transforms"]))
    result = TorchVOCSeg(**(dataset_cfg["params"]), transform=trm, target_transform=target_trm)
    return result


def get_transforms(trm_cfg):
    trm_list = []
    trm_dict = {"resize": transforms.Resize,
                "to_tensor": transforms.ToTensor}
    for name in trm_cfg.keys():
        if trm_cfg[name]["params"] is not None:
            trm_list.append(trm_dict[name](**trm_cfg[name]["params"]))
        elif trm_cfg[name]["params"] is None:
            trm_list.append(trm_dict[name]())
    return transforms.Compose(trm_list)


def get_losses(loss_cfg):
    loss_dict = {}
    name = next(iter(loss_cfg))
    if loss_cfg[name]["params"] is not None:
        loss_dict[name] = nn.CrossEntropyLoss(**loss_cfg[name]["params"])
    else:
        loss_dict[name] = nn.CrossEntropyLoss()
    return loss_dict


def get_model(model_name="sem_sam", **kwargs):
    assert model_name == "sem_sam"
    return SemSam(**kwargs).cuda()


def get_opt_params(model, lr_list, group_keys, wd_list):
    assert len(lr_list) == len(group_keys)
    assert len(lr_list) == len(wd_list)
    params_group = [[] for _ in range(len(lr_list))]
    for name, value in model.named_parameters():
        for index, g_keys in enumerate(group_keys):
            for g_key in g_keys:
                if g_key in name:
                    params_group[index].append(value)
    return [{"params": params_group[i],
             "lr": lr_list[i],
             "weight_decay": wd_list[i]} for i in range(len(lr_list))]


def get_optimizer(opt_name, **kwargs):
    assert opt_name == "sgd"
    return torch.optim.SGD(
        **{k: v for k, v in kwargs.items() if v is not None}
    )


def get_schduler(
        optimizer,  # 优化器对象，用于更新模型参数的优化器，例如Adam、SGD等。
        lr_scheduler="single_step",  # 学习率调度器方法，表示选择哪种学习率调度器。
        stepsize=1,  # 学习率调整的步长（或里程碑），用于指定学习率调整的时机。
        gamma=0.1,  # 学习率衰减率，用于指定学习率的衰减速度。
        warmup_factor=0.01,  # 热身阶段的学习率因子，用于在训练初期进行学习率的预热。
        warmup_steos=10,  # 热身阶段的步数，表示进行学习率预热的迭代次数。
        max_epoch=1,  # 最大的训练轮数，用于指定在"cosine"学习率调度器中的最大训练轮数。
        n_epochs_init=50,  # 线性学习率调度器中的初始训练轮数。
        n_epochs_decay=50,  # 线性学习率调度器中的衰减训练轮数。
):
    assert lr_scheduler == "cosine"
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, int(max_epoch)
    )
    return scheduler


def get_runner(runner_name):
    assert runner_name == "sem_runner"
    return SemRunner


class TorchVOCSeg(VOCSegmentation):
    def __init__(
            self,
            root,
            year="2012",
            image_set="train",
            download=False,
            transform=None,
            target_transform=None
    ):
        super().__init__(
            root=root,
            year=year,
            image_set=image_set,
            download=download,
            transform=transform,
            target_transform=target_transform
        )

        self.class_names = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                           'bus', 'car', 'cat', 'chair', 'cow',
                           'diningtable', 'dog', 'horse', 'motorbike', 'person',
                           'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

    def __getitem__(self, i):
        image = Image.open(self.images[i]).convert("RGB")
        target = Image.open(self.masks[i])

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        target = np.array(target)
        return image, target
