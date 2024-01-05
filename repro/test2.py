# -*- coding: utf-8 -*-
"""
@File : test2.py
@Time : 2024/1/3 下午4:55
@Auth : Yue Zheng
@IDE  : Pycharm2022.3
@Ver. : Python3.9
@Comm : ···
"""
import yaml
from omegaconf import OmegaConf
#
with open("config.yaml") as file:
    config = yaml.safe_load(file)



a=config["train"]["opt_params"]["lr_list"][0]
b=config["train"]["model"]["params"]["fix_image_encoder"]
c=config["train"]["model"]["params"]["class_num"]
d=config["train"]["dataset"]["transforms"]["resize"]["params"]["size"][0]
e=config["train"]["opt_params"]["lr_default"]
f=config["train"]["opt_params"]["wd_list"][0]

print(type(e))
print(type(f))


# cfg = OmegaConf.load("./config.yaml")
# b = cfg.train.opt_params.lr_list
# print(b)
