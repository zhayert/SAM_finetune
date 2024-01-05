# -*- coding: utf-8 -*-
"""
@File : main.py
@Time : 2024/1/3 上午8:44
@Auth : Yue Zheng
@IDE  : Pycharm2022.3
@Ver. : Python3.9
@Comm : ···
"""

# import argparse
# from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from utils import get_dataset, get_losses, get_model, get_opt_params, get_optimizer, get_schduler, get_runner
import yaml

if __name__ == '__main__':
    # config = OmegaConf.load("config.yaml")
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    train_config = config["train"]
    val_config = config["val"]
    test_config = config["test"]

    train_dataset = get_dataset(train_config["dataset"])
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=train_config["bs"],
        shuffle=True,
        num_workers=train_config["num_workers"],
        drop_last=train_config["drop_last"])

    val_dataset = get_dataset(val_config["dataset"])
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=val_config["bs"],
        shuffle=False,
        num_workers=val_config["num_workers"],
        drop_last=val_config["drop_last"]
    )
    losses = get_losses(loss_cfg=train_config["losses"])
    model = get_model(
        model_name=train_config["model"]["sam_name"],
        **train_config["model"]["params"])
    opt_params = get_opt_params(
        model=model,
        lr_list=train_config["opt_params"]["lr_list"],
        group_keys=train_config["opt_params"]["group_keys"],
        wd_list=train_config["opt_params"]["wd_list"],
    )
    optimizer = get_optimizer(
        opt_name=train_config["opt_name"],
        params=opt_params,
        lr=train_config["opt_params"]["lr_default"],
        momentum=train_config["opt_params"]["momentum"],
        weight_decay=train_config["opt_params"]["wd_default"],
    )
    scheduler = get_schduler(
        optimizer=optimizer,
        lr_scheduler=train_config["scheduler_name"]
    )
    runner = get_runner(train_config["runner_name"])(
        model,
        optimizer,
        losses,
        train_loader,
        val_loader,
        scheduler
    )
    # train_step
    runner.train(train_config)
