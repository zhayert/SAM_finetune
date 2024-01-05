# -*- coding: utf-8 -*-
"""
@File : runner.py
@Time : 2024/1/3 上午9:04
@Auth : Yue Zheng
@IDE  : Pycharm2022.3
@Ver. : Python3.9
@Comm : ···
"""
import os

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import time


def log_print(iter, log_msg, timer):
    log_msg["iter"] = iter
    log_msg["time"] = timer.end(clear=True)
    msg = "iteration : {val}, ".format(val=log_msg["iter"])
    for k, v in log_msg.items():
        if k == "iter":
            continue
        msg += "{} : {}, ".format(k, v)
    msg = msg[:-2]
    print(msg)


class SemRunner:
    def __init__(
            self,
            model,
            optimizer,
            losses,
            train_loader,
            val_loader,
            schedular,
    ):
        self.optimizer = optimizer
        self.losses = losses
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model
        self.schedular = schedular
        self.train_timer = Timer()
        self.eval_timer = Timer()

        try:
            use_gpu = os.environ["CUDA_VISIBLE_DEVICES"]
        except KeyError:
            use_gpu = "0"
        self.the_number_of_gpu = len(use_gpu.split(","))
        self.original_size = self.model.image_adapter.image_encoder.img_size
        if self.the_number_of_gpu > 1:
            self.model = nn.DataParallel(self.model)
        self.exist_status = ["train", "eval", "test"]

    def train(self, cfg):
        # 管理和计算一组数据的平均值，跟踪损失值、准确率等指标的平均变化。
        train_meter = AveMeter(list(self.losses.keys()) + ["total_loss"])
        # 简化数据加载器
        train_iter = Iter(self.train_loader)
        best_miou = -1

        # train
        for iter in range(cfg["max_iter"]):
            images, labels = train_iter.get()
            images, labels = images.cuda(), labels.cuda().long()
            masks_pred, iou_pred = self.model(images)
            masks_pred = F.interpolate(
                masks_pred,
                self.original_size,
                mode="bilinear",
                align_corners=False)
            total_loss = torch.zeros(1).cuda()
            loss_dict = {}
            self._compute_loss(total_loss, loss_dict, masks_pred, labels, cfg)
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            self.schedular.step()
            loss_dict["total_loss"] = total_loss.item()
            train_meter.add(loss_dict)

            # log
            if (iter + 1) % cfg["log_iter"] == 0:
                log_print(iter, train_meter.get(clear=True), timer=self.train_timer)
            # eval
            if (iter + 1) % cfg["eval_iter"] == 0:
                miou, _ = self._eval()
                if best_miou == -1 or best_miou < miou:
                    best_miou = miou
                log_data = {"mIoU": miou, "best_valid_mIoU": best_miou}
                log_print(iter,log_data,timer=self.eval_timer)

    def test(self):
        pass

    def _eval(self):
        self.model.eval()
        self.eval_timer.start()
        class_names = self.val_loader.dataset.class_names
        eval_metric = mIoUOnline(class_names=class_names)
        with torch.no_grad():
            for index, (images, labels) in enumerate(self.val_loader):
                images = images.cuda()
                labels = labels.cuda()
                masks_pred, iou_pred = self.model(images)
                preds = torch.argmax(masks_pred, dim=1)
                for batch_index in range(images.size()[0]):
                    pred_mask = preds[batch_index].cpu().detach().numpy()
                    gt_mask = labels[batch_index].squeeze(0).cpu().detach().numpy()
                    h, w = pred_mask.shape
                    gt_mask = cv2.resize(gt_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                    eval_metric.add(pred_mask, gt_mask)
        self.model.train()
        return eval_metric.get(clear=True)

    def _compute_loss(self, total_loss, loss_dict, mask_pred, labels, cfg):
        loss_cfg = cfg["losses"]
        for index, item in enumerate(self.losses.items()):
            # item -> (key: loss_name, val: loss)
            real_labels = labels  # real_labels: B H W
            if loss_cfg[item[0]]["label_one_hot"]:
                class_num = cfg["model"]["params"]["class_num"]  # class_num: N
                one_hot_labels = real_labels.clone
                one_hot_labels[one_hot_labels == 255] = 0  # 0 is background
                real_labels = F.one_hot(
                    tensor=one_hot_labels,
                    num_classes=class_num
                ).permute(0, 3, 1, 2).contiguous().float()  # B N H W
            tmp_loss = item[1](mask_pred, real_labels)
            loss_dict[item[0]] = tmp_loss.item()
            total_loss += loss_cfg[item[0]]["weight"] * tmp_loss


class AveMeter:
    def __init__(self, keys):
        self.keys = keys
        self.clear()

    def add(self, dic):
        for key, value in dic.items():
            self.data_dic[key].append(value)

    def get(self, keys=None, clear=False):
        if keys is None:
            keys = self.keys

        dataset = {}
        for key in keys:
            dataset[key] = float(np.mean(self.data_dic[key]))

        if clear:
            self.clear()

        return dataset

    def clear(self):
        self.data_dic = {key: [] for key in self.keys}


class Iter:
    def __init__(self, loader):
        self.loader = loader
        self.iter = iter(self.loader)

    def get(self):
        try:
            data = next(self.iter)
        except StopIteration:
            self.iter = iter(self.loader)
            data = next(self.iter)
        return data


class Timer:
    def __init__(self):
        self.start_time = 0.0
        self.end_time = 0.0

        self.start()

    def start(self):
        self.start_time = time.time()

    def end(self, ms=False, clear=False):
        self.end_time = time.time()

        if ms:
            duration = int((self.end_time - self.start_time) * 1000)
        else:
            duration = int(self.end_time - self.start_time)

        if clear:
            self.start()
        return duration


class mIoUOnline:
    def __init__(self, class_names):
        self.class_names = ['background'] + class_names
        self.class_num = len(self.class_names)

        self.clear()

    def get_data(self, pred_mask, gt_mask):
        obj_mask = gt_mask < 255
        correct_mask = (pred_mask == gt_mask) * obj_mask

        P_list, T_list, TP_list = [], [], []
        for i in range(self.class_num):
            P_list.append(np.sum((pred_mask == i) * obj_mask))
            T_list.append(np.sum((gt_mask == i) * obj_mask))
            TP_list.append(np.sum((gt_mask == i) * correct_mask))

        return (P_list, T_list, TP_list)

    def add_using_data(self, data):
        P_list, T_list, TP_list = data
        for i in range(self.class_num):
            self.P[i] += P_list[i]
            self.T[i] += T_list[i]
            self.TP[i] += TP_list[i]

    def add(self, pred_mask, gt_mask):
        obj_mask = gt_mask < 255
        correct_mask = (pred_mask == gt_mask) * obj_mask

        for i in range(self.class_num):
            self.P[i] += np.sum((pred_mask == i) * obj_mask)
            self.T[i] += np.sum((gt_mask == i) * obj_mask)
            self.TP[i] += np.sum((gt_mask == i) * correct_mask)

    def get(self, detail=False, clear=True):
        IoU_dic = {}
        IoU_list = []

        FP_list = []  # over activation
        FN_list = []  # under activation

        for i in range(self.class_num):
            IoU = self.TP[i] / (self.T[i] + self.P[i] - self.TP[i] + 1e-10) * 100
            FP = (self.P[i] - self.TP[i]) / (self.T[i] + self.P[i] - self.TP[i] + 1e-10)
            FN = (self.T[i] - self.TP[i]) / (self.T[i] + self.P[i] - self.TP[i] + 1e-10)

            IoU_dic[self.class_names[i]] = IoU

            IoU_list.append(IoU)
            FP_list.append(FP)
            FN_list.append(FN)

        mIoU = np.mean(np.asarray(IoU_list))
        mIoU_foreground = np.mean(np.asarray(IoU_list)[1:])

        FP = np.mean(np.asarray(FP_list))
        FN = np.mean(np.asarray(FN_list))

        if clear:
            self.clear()

        if detail:
            return mIoU, mIoU_foreground, IoU_dic, FP, FN
        else:
            return mIoU, mIoU_foreground

    def clear(self):
        self.TP = []
        self.P = []
        self.T = []

        for _ in range(self.class_num):
            self.TP.append(0)
            self.P.append(0)
            self.T.append(0)
