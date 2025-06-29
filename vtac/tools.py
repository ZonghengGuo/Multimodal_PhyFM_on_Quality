import logging
import torch
from torch.nn.functional import threshold
from torch.utils.data import Dataset
import torch.nn.functional as F
import random
import numpy as np


def get_logger(
    logpath, filepath, package_files=[], displaying=True, saving=True, debug=False
):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="w")
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)

    logger.info(filepath)

    for f in package_files:
        logger.info(f)
        with open(f, "r") as package_f:
            logger.info(package_f.read())

    return logger


class Dataset_train(Dataset):
    # 'Characterizes a dataset for PyTorch'
    def __init__(self, signal_train, y_train):
        # 'Initialization'
        # self.X = torch.load(path)
        # self.X = pickle.load(open(path, 'rb'))
        self.strain = signal_train  # 信号
        self.ytrain = y_train  # 真假

    def __len__(self):
        # 'Denotes the total number of samples'
        return len(self.ytrain)

    def __getitem__(self, index):
        # 'Generates one sample of data'
        # Select sample
        return self.strain[index], self.ytrain[index]


def train_model(batch, model, loss_ce, device, weight):
    signal_train, y_train = batch
    batch_size = len(signal_train)
    length = 7500
    alarm_time = 75000
    # use the last 30s signal as model input
    signal_train = signal_train[:, :, alarm_time - length : alarm_time].float().to(device)
    y_train = y_train.float().view(-1, 1).to(device)

    # model prediction, feature of alarm signal, feature of randomly selected signal
    _, Y_train_prediction = model(signal_train)

    # calculate loss
    loss = loss_ce(Y_train_prediction, y_train)

    return loss, Y_train_prediction, y_train


def eval_model(
    batch, model, loss_ce, device
):  # signal_train, alarm_train, y_train, signal_test, alarm_test, y_test = batch
    signal_train, y_train = batch
    length = 7500
    alarm_time = 75000

    signal_train = signal_train[:, :, alarm_time - length : alarm_time].float().to(device)

    y_train = y_train.float().view(-1, 1).to(device)

    _,Y_train_prediction = model(signal_train)

    loss = loss_ce(Y_train_prediction, y_train)

    return loss, Y_train_prediction, y_train



def evaluation(
    Y_eval_prediction, y_test, TP, FP, TN, FN
):  # b 2   set 0 is false alarm and 1 is true alarm
    pre = (Y_eval_prediction >= 0).int()
    for i, j in zip(pre, y_test):
        if i.item() == 1 and j.item() == 1:  # 1 -> 1
            TP += 1
        if i.item() == 1 and j.item() == 0:  # 0 -> 1
            FP += 1
        if i.item() == 0 and j.item() == 0:  # 0 -> 0  # false classified to false
            TN += 1
        if (
            i.item() == 0 and j.item() == 1
        ):  # 1 -> 0  # true alarm classified to false alarm
            FN += 1
    return TP, FP, TN, FN


def evaluate_rule_based(rule_based_results, y_test):
    TP = FP = TN = FN = 0
    for i, j in zip(rule_based_results, y_test):
        if i.item() == 1 and j.item() == 1:  # 1 -> 1
            TP += 1
        if i.item() == 1 and j.item() == 0:  # 0 -> 1
            FP += 1
        if i.item() == 0 and j.item() == 0:  # 0 -> 0  # false classified to false
            TN += 1
        if (
            i.item() == 0 and j.item() == 1
        ):  # 1 -> 0  # true alarm classified to false alarm
            FN += 1
    return (
        100 * TP / (TP + FN),
        100 * TN / (TN + FP),
        100 * (TP + TN) / (TP + TN + FP + 5 * FN),
        100 * (TP + TN) / (TP + TN + FP + FN),
    )


def evaluation_test(
    Y_eval_prediction, y_test, types_TP, types_FP, types_TN, types_FN
):  # b 2   set 0 is false alarm and 1 is true alarm
    pre = (Y_eval_prediction >= 0).int()
    for i, j in zip(pre, y_test):
        if i.item() == 1 and j.item() == 1:  # 1 -> 1
            types_TP += 1
        if i.item() == 1 and j.item() == 0:  # 0 -> 1
            types_FP += 1
        if i.item() == 0 and j.item() == 0:  # 0 -> 0  # false classified to false
            types_TN += 1
        if (
            i.item() == 0 and j.item() == 1
        ):  # 1 -> 0  # true alarm classified to false alarm
            types_FN += 1
    return types_TP, types_FP, types_TN, types_FN


def evaluate_raise_threshold(
    prediction, groundtruth, types_TP, types_FP, types_TN, types_FN, threshold
):
    prediction = torch.sigmoid(prediction)

    pre = 1 if prediction >= threshold else 0

    if pre == 1 and groundtruth == 1:
        types_TP += 1
    elif pre == 1 and groundtruth == 0:
        types_FP += 1
    elif pre == 0 and groundtruth == 1:
        types_FN += 1
    elif pre == 0 and groundtruth == 0:
        types_TN += 1

    return types_TP, types_FP, types_TN, types_FN