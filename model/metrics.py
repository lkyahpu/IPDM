import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from numba import jit
import cv2
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import threading



def batch_pix_accuracy(output, target):

    assert output.shape == target.shape, "Predict and Label Shape Don't Match"
    predict = (output > 0)
    pixel_labeled = (target > 0).sum()
    pixel_correct = (((predict == target))*((target > 0))).sum()



    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
    return pixel_correct, pixel_labeled


def batch_intersection_union(output, target):

    mini = 1
    maxi = 1
    nbins = 1
    predict = (output > 0)

    intersection = predict * ((predict == target))

    area_inter, _  = np.histogram(intersection, bins=nbins, range=(mini, maxi))
    area_pred,  _  = np.histogram(predict, bins=nbins, range=(mini, maxi))
    area_lab,   _  = np.histogram(target, bins=nbins, range=(mini, maxi))
    area_union     = area_pred + area_lab - area_inter

    # assert (area_inter <= area_union).all(), \
    #     "Error: Intersection area should be smaller than Union area"
    return area_inter, area_union


def IoU(preds, labels):

    correct, labeled = batch_pix_accuracy(preds, labels)
    inter, union = batch_intersection_union(preds, labels)

    pixAcc = 1.0 * correct / (np.spacing(1) + labeled)
    IoU = 1.0 * inter / (np.spacing(1) + union)
    # mIoU = IoU.mean()

    return IoU



def calculateF1Measure(output_image,gt_image,thre):
    output_image = np.squeeze(output_image)
    gt_image = np.squeeze(gt_image)
    out_bin = output_image>thre
    gt_bin = gt_image>thre
    recall = np.sum(gt_bin*out_bin)/np.maximum(1,np.sum(gt_bin))
    prec   = np.sum(gt_bin*out_bin)/np.maximum(1,np.sum(out_bin))
    F1 = 2*recall*prec/np.maximum(0.001,recall+prec)
    return prec, recall, F1



def cal_tp_pos_fp_neg(out, gt, score_thresh):
    predict = ((out / 255.0) > score_thresh).astype('uint8')  # P
    target = (gt / 255).astype('uint8')  # T

    intersection = predict * (predict == target) # TP
    tp = intersection.sum()
    fp = (predict * (predict != target)).sum()  # FP
    tn = ((1 - predict) * (predict == target)).sum()  # TN
    fn = ((predict != target) * (1 - predict)).sum()   # FN
    pos = tp + fn
    neg = fp + tn
    return tp, fp, tn, fn, pos, neg


def metrics(out,gt,thres):

    # image = (image - np.min(image)) / (np.max(image) - np.min(image))

    tp, fp, tn, fn, pos, neg = cal_tp_pos_fp_neg(out,  gt,  thres)

    predict = ((out/255.0) > thres).astype('uint8')  # P
    target = (gt/255).astype('uint8')   # T

    recall = np.sum(target * predict) / np.maximum(1, np.sum(target))
    prec = np.sum(target * predict) / np.maximum(1, np.sum(predict))

    F1 = 2 * recall * prec / np.maximum(0.001, recall + prec)

    iou_measure = IoU(predict, target)

    acc_measure = (tp + tn) / (pos + neg + 0.001)

    fp_rates = fp / (fp + tn + 0.001)

    fn_rates = fn / (fn + tp + 0.001)

    return acc_measure, F1, iou_measure, fp_rates, fn_rates, prec, recall


class ROCMetric():
    def __init__(self, nclass, bins):
        self.nclass = nclass
        self.bins = bins
        self.reset()

    def update(self, preds, labels):
        for iBin in range(self.bins+1):
            score_thresh = (iBin + 0.0) / self.bins
            i_tp, i_fp, i_tn, i_fn,  i_pos, i_neg = cal_tp_pos_fp_neg(preds, labels,  score_thresh)

            self.tp_arr[iBin] += i_tp
            self.pos_arr[iBin] += i_pos
            self.fp_arr[iBin] += i_fp
            self.neg_arr[iBin] += i_neg

    def get(self):
        tp_rates = self.tp_arr / (self.pos_arr + 0.001)
        fp_rates = self.fp_arr / (self.neg_arr + 0.001)

        return tp_rates, fp_rates

    def reset(self):
        self.tp_arr = np.zeros(self.bins + 1)
        self.pos_arr = np.zeros(self.bins + 1)
        self.fp_arr = np.zeros(self.bins + 1)
        self.neg_arr = np.zeros(self.bins + 1)


class PRMetric():
    def __init__(self, nclass, bins):
        self.nclass = nclass
        self.bins = bins
        self.reset()

    def update(self, preds, labels):
        for iBin in range(self.bins+1):
            score_thresh = (iBin + 0.0) / self.bins
            i_tp, i_fp, i_tn, i_fn,  i_pos, i_neg = cal_tp_pos_fp_neg(preds, labels,  score_thresh)

            self.tp_arr[iBin] += i_tp
            self.pos_arr[iBin] += i_pos
            self.fp_arr[iBin] += i_fp
            self.neg_arr[iBin] += i_neg

    def get(self):
        P_rates = self.tp_arr / (self.tp_arr + self.fp_arr + 0.001)
        R_rates = self.tp_arr / (self.pos_arr + 0.001)

        return P_rates, R_rates

    def reset(self):
        self.tp_arr = np.zeros(self.bins + 1)
        self.pos_arr = np.zeros(self.bins + 1)
        self.fp_arr = np.zeros(self.bins + 1)
        self.neg_arr = np.zeros(self.bins + 1)












