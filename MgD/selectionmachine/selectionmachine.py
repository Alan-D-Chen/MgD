# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @Site: 
# @File: selectionmachine.py
# @Author: Alan D.Chen
# @E-mail: chense_mail@126.com
# @Time: 2021,04 08
# ---

import torch
import torch.nn as nn
import cv2
import numpy as np
from efficientdet.ious import calc_iou, calc_cdiou, calc_giou

def select_machine(anchor, bbox_annotation):
    ## Multi level selection machine ##

    IoU = calc_iou(anchor, bbox_annotation)
    CDIoU, diou = calc_cdiou(anchor, bbox_annotation)
    GIoU = calc_giou(anchor, bbox_annotation)

    IoU_max, IoU_argmax = torch.max(IoU, dim=1)
    CDIoU_max, CDIoU_argmax = torch.max(CDIoU, dim=1)
    # GIoU_max, GIoU_argmax = torch.max(GIoU, dim=1)

    positive_indices = torch.ge(IoU_max, 0.5)
    positive_indices2 = torch.ge(CDIoU_max, CDIoU_max.mean())
    # positive_indices3 = torch.ge(GIoU_max, GIoU_max.mean())

    for i in range(len(positive_indices)):
        if positive_indices2[i] and GIoU[i] > GIoU.mean():
            positive_indices[i] = 1

    return IoU, CDIoU, GIoU, positive_indices, IoU_max, IoU_argmax, diou

def Tselect_machine(anchor, bbox_annotation):
    ## Multi level selection machine 2 ##

    IoU = calc_iou(anchor, bbox_annotation)
    CDIoU, diou = calc_cdiou(anchor, bbox_annotation)
    GIoU = calc_giou(anchor, bbox_annotation)

    IoU_max, IoU_argmax = torch.max(IoU, dim=1)
    CDIoU_max, CDIoU_argmax = torch.max(CDIoU, dim=1)
    #GIoU_max, GIoU_argmax = torch.max(GIoU, dim=1)

    positive_indices = torch.ge(IoU_max, 0.5)
    positive_indices2 = torch.ge(CDIoU_max, CDIoU_max.mean())
    #positive_indices3 = torch.ge(GIoU_max, GIoU_max.mean())

    for i in range(len(positive_indices)):
        if 0.5 > IoU_max[i] > 0.4:
            if positive_indices2[i]:
                positive_indices[i] = 1

    return IoU, CDIoU, GIoU, positive_indices, IoU_max, IoU_argmax, diou

def Gselect_machine(anchor, bbox_annotation):
    ## Multi level selection machine 2 ##

    IoU = calc_iou(anchor, bbox_annotation)
    CDIoU, diou = calc_cdiou(anchor, bbox_annotation)
    GIoU = calc_giou(anchor, bbox_annotation)

    IoU_max, IoU_argmax = torch.max(IoU, dim=1)
    CDIoU_max, CDIoU_argmax = torch.max(CDIoU, dim=1)
    #GIoU_max, GIoU_argmax = torch.max(GIoU, dim=1)

    positive_indices = torch.ge(IoU_max, 0.5)
    positive_indices2 = torch.ge(CDIoU_max, CDIoU_max.mean())
    #positive_indices3 = torch.ge(GIoU_max, GIoU_max.mean())

    for i in range(len(positive_indices)):
        if 0.5 > IoU_max[i] > 0.4:
            if positive_indices2[i] or GIoU[i] > GIoU.mean():
                positive_indices[i] = 1

    return IoU, CDIoU, GIoU, positive_indices, IoU_max, IoU_argmax, diou