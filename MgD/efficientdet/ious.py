# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @Site: 
# @File: ious.py
# @Author: Alan D.Chen
# @E-mail: chense_mail@126.com
# @Time: 2021, 04, 01
# ---

import torch
import torch.nn as nn
import cv2
import numpy as np

def calc_iou(a, b):
    # a(anchor) [boxes, (y1, x1, y2, x2)]
    # b(gt, coco-style) [boxes, (x1, y1, x2, y2)]
    # print("anchor:\n", a)
    # print("gt:\n",b)

    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    iw = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 0])
    ih = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 1])
    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)
    ua = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1) + area - iw * ih
    ua = torch.clamp(ua, min=1e-8)
    intersection = iw * ih
    IoU = intersection / ua

    return IoU

def calc_cdiou(a, b):
    """
    calculate cdiou
    :param box_1: (x0, y0, x1, y1)
    :param box_2: (x0, y0, x1, y1)
    :return: value of cdiou
    """
   # Get the minimum bounding box/rectangle (including region proprosal and ground truth) #
   #  a1 = torch.cat((a[:, 1], a[:, 0]), dim=0)
   #  a2 = torch.cat((a[:, 3], a[:, 2]), dim=0)
   #  a = torch.cat((a1, a2), dim=0)

    n = a.cuda()
    m = b.cuda()
    nd0 = n.shape[0]
    md0 = m.shape[0]
    nss = n.unsqueeze(1)
    nss = nss.expand(nd0, md0, 4)

    mss = m.unsqueeze(0)
    mss = mss.expand(nd0, md0, 4)
    nms = torch.cat((nss, mss), dim=2)

    A = nms[:, :, [0, 4]]
    B = nms[:, :, [1, 5]]
    C = nms[:, :, [2, 6]]
    D = nms[:, :, [3, 7]]
    Am = torch.min(A, 2)[0]
    Bm = torch.min(B, 2)[0]
    Cm = torch.max(C, 2)[0]
    Dm = torch.max(D, 2)[0]

    XY = torch.zeros([Am.shape[0], Am.shape[1], 4]).cuda()
    XY[:, :, 0] = Am
    XY[:, :, 1] = Bm
    XY[:, :, 2] = Cm
    XY[:, :, 3] = Dm
    XYx = (XY[:, :, [2, 3]] - XY[:, :, [0, 1]]) ** 2
    XxY = XYx[:, :, 0] + XYx[:, :, 1]
    XYs = XxY.sqrt()  ###########################-> to get square root(diagonal of MBR)

    # The average distance between GT and RP is obtained #

    n0 = n[:, [0, 3]]  # .unsqueeze(1)
    n1 = n[:, [1, 2]]  # .unsqueeze(1)
    ns = torch.cat((n, n0, n1), dim=1)

    m0 = m[:, [0, 3]]  # .unsqueeze(1)
    m1 = m[:, [1, 2]]  # .unsqueeze(1)
    ms = torch.cat((m, m0, m1), dim=1)

    ns = ns.unsqueeze(1)
    ms = ms.unsqueeze(0)

    tmp = (ns - ms) ** 2
    tmps = tmp[:, :, 0::2] + tmp[:, :, 1::2]
    tmp = tmps.sqrt()
    tmp = torch.mean(tmp, dim=2, keepdim=False)
    tmpx = (tmp / XYs).cuda()

    return 1-tmpx, tmpx

def calc_giou1(a, b):
    """
    calculate giou
    :param box_1: (x0, y0, x1, y1)
    :param box_2: (x0, y0, x1, y1)
    :return: value of giou
    """
    box_1 = a
    box_2 = b

    # calculate area of each box
    area_1 = (box_1[:,2] - box_1[:,0]) * (box_1[:,3] - box_1[:,1])
    area_2 = (box_2[:,2] - box_2[:,0]) * (box_1[:,3] - box_1[:,1])

    # calculate minimum external frame
    area_c = (max(box_1[:,2], box_2[:,2]) - min(box_1[:,0], box_2[:,0])) * (max(box_1[:,3], box_2[:,3]) - min(box_1[:,1], box_2[:,1]))

    # find the edge of intersect box
    top = max(box_1[:,0], box_2[:,0])
    left = max(box_1[:,1], box_2[:,1])
    bottom = min(box_1[:,3], box_2[:,3])
    right = min(box_1[:,2], box_2[:,2])

    # calculate the intersect area
    area_intersection = (right - left) * (bottom - top)

    # calculate the union area
    area_union = area_1 + area_2 - area_intersection

    # calculate iou
    iou = float(area_intersection) / area_union

    # calculate giou(iou - (area_c - area_union)/area_c)
    giou = iou - float((area_c - area_union)) / area_c

    return giou

def calc_giou2(boxes1,boxes2):
    '''
    cal GIOU of two boxes or batch boxes
    such as: (1)
            boxes1 = np.asarray([[0,0,5,5],[0,0,10,10],[15,15,25,25]])
            boxes2 = np.asarray([[5,5,10,10]])
            and res is [-0.49999988  0.25       -0.68749988]
            (2)
            boxes1 = np.asarray([[0,0,5,5],[0,0,10,10],[0,0,10,10]])
            boxes2 = np.asarray([[0,0,5,5],[0,0,10,10],[0,0,10,10]])
            and res is [1. 1. 1.]
    :param boxes1:[xmin,ymin,xmax,ymax] or
                [[xmin,ymin,xmax,ymax],[xmin,ymin,xmax,ymax],...]
    :param boxes2:[xmin,ymin,xmax,ymax]
    :return:
    '''

    # cal the box's area of boxes1 and boxess
    boxes1Area = (boxes1[...,2]-boxes1[...,0])*(boxes1[...,3]-boxes1[...,1])
    boxes2Area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    # ===========cal IOU=============#
    #cal Intersection
    left_up = np.maximum(boxes1[...,:2],boxes2[...,:2])
    right_down = np.minimum(boxes1[...,2:],boxes2[...,2:])

    inter_section = np.maximum(right_down-left_up,0.0)
    inter_area = inter_section[...,0] * inter_section[...,1]
    union_area = boxes1Area+boxes2Area-inter_area
    ious = np.maximum(1.0*inter_area/union_area,np.finfo(np.float32).eps)

    # ===========cal enclose area for GIOU=============#
    enclose_left_up = np.minimum(boxes1[..., :2], boxes2[..., :2])
    enclose_right_down = np.maximum(boxes1[..., 2:], boxes2[..., 2:])
    enclose = np.maximum(enclose_right_down - enclose_left_up, 0.0)
    enclose_area = enclose[..., 0] * enclose[..., 1]

    # cal GIOU
    gious = ious - 1.0 * (enclose_area - union_area) / enclose_area

    return gious

def calc_giou3(bboxes1, bboxes2):
    # a = bboxes1
    #     # a1 = torch.cat((a[:, 1], a[:, 0]), dim=0)
    #     # a2 = torch.cat((a[:, 3], a[:, 2]), dim=0)
    #     # a = torch.cat((a1, a2), dim=0)
    #     # bboxes1 = a

    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    ious = torch.zeros((rows, cols))
    if rows * cols == 0:
        return ious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        ious = torch.zeros((cols, rows))
        exchange = True
    area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (
        bboxes1[:, 3] - bboxes1[:, 1])
    area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (
        bboxes2[:, 3] - bboxes2[:, 1])

    inter_max_xy = torch.min(bboxes1[:, 2:],bboxes2[:, 2:])
    #inter_max_xy = torch.min(bboxes1[:, [2,3]], bboxes2[:, 2:])

    inter_min_xy = torch.max(bboxes1[:, :1],bboxes2[:, :1])

    out_max_xy = torch.max(bboxes1[:, 2:],bboxes2[:, 2:])

    out_min_xy = torch.min(bboxes1[:, :2],bboxes2[:, :2])

    inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)
    inter_area = inter[:, 0] * inter[:, 1]
    outer = torch.clamp((out_max_xy - out_min_xy), min=0)
    outer_area = outer[:, 0] * outer[:, 1]
    union = area1+area2-inter_area
    closure = outer_area

    ious = inter_area / union - (closure - union) / closure
    ious = torch.clamp(ious,min=-1.0,max = 1.0)
    if exchange:
        ious = ious.T
    return ious

def calc_giou(a, b):

    ## calculation for GIoU and loss

    # print("anchor:\n", b)
    # print("gt:\n", a)

    c = torch.empty(len(a)-len(b), 4) * 0
    c = c.cuda()
    b = torch.cat((b, c), 0)

    # print("c:\n", c)
    # print("type of c:", type(c))
    # print("c.shape:", c.shape)
    # print("length of c:", len(c))
    #
    # print("anchor:\n", b)
    # print("type of anchor:", type(b))
    # print("anchor.shape:", b.shape)
    # print("length of anchor:", len(b))
    #
    # print("gt:\n", a)
    # print("type of gt:", type(a))
    # print("gt.shape:", a.shape)
    # print("length of gt:", len(a))

    pred_boxes = b
    pred_x1 = pred_boxes[:, 0]
    pred_y1 = pred_boxes[:, 1]
    pred_x2 = pred_boxes[:, 2]
    pred_y2 = pred_boxes[:, 3]
    pred_x2 = torch.max(pred_x1, pred_x2)
    pred_y2 = torch.max(pred_y1, pred_y2)
    pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)

    gt_boxes = a
    target_x1 = gt_boxes[:, 0]
    target_y1 = gt_boxes[:, 1]
    target_x2 = gt_boxes[:, 2]
    target_y2 = gt_boxes[:, 3]
    target_area = (target_x2 - target_x1) * (target_y2 - target_y1)

    x1_intersect = torch.max(pred_x1, target_x1)
    y1_intersect = torch.max(pred_y1, target_y1)
    x2_intersect = torch.min(pred_x2, target_x2)
    y2_intersect = torch.min(pred_y2, target_y2)
    area_intersect = torch.zeros(pred_x1.size()).to(torch.device("cuda"))
    mask = (y2_intersect > y1_intersect) * (x2_intersect > x1_intersect)
    area_intersect[mask] = (x2_intersect[mask] - x1_intersect[mask]) * (y2_intersect[mask] - y1_intersect[mask])

    x1_enclosing = torch.min(pred_x1, target_x1)
    y1_enclosing = torch.min(pred_y1, target_y1)
    x2_enclosing = torch.max(pred_x2, target_x2)
    y2_enclosing = torch.max(pred_y2, target_y2)
    area_enclosing = (x2_enclosing - x1_enclosing) * (y2_enclosing - y1_enclosing) + 1e-7

    area_union = pred_area + target_area - area_intersect + 1e-7
    ious = area_intersect / area_union
    gious = ious - (area_enclosing - area_union) / area_enclosing
    # print("gious:\n", gious)
    # print("type of gious:", type(gious))
    # print("gious.shape:", gious.shape)
    # print("length of gious:", len(gious))
    return gious