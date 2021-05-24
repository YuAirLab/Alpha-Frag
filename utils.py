#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@File   :   utils.py
@Author :   Song
@Time   :   2021/3/15 9:52
@Contact:   songjian@westlake.edu.cn
@intro  :
'''
import numpy as np
import operator
import predifine
from numba import jit
from model import Frag_GRU
import torch
import math

try:
    profile
except:
    profile = lambda x: x


class Set_To_Scores():
    # @profile
    def __init__(self, measure, pred, seq_len, pr_charge):
        self.seq_len = seq_len
        self.pr_charge = pr_charge

        self.A = set(measure)
        y_idx = (measure.astype('<U1') == 'y')
        self.Ay = set(measure[y_idx])
        self.Ab = set(measure[~y_idx])

        self.B = set(pred)
        y_idx = (pred.astype('<U1') == 'y')
        self.By = set(pred[y_idx])
        self.Bb = set(pred[~y_idx])

    def get_intersect_scores(self):
        # return: np.array_9(3*3=9)
        scores_v = []

        S = self.A & self.B
        s1 = len(S)
        s2 = s1 / (2 * self.seq_len * self.pr_charge)
        scores_v.extend([s2])

        S = self.Ay & self.By
        s1 = len(S)
        s2 = s1 / (self.seq_len * self.pr_charge)
        scores_v.extend([s2])

        S = self.Ab & self.Bb
        s1 = len(S)
        s2 = s1 / (self.seq_len * self.pr_charge)
        scores_v.extend([s2])

        return scores_v


class Spectra_SA():
    def __init__(self, pred, measure, annos):
        y_idx = (annos.astype('<U1') == 'y')
        self.pred_yb = pred
        self.pred_y = pred[y_idx]
        self.pred_b = pred[~y_idx]
        self.measure_yb = measure
        self.measure_y = measure[y_idx]
        self.measure_b = measure[~y_idx]

        e = 0.000001
        self.pred_yb_max = self.pred_yb.max() + e
        self.pred_y_max = self.pred_y.max() + e
        self.pred_b_max = self.pred_b.max() + e

        self.measure_yb_max = self.measure_yb.max() + e
        self.measure_y_max = self.measure_y.max() + e
        self.measure_b_max = self.measure_b.max() + e

    def get_sa(self):
        sa_yb = calculate_sa(self.pred_yb, self.pred_yb_max,
                             self.measure_yb, self.measure_yb_max)
        sa_y = calculate_sa(self.pred_y, self.pred_y_max,
                            self.measure_y, self.measure_y_max)
        sa_b = calculate_sa(self.pred_b, self.pred_b_max,
                            self.measure_b, self.measure_b_max)
        scores = [sa_yb, sa_y, sa_b]
        return scores


def load_model(gpu_id):
    model = Frag_GRU()
    device = torch.device('cuda:{}'.format(gpu_id))
    model.load_state_dict(torch.load('model.pt', map_location=device))
    model = model.to(device)
    model.eval()
    return model


def pad_seq_to_mass(simple_seq, seq_len):
    seq_len_max = seq_len.max()
    assert seq_len_max <= 30

    pad_len = max(seq_len) - seq_len
    pad = pad_len.apply(lambda v: 'x' * v)
    pad_seq = simple_seq + pad

    s = pad_seq.str.cat()
    s = list(s)

    f = operator.itemgetter(*s)
    paded_mass = f(predifine.g_aa_to_mass)

    return paded_mass, seq_len_max


@jit(nopython=True)
def calculate_possible_mz(masses, pr_charge, out):
    '''
        return out: 4 * len(masses):
            y1_1, y1_2, y2_1, y2_2, b1_1, b1_2, b2_1, b2_2...
    '''
    mass_proton = 1.007276466771
    mass_h2o = 18.0105650638
    mass_neutron = 1.0033548378

    pr_len = len(masses)
    fg_charge_max = min(pr_charge, 2)

    masses_forward = masses.cumsum()
    masses_backward = masses[::-1].cumsum()

    idx = 0
    for y_or_b in [1, 0]:
        for fg_len in range(1, pr_len):
            for fg_charge in range(1, fg_charge_max + 1):
                if y_or_b == 1:  # b
                    product_mass = masses_forward[fg_len - 1] - (fg_len - 1) * mass_h2o - mass_h2o
                else:  # y
                    product_mass = masses_backward[fg_len - 1] - (fg_len - 1) * mass_h2o
                for neutron_num in (-1, 0, 1, 2):
                    product_mz = (product_mass + fg_charge * mass_proton + neutron_num * mass_neutron) / fg_charge
                    out[idx] = product_mz
                    idx = idx + 1


@profile
@jit(nopython=True)
def get_exist_idx(intensity_nearest, good_idx):
    '''
            1. M > M+1 > M+2
            2. M > M-1
    '''
    assert len(intensity_nearest) % 4 == 0
    i = 0
    while i < len(intensity_nearest):
        idx = i // 4
        mz_left = intensity_nearest[i]
        mz = intensity_nearest[i + 1]
        mz_right_1 = intensity_nearest[i + 2]
        mz_right_2 = intensity_nearest[i + 3]
        if (mz > mz_right_1 > mz_right_2 > 0) and (mz > mz_left):
            good_idx[idx] = True
        else:
            good_idx[idx] = False
        i += 4


@jit(nopython=True)
def calculate_sa(x, x_max, y, y_max):
    e = 0.000001
    s = 0.
    norm_x = 0.
    norm_y = 0.
    for i in range(len(x)):
        xx = x[i] / x_max
        yy = y[i] / y_max
        norm_x += xx ** 2
        norm_y += yy ** 2
        s += xx * yy
    norm_x = math.sqrt(norm_x) + e
    norm_y = math.sqrt(norm_y) + e
    sa = 1 - 2 * math.acos(s / norm_x / norm_y) / np.pi
    return sa
