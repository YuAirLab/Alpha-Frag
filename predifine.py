#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@File   :   predifine.py
@Author :   song
@Time   :   2020/10/01
@Contact:   songjian@westlake.edu.cn
@intro  :
'''
# amino acid to idx
g_aa_to_idx = {'A':1, 'C':2, 'D':3, 'E':4, 'F':5, 'G':6, 'H':7,
               'I':8, 'K':9, 'L':10, 'M':11, 'm':12,'N':13, 'P':14,
               'Q':15, 'R':16, 'S':17, 'T':18, 'V':19, 'W':20, 'Y':21,
               'x':0}

# ion label to idx: y1+ --> 0
g_anno_to_idx = {}
for fg_type in ['y', 'b']:
    for fg_len in range(1, 30):
        for fg_charge in [1, 2]:
            if fg_charge == 1:
                fg_anno = fg_type + str(fg_len) + '_1'
                g_anno_to_idx[fg_anno] = len(g_anno_to_idx)
            if (fg_charge == 2) and (fg_type in ['y', 'b']):
                fg_anno = fg_type + str(fg_len) + '_2'
                g_anno_to_idx[fg_anno] = len(g_anno_to_idx)

g_idx_to_anno = {v: k for k, v in g_anno_to_idx.items()}