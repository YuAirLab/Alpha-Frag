#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@File   :   predifine.py
@Author :   Song
@Time   :   2020/10/01
@Contact:   songjian@westlake.edu.cn
@intro  :
'''
# amino acid to idx
g_aa_to_idx = {'A':1, 'C':2, 'D':3, 'E':4, 'F':5, 'G':6, 'H':7,
               'I':8, 'K':9, 'L':10, 'M':11, 'm':12,'N':13, 'P':14,
               'Q':15, 'R':16, 'S':17, 'T':18, 'V':19, 'W':20, 'Y':21,
               'x':0}

g_aa_to_mass = {'A':89.0476792233, 'C':178.04121404900002, 'D':133.0375092233, 'E':147.05315928710002,
                'F':165.07897935090006, 'G':75.0320291595, 'H':155.06947728710003, 'I':131.0946294147,
                'K':146.10552844660003, 'L':131.0946294147, 'M':149.05105008089998, 'm':165.04596508089998,
                'N':132.0534932552, 'P':115.06332928709999, 'Q':146.06914331900003, 'R':174.11167644660003,
                'S':105.0425942233, 'T':119.05824428710001, 'V':117.0789793509, 'W':204.0898783828,
                'Y':181.07389435090005,
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

ppm = 20.