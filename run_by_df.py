#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@File   :   example.py
@Author :   馒头
@Time   :   2020/10/01
@Contact:   songjian@westlake.edu.cn
@intro  :   demo
'''
import torch
import pandas as pd

import predict
import predifine
from model import Frag_GRU

# query precursor
df = pd.DataFrame({'simple_seq': ['AAACEEK'],
                   'pr_charge': [2]})
df['seq_len'] = df.simple_seq.str.len()

# load model
model = Frag_GRU()
device = torch.device('cuda')
model.load_state_dict(torch.load('model.pt', map_location=device))
model = model.to(device)
model.eval()

# predict presence probability
m_fg_prob = predict.predict_anno(df, model)

# idx --> anno
for idx in range(len(df)):

    seq = df['simple_seq'][idx]
    pr_charge = df['pr_charge'][idx]
    pred_annos = ';'.join([predifine.g_idx_to_anno[i] for (i, anno) in enumerate(m_fg_prob[idx]) if anno == 1])

    print(seq + '_' + str(pr_charge) + ': ' + pred_annos)
