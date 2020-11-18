#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@File   :   example.py
@Author :   馒头
@Time   :   2020/10/01
@Contact:   songjian@westlake.edu.cn
@intro  :   demo
'''
import sys
import torch
import pandas as pd

import predict
import predifine
from model import Frag_GRU

if __name__ == '__main__':
    simple_seq = sys.argv[1]
    pr_charge = int(sys.argv[2])

    # query precursor
    df = pd.DataFrame({'simple_seq': [simple_seq],
                       'pr_charge': [pr_charge]})
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
    pred_annos = ';'.join([predifine.g_idx_to_anno[i] for (i, anno) in enumerate(m_fg_prob[0]) if anno == 1])

    # print
    print(simple_seq + '_' + str(pr_charge) + ': ' + pred_annos)
