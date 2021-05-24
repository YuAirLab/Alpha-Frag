#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@File   :   predict.py
@Author :   Song
@Time   :   2020/10/01
@Contact:   songjian@westlake.edu.cn
@intro  :   predict
'''
import operator
import numpy as np
import torch
import predifine

def get_theory_mask(pr_charge_range, seq_len_range):
    mask_dict = {}
    for pr_charge in pr_charge_range:
        for seq_len in seq_len_range:
            key = (pr_charge, seq_len)
            mask = np.zeros(len(predifine.g_anno_to_idx))
            for i in range(len(mask)):
                anno = predifine.g_idx_to_anno[i]

                anno_split = anno.split('_')
                fg_len = int(anno_split[0][1:])
                fg_charge = int(anno_split[-1])

                if (fg_len < seq_len and fg_charge <= pr_charge and fg_charge <= 2):
                    mask[i] = 1
            mask_dict[key] = mask
    return mask_dict

def pad_seq_to_idx(simple_seq, seq_len):
    seq_len_max = seq_len.max()
    assert seq_len_max <= 30

    pad_len = max(seq_len) - seq_len
    pad = pad_len.apply(lambda v: 'x' * v)
    pad_seq = simple_seq + pad

    s = pad_seq.str.cat()
    s = list(s)

    f = operator.itemgetter(*s)
    paded_idx = f(predifine.g_aa_to_idx)

    return paded_idx, seq_len_max

def predict_batch_anno(df, model):
    paded_idx, seq_len_max = pad_seq_to_idx(df.simple_seq, df.seq_len)
    seq_len = df.seq_len.values

    # transfer to gpu
    device = torch.device('cuda')
    paded_idx = torch.tensor(paded_idx, dtype=torch.long, device=device)
    paded_idx = paded_idx.view(len(df), -1)
    seq_len = torch.tensor(seq_len, dtype=torch.int8, device=device)

    # charge-1,
    batch_charges = torch.tensor(df.pr_charge.to_numpy() - 1).long().to(device)

    # predict
    with torch.no_grad():
        output_frag = model(paded_idx, seq_len, batch_charges)
        output_frag = torch.sigmoid(output_frag)

        output_frag[output_frag > 0.5] = 1
        output_frag[output_frag <= 0.5] = 0

    output_frag = output_frag.cpu().numpy().astype(np.int8)

    return output_frag

def predict_anno(df, model):
    m = []

    mask_dict = get_theory_mask(pr_charge_range=[1, 2, 3, 4], seq_len_range=range(7, 31))
    mask = np.vstack(df.apply(lambda row: mask_dict[(row.pr_charge, row.seq_len)], axis=1)).astype(np.int8)

    for _, df_batch in df.groupby(df.index // 10000):
        df_batch = df_batch.reset_index(drop=True)
        batch_frag = predict_batch_anno(df_batch, model)
        m.append(batch_frag)
        torch.cuda.empty_cache()

    m = np.concatenate(m)
    m = m * mask

    return m

def get_pred_probability(df, model):
    mask_dict = get_theory_mask(pr_charge_range=[1, 2, 3, 4], seq_len_range=range(7, 31))
    pr_charge_v = df['pr_charge'].values
    seq_len_v = df['seq_len'].values
    key_v = list(zip(pr_charge_v, seq_len_v))

    f = operator.itemgetter(*key_v)
    mask = f(mask_dict)
    mask = np.vstack(mask)

    m = []
    for _, df_batch in df.groupby(df.index // 3000):
        df_batch = df_batch.reset_index(drop=True)
        paded_idx, seq_len_max = pad_seq_to_idx(df_batch.simple_seq, df_batch.seq_len)
        seq_len = df_batch.seq_len.values

        device = torch.device('cuda:{}'.format(0))
        paded_idx = torch.tensor(paded_idx, dtype=torch.long, device=device)
        paded_idx = paded_idx.view(len(df_batch), -1)
        seq_len = torch.tensor(seq_len, dtype=torch.int8, device=device)

        batch_charges = torch.tensor(df_batch.pr_charge.to_numpy() - 1).long().to(device)

        # predict
        with torch.no_grad():
            output_frag = model(paded_idx, seq_len, batch_charges)
            output_frag = torch.sigmoid(output_frag)

        batch_pred = output_frag.cpu().numpy()
        m.append(batch_pred)
        torch.cuda.empty_cache()
        print('\r{:<30}{} finished'.format('[Predicting]', sum(map(len, m))), end='', flush=True)
    print('\r')

    m = np.concatenate(m)
    m = m * mask

    return m