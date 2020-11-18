#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@File   :   model.py
@Author :   song
@Time   :   2020/10/01
@Contact:   songjian@westlake.edu.cn
@intro  :   Bi-GRU + self-intention
'''
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence

import predifine

class Frag_GRU(nn.Module):

    def __init__(self):
        super(Frag_GRU, self).__init__()

        self.aa_embed = nn.Embedding(22, 32)
        self.charge_embed = nn.Embedding(4, 32)

        self.gru = nn.GRU(batch_first=True,
                          bidirectional=True,
                          num_layers=2,
                          dropout=0.5,
                          input_size=32,
                          hidden_size=32)

        # attention
        self.attention = nn.Linear(64, 32)
        self.context = nn.Linear(32, 1, bias=False)

        self.fc = nn.Linear(64 + 32, len(predifine.g_anno_to_idx))

    def forward(self, batch_padded, batch_lens, batch_charge):

        aa_embedding = self.aa_embed(batch_padded)

        # pack and pad
        outputs = pack_padded_sequence(aa_embedding,
                                       batch_lens,
                                       batch_first=True,
                                       enforce_sorted=False) # unsort with false
        self.gru.flatten_parameters()
        outputs, _ = self.gru(outputs, None)

        # self attention
        att_w = torch.tanh(self.attention(outputs.data))
        att_w = self.context(att_w).squeeze(1)
        max_w = att_w.max()
        att_w = torch.exp(att_w - max_w)
        att_w, _ = pad_packed_sequence(PackedSequence(data=att_w,
                                                      batch_sizes=outputs.batch_sizes,
                                                      sorted_indices=outputs.sorted_indices,
                                                      unsorted_indices=outputs.unsorted_indices),
                                       batch_first=True)
        alphas = att_w / torch.sum(att_w, dim=1, keepdim=True)

        outputs, _ = pad_packed_sequence(outputs, batch_first=True)
        outputs = (outputs * alphas.unsqueeze(2)).sum(dim=1)  # [batch_size, out_dim]

        # cat charge embed
        charge_embed = self.charge_embed(batch_charge)
        result = torch.cat((outputs, charge_embed), dim=1)

        result = self.fc(result)

        return result