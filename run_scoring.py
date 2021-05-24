#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@File   :   add_scores.py
@Author :   Song
@Time   :   2020/10/10 16:07
@Contact:   songjian@westlake.edu.cn
@intro  :   add qualitative and quantitative scores to OpenSWATH output
'''
import numpy as np
import pandas as pd
import ms
from pathlib import Path
import predifine
import sys
import time

import utils
from predict import get_pred_probability, get_theory_mask
import predifine

try:
    profile
except:
    profile = lambda x: x

@profile
def get_scores(row):
    # apex rt -- ms2 scan -- pred anno -- pred mz -- searchsort
    row_idx = row.name
    if row_idx % 10000 == 0:
        print('\r{:<30}{} finished'.format('[Scoring]', row_idx), end='', flush=True)

    seq_len = seq_len_v[row_idx]
    pr_charge = pr_charge_v[row_idx]
    apex_rt = apex_rt_v[row_idx]
    pr_mz = pr_mz_v[row_idx]

    mask = mask_dict[(pr_charge, seq_len)]
    mask = mask.astype(bool)
    all_annos = np.array(list(predifine.g_anno_to_idx.keys()))

    # # pred anno and probability
    pred_intensity = m_fg_prob[row_idx]
    pred_annos = all_annos[pred_intensity > 0.5]
    pred_intensity = pred_intensity[mask]

    all_possible_annos = all_annos[mask]
    sort_annos_idx = np.argsort(all_possible_annos)  # first b, then y
    all_possible_annos = all_possible_annos[sort_annos_idx]
    pred_intensity = pred_intensity[sort_annos_idx]

    # possible fg mz
    aa_mass = m_aa_mass[row_idx][0:seq_len]
    fg_mz = np.zeros(len(all_possible_annos) * 4)
    utils.calculate_possible_mz(aa_mass, pr_charge, fg_mz)

    # scan
    num_windows = len(mzml.SwathSettings) - 1
    ms2_win_idx = np.digitize(pr_mz, mzml.SwathSettings)

    ms2_all_rts = mzml.all_rt[ms2_win_idx:: (num_windows + 1)]
    ms2_all_mz = mzml.all_mz[ms2_win_idx:: (num_windows + 1)]
    ms2_all_intensity = mzml.all_intensity[ms2_win_idx:: (num_windows + 1)]

    scan_idx = np.abs(ms2_all_rts - apex_rt).argmin()

    if scan_idx >= len(ms2_all_mz):
        scan_idx = len(ms2_all_mz) - 1

    scan_mz = ms2_all_mz[scan_idx]
    scan_intensity = ms2_all_intensity[scan_idx]

    idx_insert = np.searchsorted(scan_mz, fg_mz)
    try:
        idx_nearest = idx_insert - (np.abs(fg_mz - scan_mz[idx_insert - 1]) < np.abs(
            fg_mz - scan_mz[idx_insert]))
    except:
        idx_insert[idx_insert == 0] = 1
        idx_insert[idx_insert == len(scan_mz)] = len(scan_mz) - 1
        idx_nearest = idx_insert - (np.abs(fg_mz - scan_mz[idx_insert - 1]) < np.abs(
            fg_mz - scan_mz[idx_insert]))
    mz_nearest = scan_mz[idx_nearest]  # 最近邻
    intensity_nearest = scan_intensity[idx_nearest]

    ppm_bad_idx = np.abs(mz_nearest - fg_mz) * 1000000. > (predifine.ppm * fg_mz)
    intensity_nearest[ppm_bad_idx] = 0.

    # 综合四个同位离子强度判断存在性
    good_idx = np.array([False] * len(all_possible_annos))
    utils.get_exist_idx(intensity_nearest, good_idx)

    measure_annos = all_possible_annos[good_idx]
    qual_scores = utils.Set_To_Scores(measure_annos, pred_annos, seq_len, pr_charge)
    qual_scores = qual_scores.get_intersect_scores()

    # quan scores
    measure_intensity = intensity_nearest[1::4]
    measure_intensity[~good_idx] = 0.
    scores_sa = utils.Spectra_SA(pred_intensity, measure_intensity, all_possible_annos)
    scores_sa = scores_sa.get_sa()

    # combination
    qual_scores.extend(scores_sa)
    return qual_scores

def choose_dataset(fold_path):
    fold_path = Path(fold_path)
    osw_path = fold_path / 'osw.tsv'
    mzml_path = list(fold_path.glob('*.mzML'))[0]
    qual_output_path = fold_path / 'osw_alpha_scores.tsv'
    quan_output_path = fold_path / 'osw_alpha_scores_sa.tsv'
    combine_output_path = fold_path / 'osw_merge_alpha_alpha.tsv'
    return osw_path, mzml_path, qual_output_path, quan_output_path, combine_output_path

def load_osw_df(fpath):
    df = pd.read_csv(fpath, sep='\t')
    df['simple_seq'] = df['FullPeptideName'].replace(['C\(UniMod:4\)', 'M\(UniMod:35\)'], ['C', 'm'], regex=True)
    df = df[~df.simple_seq.str.contains('[BJOUXZ]', regex=True)]
    df['seq_len'] = df['simple_seq'].str.len()
    df['pr_charge'] = df['Charge']
    df = df[(df.pr_charge <= 4) & (df.seq_len >= 7) & (df.seq_len <= 30)]
    df = df.reset_index(drop=True)

    return df

if __name__ == '__main__':
    t0 = time.time()
    ws = sys.argv[1]
    print('Alpha-Frag scoring for ', ws)
    osw_path, mzml_path, qual_output_path, quan_output_path, combine_output_path = choose_dataset(ws)

    # load data
    df = load_osw_df(osw_path)
    mzml = ms.load_ms(mzml_path, type='DIA')

    # load model
    model = utils.load_model(gpu_id=0)

    # predict
    m_fg_prob = get_pred_probability(df, model)

    mask_dict = get_theory_mask(pr_charge_range=[1, 2, 3, 4], seq_len_range=range(7, 31))
    m_aa_mass, _ = utils.pad_seq_to_mass(df.simple_seq, df.seq_len)
    m_aa_mass = np.array(m_aa_mass).reshape(len(df), -1)

    # score
    seq_len_v = df['seq_len'].to_numpy()
    pr_charge_v = df['pr_charge'].to_numpy()
    apex_rt_v = df['RT'].to_numpy()
    pr_mz_v = df['m/z'].to_numpy()
    scores = df.apply(get_scores, axis=1, result_type='expand').values
    print('\r{:<30}{} finished'.format('[Scoring]', len(df)), end='', flush=True)
    print('\r')

    scores = scores.reshape(len(df), -1)
    for i in range(3):
        col = 'var_qual_score_' + str(i)
        df[col] = scores[:, i]
    for i in range(3, 6):
        col = 'var_quan_score_' + str(i)
        df[col] = scores[:, i]

    print(f'time: {(time.time() - t0) / 60.:.2f}min')

    # df.drop(df.columns[-3:], axis=1).to_csv(qual_output_path, index=False, sep='\t')
    # df.drop(df.columns[-6:-3], axis=1).to_csv(quan_output_path, index=False, sep='\t')
    df.to_csv(combine_output_path, index=False, sep='\t')
