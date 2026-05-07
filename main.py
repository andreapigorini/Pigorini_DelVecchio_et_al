import pandas as pd
import os.path as op
import os
from fx_calculation import (import_data, run_gamma, run_gamma_habituation, run_lfp, mni2fsav_coords, map_annot,
                            reject_outliers, print_stats_per_area, print_results, df_from_stats, remove_spurious_ones,
                            run_gamma_and_lfp_examples, print_crossmodal_per_area, compute_ccep_connectivity)
from fx_plot import (surface_fsav, flatmap_fsav, continuous_maps_one_cond, pointplot_gamma_by_area,
                     continuous_maps_multi, continuous_maps_gamma_vs_lfp , quickflat_with_atlas)
import glob
import nibabel as nib
import numpy as np
from mne.transforms import apply_trans
import pickle
import matplotlib.colors as mcolors
import ast
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import re
import mne
import matplotlib
matplotlib.use('Qt5Agg')
import mne_connectivity
import pymatreader as pym
from natsort import natsorted
import cortex
from scipy.spatial import cKDTree
from neuromaps import transforms
from neuromaps import datasets
import pyvista as pv




# DEFINE PATHS and CREATE A SUBJECT LIST
path_base = '/home/andrea/data/Pigorini_DelVecchio_et_al/'  # please download the data from DOI:
path_imaging = '/home/andrea/imaging' # contains imaging data for plotting
path_original_data = op.join(path_base, 'per-seeg')
path_results = op.join(path_base, 'results')
subjects_dir = op.join(path_imaging, 'fs_subjects')  # contains a MNI125 surface
fname_affine = op.join(path_imaging,'misc/mni2fsav/mni2fsav_0GenericAffine.mat')  # to be changed
fname_warp = op.join(path_imaging, 'misc/mni2fsav/mni2fsav_1InverseWarp.nii.gz')  # to be changed
subj_list = pd.read_csv(op.join(path_original_data, 'participants.tsv'), sep='\t')['participant_id']



# CALCULATION of GAMMA and LFP ACTIVITY
# iterations over subjects and conditions takes a lot of computations (up to 3 days on a single PC) - we run them in parallel on https://www.indaco.unimi.it/
# -----------------------------------------------------------------------------------------------------------------------------------------------------------
conds = ['acoustictask_run-01', 'somatosensorytask_run-01', 'visualtask_run-01']
conds_bil = ['acoustictask-left_run-01', 'acoustictask-right_run-01',  'somatosensorytask-left_run-01', 'somatosensorytask-right_run-01', 'visualtask-bilat_run-01']
subjs_bil = ['sub-04', 'sub-06', 'sub-12', 'sub-14', 'sub-19', 'sub-22', 'sub-24', 'sub-27', 'sub-32', 'sub-33', 'sub-37', 'sub-40', 'sub-44', 'sub-46', 'sub-47', 'sub-59']
for subj in subj_list:
    if subj in subjs_bil:
        conds_sess = conds_bil
    else:
        conds_sess = conds
    for cond in conds_sess:
        subj_cond_fname = subj + '_task-' + cond

        # IMPORT DATA FROM BIDS
        path_original_data = op.join(path_base, 'per-seeg')
        imported_data = import_data(op.join(path_original_data, subj + '/seeg/'), subj_cond_fname)

        # RUN TIMEF, SAVE GAMMA PROFILE and CALCULATE STATISTICS for GAMMA
        path_save_gamma = op.join(path_base, 'gamma_analyses')
        os.makedirs(path_save_gamma, exist_ok=True)
        run_gamma(imported_data, subj, cond, path_save_gamma)

        # SAVE EVOKED LFP and CALCULATE STATISTICS for MONOPOLAR LFP
        path_save_lfp = op.join(path_base, 'lfp_analyses')
        os.makedirs(path_save_lfp, exist_ok=True)
        run_lfp(imported_data, subj, cond, path_save_lfp)
# -----------------------------------------------------------------------------------------------------------------------------------------------------------
# end of iteration






# IMPORT ALL CONTACTS COORDINATES
all_subj_coords = pd.DataFrame(columns=['subj', 'ez', 'ch_name', 'hemis', 'x_norm_mri', 'y_norm_mri', 'z_norm_mri'])
for subj in subj_list:
    path_subj = op.join(path_original_data, subj + '/seeg/')
    subj_coords = pd.read_csv(glob.glob(os.path.join(path_subj, '*_electrodes.tsv'))[0], sep='\t')

    subj_coords_df = pd.DataFrame({
        'subj': subj,
        'ez': 0,
        'ch_name': subj_coords['name'],
        'hemis': subj_coords['name'].apply(lambda x: 'lh' if "'" in x else 'rh'),
        'x_norm_mri': subj_coords['x'] * 1000,
        'y_norm_mri': subj_coords['y'] * 1000,
        'z_norm_mri': subj_coords['z'] * 1000
    })
    all_subj_coords = pd.concat([all_subj_coords, subj_coords_df], ignore_index=True)
all_subj_coords['ez'] = all_subj_coords['ez'].astype(int)
coords_ras_norm = all_subj_coords[['x_norm_mri', 'y_norm_mri', 'z_norm_mri']]
fname_mni = op.join(subjects_dir, 'mni152', 'mri', 'T1.mgz')
mni = nib.load(fname_mni)
cras_mni = mni.header['Pxyz_c']
trans_mri2surf = np.array([[1, 0, 0, -cras_mni[0]],
                           [0, 1, 0, -cras_mni[1]],
                           [0, 0, 1, -cras_mni[2]],
                           [0, 0, 0, 1]])
coords_ras_norm_surf_arr = apply_trans(trans_mri2surf, coords_ras_norm.to_numpy())
all_subj_coords[['x_norm_surf', 'y_norm_surf', 'z_norm_surf']] = coords_ras_norm_surf_arr
coords_for_cont_maps = mni2fsav_coords(coords_ras_norm, fname_affine, fname_warp)
all_subj_coords[['x_norm_fsav', 'y_norm_fsav', 'z_norm_fsav']] = coords_for_cont_maps
all_subj_coords.loc[:, 'unique_ch_names'] = all_subj_coords['subj'] + '_' + all_subj_coords['ch_name']
areas = map_annot(all_subj_coords.rename(columns={"unique_ch_names": "name"}), subjects_dir, 'desikan')
glasser = map_annot(all_subj_coords.rename(columns={"unique_ch_names": "name"}), subjects_dir, 'glasser')
lobes = map_annot(all_subj_coords.rename(columns={"unique_ch_names": "name"}), subjects_dir, 'lobe')
all_subj_coords_plot_all = all_subj_coords.copy()
areas = areas.rename(columns={"name" : "unique_ch_names", "desikan" : "area"})
lobes = lobes.rename(columns={"name" : "unique_ch_names"})
glasser = glasser.rename(columns={"name" : "unique_ch_names"})
all_subj_coords = all_subj_coords.merge(lobes[['unique_ch_names', 'lobe']], on='unique_ch_names', how='left')
all_subj_coords = all_subj_coords.merge(areas[['unique_ch_names', 'area']], on='unique_ch_names', how='left')
all_subj_coords = all_subj_coords.merge(glasser[['unique_ch_names', 'glasser']], on='unique_ch_names', how='left')
df_coords = all_subj_coords.copy()
df_coords.rename(columns={'x_norm_fsav': 'x_fsav', 'y_norm_fsav': 'y_fsav', 'z_norm_fsav': 'z_fsav'}, inplace=True)
df_coords['hemi'] = df_coords.hemis.map({'rh': 'R', 'lh': 'L'})
annot_lr = datasets.fetch_annotation(source='margulies2016', desc='fcgradient01', den='32k', hemi=['L', 'R'])
marg_fsav = transforms.fslr_to_fsaverage(annot_lr, '10k', hemi=['L', 'R'])
l_map = np.asarray(marg_fsav[0].agg_data()).squeeze()
r_map = np.asarray(marg_fsav[1].agg_data()).squeeze()
valid_l_map = l_map != 0
valid_r_map = r_map != 0
fsavg = datasets.fetch_atlas(atlas="fsaverage", density='10k')
l_surf, r_surf = fsavg["white"]
l_verts, l_tri = nib.load(l_surf).agg_data()  # (n_vert, 3)
r_verts, r_tri = nib.load(r_surf).agg_data()
l_tree = cKDTree(l_verts[valid_l_map])
r_tree = cKDTree(r_verts[valid_r_map])
xyz = df_coords[['x_fsav', 'y_fsav', 'z_fsav']].to_numpy(float)
hemi = df_coords["hemi"].astype(str).str.upper().to_numpy()
vals = np.full(len(df_coords), np.nan, float)
idxL = np.where(hemi == "L")[0]
if len(idxL):
    _, ii = l_tree.query(xyz[idxL], k=1)
    vals[idxL] = l_map[valid_l_map][ii]
idxR = np.where(hemi == "R")[0]
if len(idxR):
    _, ii = r_tree.query(xyz[idxR], k=1)
    vals[idxR] = r_map[valid_r_map][ii]
all_subj_coords['PC1_margulies'] = vals
surface_fsav(all_subj_coords_plot_all, 'ez', subjects_dir=subjects_dir, surf='inflated', scale=5, cmap='hot')






# IMPORT THE PATHOLOGICAL CONTACTS (SOZ), PLOT THEM (FIGURE S1) and REMOVE from FURTHER ANALYSES
ez_contacts = pd.read_csv(path_base + 'per-seeg/bad_contacts.csv')
ez_contacts = ez_contacts.rename(columns={'subj_id': 'subj', 'ez_ch': 'ch_name'})
ez_contacts['ch_name'] = ez_contacts['ch_name'].apply(ast.literal_eval)
ez_contacts_exp = ez_contacts.explode('ch_name')
bad_set = set(ez_contacts_exp[['subj', 'ch_name']].itertuples(index=False, name=None))
all_subj_coords['ez'] = all_subj_coords.apply(lambda row: 1 if (row['subj'], row['ch_name']) in bad_set else row['ez'], axis=1)
all_subj_coords_plot_ez = all_subj_coords.copy()
all_subj_coords = all_subj_coords[all_subj_coords['ez'] != 1]
surface_fsav(all_subj_coords_plot_ez, 'ez', subjects_dir=subjects_dir, surf='inflated', scale=9, cmap='Reds', surf_color='white')  # FIGURE S1A
flatmap_fsav(all_subj_coords_plot_ez, 'ez', subjects_dir=subjects_dir, cmap='Reds')  # FIGURE S1B






# LOAD ORIGINAL GAMMA TIME COURSES and REMOVE OUTLIERS
gamma_path = op.join(path_base, 'gamma_analyses')
time_series_all = {}
ch_names_ts_all = {}
bad_chs_all = {}
def ensure_2d(ts):
    ts = np.array(ts)
    if ts.ndim == 1:
        return ts[np.newaxis, :]
    elif ts.ndim == 2:
        return ts
    else:
        raise ValueError(f"Expected 1D or 2D time series, got shape {ts.shape}")
for cond in ['acoustic', 'somatosensory', 'visual']:
    files = [f for f in os.listdir(gamma_path) if f.endswith('.pkl') and cond in f]
    time_series_list = []
    contact_list = []
    for filename in files:
        subj_id = filename.split('_')[0]
        filepath = os.path.join(gamma_path, filename)
        if '-right' in filename:
            continue
        if '-left' in filename:
            filepath_left = filepath
            filename_right = filename.replace('-left', '-right')
            filepath_right = os.path.join(gamma_path, filename_right)
            if not os.path.exists(filepath_right):
                continue
            with open(filepath_left, 'rb') as f:
                data_left = pickle.load(f)
            with open(filepath_right, 'rb') as f:
                data_right = pickle.load(f)
            ch_names = data_left['ch_names']
            if not ch_names:
                continue
            ts_to_use = None
            if 'gamma_ts' in data_left and 'gamma_ts' in data_right:
                ts_left = ensure_2d(data_left['gamma_ts'])
                ts_right = ensure_2d(data_right['gamma_ts'])
                ts_to_use = (ts_left + ts_right) / 2
            elif 'gamma_ts' in data_left:
                ts_to_use = ensure_2d(data_left['gamma_ts'])
            elif 'gamma_ts' in data_right:
                ts_to_use = ensure_2d(data_right['gamma_ts'])
            else:
                continue
            for ch_name, single_ts in zip(ch_names, ts_to_use):
                contact_list.append(f"{subj_id}_{ch_name}")
                time_series_list.append(single_ts)
        else:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            ch_names = data['ch_names']
            if not ch_names or 'gamma_ts' not in data:
                continue
            ts_to_use = ensure_2d(data['gamma_ts'])
            for ch_name, single_ts in zip(ch_names, ts_to_use):
                contact_list.append(f"{subj_id}_{ch_name}")
                time_series_list.append(single_ts)
    if not time_series_list:
        continue
    time_series_cond = np.stack(time_series_list, axis=0)
    ch_names_ts_cond = pd.DataFrame({
        'unique_ch_names': contact_list,
        'orig_idx': np.arange(len(contact_list))
    })
    all_subj_coords_cond = all_subj_coords.copy()
    all_subj_coords_cond.loc[:, 'unique_ch_names'] = (
        all_subj_coords_cond['subj'] + '_' + all_subj_coords_cond['ch_name']
    )
    merged = ch_names_ts_cond.merge(
        all_subj_coords_cond[['unique_ch_names', 'x_norm_fsav', 'y_norm_fsav', 'z_norm_fsav']],
        on='unique_ch_names',
        how='inner'
    )
    valid_orig_indices = merged['orig_idx'].to_numpy()
    time_series_all[cond] = time_series_cond[valid_orig_indices]
    ch_names_ts_all[cond] = merged.reset_index(drop=True)
    if cond == "acoustic":
        abs_clip = 100
    elif cond == "somatosensory":
        abs_clip = 300
    elif cond == "visual":
        abs_clip = 250
    else:
        abs_clip = None
    bad_chs_all[cond] = reject_outliers(ch_names_ts_all[cond], time_series_all[cond], abs_clip, plot=False)
bad_chs_tot = set().union(*bad_chs_all.values())
all_subj_coords = all_subj_coords[~all_subj_coords['unique_ch_names'].isin(bad_chs_tot)].reset_index(drop=True)






# PRINT DATABASE SUMMARY
TOTAL_CONTACTS_ALL = 20464  # fixed total
def _mk_uid(df, subj_col='subj', ch_col='ch_name'):
    return (df[subj_col].astype(str) + '_' + df[ch_col].astype(str))
gm_df = all_subj_coords_plot_ez[['subj', 'ch_name']].drop_duplicates().copy()
gm_df['uid'] = _mk_uid(gm_df)
uids_gm = set(gm_df['uid'].tolist())
n_gm = len(uids_gm)
pct_gm_vs_total = (n_gm / TOTAL_CONTACTS_ALL * 100.0) if TOTAL_CONTACTS_ALL > 0 else 0.0
def _infer_hemi(df):
    df = df.copy()
    if 'hemis' in all_subj_coords_plot_ez.columns:
        hemi_map = all_subj_coords_plot_ez[['subj','ch_name','hemis']].drop_duplicates()
        df = df.merge(hemi_map, on=['subj','ch_name'], how='left')
        df['hemi'] = df['hemis'].map({'lh':'sx','rh':'dx','left':'sx','right':'dx'}).fillna(df['hemis'])
        df['hemi'] = df['hemi'].replace({'LH':'sx','RH':'dx','SX':'sx','DX':'dx'})
    elif 'x_norm_surf' in all_subj_coords_plot_ez.columns:
        xyz = all_subj_coords_plot_ez[['subj','ch_name','x_norm_surf']].drop_duplicates()
        df = df.merge(xyz, on=['subj','ch_name'], how='left')
        df['hemi'] = np.where(df['x_norm_surf'] < 0, 'sx', 'dx')
    elif 'x_norm_mri' in all_subj_coords_plot_ez.columns:
        xyz = all_subj_coords_plot_ez[['subj','ch_name','x_norm_mri']].drop_duplicates()
        df = df.merge(xyz, on=['subj','ch_name'], how='left')
        df['hemi'] = np.where(df['x_norm_mri'] < 0, 'sx', 'dx')
    else:
        df['hemi'] = df['ch_name'].astype(str).map(lambda x: 'sx' if "'" in x else 'dx')
    return df
gm_df = _infer_hemi(gm_df)
n_left  = int((gm_df['hemi'] == 'sx').sum())
n_right = int((gm_df['hemi'] == 'dx').sum())
pct_left  = (n_left  / n_gm * 100.0) if n_gm > 0 else 0.0
pct_right = (n_right / n_gm * 100.0) if n_gm > 0 else 0.0
if 'ez' in all_subj_coords_plot_ez.columns:
    soz_df = all_subj_coords_plot_ez.loc[all_subj_coords_plot_ez['ez'] == 1, ['subj','ch_name']].drop_duplicates()
    uids_soz = set(_mk_uid(soz_df).tolist()) & uids_gm
else:
    uids_soz = set([f"{s}_{c}" for (s, c) in bad_set]) & uids_gm
n_soz = len(uids_soz)
pct_soz_on_gm = (n_soz / n_gm * 100.0) if n_gm > 0 else 0.0
uids_after = set(all_subj_coords['unique_ch_names'].astype(str).tolist())  # after SOZ+outliers removed
uids_outliers = (uids_gm - uids_soz) - uids_after
n_out = len(uids_outliers)
pct_out_on_gm = (n_out / n_gm * 100.0) if n_gm > 0 else 0.0
# Denominator for subsequent analyses
uids_den = uids_gm - (uids_soz | uids_outliers)
n_den = len(uids_den)
pct_den_on_gm = (n_den / n_gm * 100.0) if n_gm > 0 else 0.0
summary_df = pd.DataFrame({
    'Item': [
        'Total (all)',
        'GM (gray matter)',
        '  GM — left hemisphere',
        '  GM — right hemisphere',
        'SOZ (within GM)',
        'Outliers (within GM)',
        'Denominator (GM − SOZ − outliers)'
    ],
    'n': [
        TOTAL_CONTACTS_ALL,
        n_gm,
        n_left,
        n_right,
        n_soz,
        n_out,
        n_den
    ],
    '%': [
        '100.00',
        f"{pct_gm_vs_total:.2f}",
        f"{pct_left:.2f}",      # over GM
        f"{pct_right:.2f}",     # over GM
        f"{pct_soz_on_gm:.2f}", # over GM
        f"{pct_out_on_gm:.2f}", # over GM
        f"{pct_den_on_gm:.2f}"  # over GM
    ]
})
print('DATABASE SUMMARY')
print(summary_df.to_string(index=False))






# GAMMA RESULTS
gamma_path = op.join(path_base, 'gamma_analyses')
temporal_threshold = 20
methods = ['unc_ts', 'fdr_ts', 'bon_ts']
gamma_stats = {}
def compute_significance(ts_array, threshold):
    ts = np.array(remove_spurious_ones(ts_array, threshold))
    return int(np.sum(ts) > threshold)
for cond in ['acoustic', 'somatosensory', 'visual']:
    files = [f for f in os.listdir(gamma_path) if f.endswith('.pkl') and cond in f]
    dfs = []
    for filename in files:
        filepath = op.join(gamma_path, filename)
        subj_id = filename.split('_')[0]
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        ch_names = data.get('ch_names', [])
        if not ch_names:
            print(f"No channel names in {filename}, skipping.")
            continue
        stat_table = pd.DataFrame({'subj': subj_id, 'ch_name': ch_names})
        if 'gamma_ts' not in data:
            print(f"No gamma_ts in {filename}, skipping.")
            continue
        gamma_ts = data['gamma_ts']
        for method in methods:
            if method not in data:
                print(f"{method} not found in {filename}")
                stat_table[method.replace('_ts', '_sig')] = np.nan
                stat_table[f'{method}_auc'] = np.nan
                stat_table[f'{method}_dur'] = np.nan
                stat_table[f'{method}_lastt'] = np.nan
                continue
            ts_list = data[method]
            sig_values = []
            amp_sum, sig_durs, sig_lasts = [], [], []
            for sig_vec, ts in zip(ts_list, gamma_ts):
                sig_vec = np.array(sig_vec)
                sig_vec = remove_spurious_ones(sig_vec, temporal_threshold)
                ts = np.array(ts)
                if len(sig_vec) < len(ts):
                    sig_vec = np.pad(sig_vec, (0, len(ts) - len(sig_vec)), constant_values=0)
                elif len(sig_vec) > len(ts):
                    sig_vec = sig_vec[:len(ts)]
                sig_values.append(compute_significance(sig_vec, temporal_threshold))
                sig_idx = np.where(sig_vec == 1)[0]
                if sig_idx.size > 0:
                    amp_sum.append(np.sum(ts[sig_idx]))
                    sig_durs.append(sig_idx.size)
                    sig_lasts.append(sig_idx.max())
                else:
                    amp_sum.append(np.nan)
                    sig_durs.append(0)
                    sig_lasts.append(np.nan)
            stat_table[method.replace('_ts', '_sig')] = sig_values
            stat_table[f'{method}_auc'] = amp_sum
            stat_table[f'{method}_dur'] = sig_durs
            stat_table[f'{method}_lastt'] = sig_lasts
        dfs.append(stat_table)
    if dfs:
        df_all = pd.concat(dfs, ignore_index=True)
        merged_df = pd.merge(
            df_all,
            all_subj_coords[['subj', 'ch_name',
                             'x_norm_surf', 'y_norm_surf', 'z_norm_surf',
                             'x_norm_mri', 'y_norm_mri', 'z_norm_mri',
                             'x_norm_fsav', 'y_norm_fsav', 'z_norm_fsav', 'PC1_margulies']],
            on=['subj', 'ch_name'],
            how='left'
        )
        coord_cols = [c for c in merged_df.columns if c.startswith(('x_norm_', 'y_norm_', 'z_norm_'))]
        merged_clean = merged_df.dropna(subset=coord_cols)
        merged_clean = merged_clean.groupby(['subj', 'ch_name'], as_index=False).max()
        merged_clean = merged_clean.rename(columns={'ch_name': 'ch_names'})
        gamma_stats[cond] = merged_clean
    else:
        print(f"No valid data found for {cond}.")
label_map = {'acoustic': 'Auditory', 'somatosensory': 'Somatosensory', 'visual': 'Visual'}
gamma_stats = {k: v.assign(unique_ch_names=v['subj'].astype(str) + '_' + v['ch_names'].astype(str))for k, v in gamma_stats.items()}
motor_glasser = ["4", "4a", "4p", "6d", "6v", "6a", "6c", "6r", "6ma", "6mp", "SCEF", "24dd", "24dv", "24ad", "24pd", "FEF"] # motor areas to be excluded for
gamma_stats_nomot = {
    k: (
        gamma_stats[k]
        .merge(all_subj_coords[['unique_ch_names', 'glasser']],
               on='unique_ch_names', how='left')
        .query(' and '.join([f'glasser.str.contains("{lab}") == False'
                             for lab in motor_glasser]))
    )
    for k in gamma_stats.keys()
}
gamma_stats_nomot = {k : gamma_stats_nomot[k].merge(all_subj_coords[['unique_ch_names', 'area']], on='unique_ch_names', how='left').query('area != "precentral"') for k in gamma_stats_nomot.keys()}
[summary_df, lobe_df, A, S, V] = print_results(gamma_stats_nomot, all_subj_coords, all_subj_coords_plot_ez, label_map, correction='fdr_sig')  # PRINT GAMMA RESULTS excludig PRECENTRAL
area_table_nomot = print_stats_per_area(gamma_stats_nomot, all_subj_coords, all_subj_coords_plot_ez, label_map, name="TABLE S2", correction='fdr_sig')  # PRINT % of GRCs per AREA
[summary_df, lobe_df, A, S, V] = print_results(gamma_stats, all_subj_coords, all_subj_coords_plot_ez, label_map, correction='fdr_sig')  # PRINT GAMMA RESULTS
area_table = print_stats_per_area(gamma_stats, all_subj_coords, all_subj_coords_plot_ez, label_map, name="TABLE S2", correction='fdr_sig')  # PRINT % of GRCs per AREA
area_table.to_excel(op.join(path_results, 'Tab_S2.xlsx'))
gamma_all = df_from_stats(gamma_stats, correction='fdr_sig')






# LFP RESULTS
lfp_path = op.join(path_base, 'lfp_analyses')
temporal_threshold = 20  # number of consecutive significant samples required
methods = ['unc_ts', 'fdr_ts', 'bon_ts']
lfp_stats = {}
def compute_significance(ts_array, threshold):
    ts = np.array(ts_array)
    return int(np.sum(ts) > threshold)
for cond in ['acoustic', 'somatosensory', 'visual']:
    files = [f for f in os.listdir(lfp_path) if f.endswith('.pkl') and cond in f]
    dfs = []
    for filename in files:
        filepath = os.path.join(lfp_path, filename)
        subj_id = filename.split('_')[0]
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        ch_names = data.get('ch_names', [])
        if not ch_names:
            print(f"No channel names in {filename}, skipping.")
            continue
        stat_table = pd.DataFrame({'subj': subj_id, 'ch_name': ch_names})
        for method in methods:
            if method not in data:
                print(f"{method} not found in {filename}")
                stat_table[method.replace('_ts', '_sig')] = np.nan
                continue
            ts_list = data[method]
            sig_values = [compute_significance(ts, temporal_threshold) for ts in ts_list]
            stat_table[method.replace('_ts', '_sig')] = sig_values
        dfs.append(stat_table)
    if dfs:
        df_all = pd.concat(dfs, ignore_index=True)
        merged_df = pd.merge(
            df_all,
            all_subj_coords[['subj', 'ch_name',
                             'x_norm_surf', 'y_norm_surf', 'z_norm_surf',
                             'x_norm_mri', 'y_norm_mri', 'z_norm_mri',
                             'x_norm_fsav', 'y_norm_fsav', 'z_norm_fsav', 'PC1_margulies']],
            on=['subj', 'ch_name'],
            how='left'
        )
        coord_cols = [c for c in merged_df.columns if c.startswith(('x_norm_', 'y_norm_', 'z_norm_'))]
        merged_clean = merged_df.dropna(subset=coord_cols)
        merged_clean = merged_clean.groupby(['subj', 'ch_name'], as_index=False).max()
        merged_clean = merged_clean.rename(columns={'ch_name': 'ch_names'})
        lfp_stats[cond] = merged_clean
    else:
        print(f"No valid data found for {cond}.")
label_map = {'acoustic': 'Auditory', 'somatosensory': 'Somatosensory', 'visual': 'Visual'}
lfp_stats = {k: v.assign(unique_ch_names=v['subj'].astype(str) + '_' + v['ch_names'].astype(str))for k, v in lfp_stats.items()}
motor_glasser = [
    "4", "4a", "4p",
    "6d", "6v", "6a", "6c", "6r",
    "6ma", "6mp", "SCEF",
    "24dd", "24dv", "24ad", "24pd",
    "FEF",
]
lfp_stats_nomot = {
    k: (
        lfp_stats[k]
        .merge(all_subj_coords[['unique_ch_names', 'glasser']],
               on='unique_ch_names', how='left')
        .query(' and '.join([f'glasser.str.contains("{lab}") == False'
                             for lab in motor_glasser]))
    )
    for k in lfp_stats.keys()
}
lfp_stats_nomot = {k : lfp_stats_nomot[k].merge(all_subj_coords[['unique_ch_names', 'area']], on='unique_ch_names', how='left').query('area != "precentral"') for k in lfp_stats_nomot.keys()}
[summary_df, lobe_df, A, S, V] = print_results(lfp_stats_nomot, all_subj_coords, all_subj_coords_plot_ez, label_map, correction='fdr_sig')  # PRINT GAMMA RESULTS excludig PRECENTRAL
[summary_df, lobe_df, A, S, V] = print_results(lfp_stats, all_subj_coords, all_subj_coords_plot_ez, label_map, correction='fdr_sig')
area_table = print_stats_per_area(lfp_stats, all_subj_coords, all_subj_coords_plot_ez, label_map, name="TABLE S3", correction='fdr_sig')
area_table.to_excel(op.join(path_results, 'Tab_S3.xlsx'))
lfp_all = df_from_stats(lfp_stats, correction='fdr_sig')





# SHOW THREE EXAMPLES (FIG.1b)
conds = ['visual', 'visual', 'visual']
subjs = ['sub-63', 'sub-63', 'sub-63']
chs = ['O_06', 'Y_11', 'D_08']
cmaps = ['Greys', 'Greys', 'Greys']
for cond, subj, ch, cmap in zip(conds, subjs, chs, cmaps):
    path_original_data = op.join(path_base, 'per-seeg')
    subj_cond_fname = subj + '_task-' + cond + 'task_run-01'
    imported_data = import_data(op.join(path_original_data, subj + '/seeg/'), subj_cond_fname)
    run_gamma_and_lfp_examples(imported_data, ch, cmap, vlim=(0, 4), correction='fdr_by', show_lfp=True)






# PLOT ALL CONTACTS within PRINCIPAL GRADIENT 1 (FIG.1c, top)
fsavg = datasets.fetch_atlas(atlas="fsaverage", density='41k')
l_white_surf, r_white_surf = fsavg["white"]
l_white_verts, l_white_tri = nib.load(l_white_surf).agg_data()
r_white_verts, r_white_tri = nib.load(r_white_surf).agg_data()
l_infl_surf, r_infl_surf = fsavg["inflated"]
l_infl_verts, l_infl_tri = nib.load(l_infl_surf).agg_data()
r_infl_verts, r_infl_tri = nib.load(r_infl_surf).agg_data()
annot_lr = datasets.fetch_annotation(
    source='margulies2016', desc='fcgradient01', den='32k', hemi=['L', 'R']
)
marg_fsav = transforms.fslr_to_fsaverage(annot_lr, '41k', hemi=['L', 'R'])
l_map = np.asarray(marg_fsav[0].agg_data()).squeeze()
r_map = np.asarray(marg_fsav[1].agg_data()).squeeze()
valid_l_map = l_map != 0
valid_r_map = r_map != 0
df_plot = all_subj_coords.copy()
cmap_obj = mcolors.ListedColormap(['black', 'white'])
norm = plt.Normalize(vmin=0, vmax=1)
rgba = cmap_obj(norm(df_plot['ez'].values))
df_plot[['r', 'g', 'b', 'a']] = rgba
infl_surfs = {}
for h, verts, tri in zip(
    ['lh', 'rh'],
    [l_infl_verts, r_infl_verts],
    [l_infl_tri, r_infl_tri]
):
    faces = np.hstack([np.full((len(tri), 1), 3), tri]).astype(np.int64)
    infl_surfs[h] = pv.PolyData(verts, faces)
l_tree = cKDTree(l_white_verts[valid_l_map])
r_tree = cKDTree(r_white_verts[valid_r_map])
plotter = pv.Plotter(shape=(1, 2), notebook=False)
plotter.set_background('white')
light = pv.Light(position=(1, 1, 1), color='white', intensity=0.01)
plotter.add_light(light)
plotter.subplot(0, 0)
plotter.add_mesh(
    infl_surfs['lh'],
    opacity=1,
    scalars=l_map,
    cmap='Spectral_r',
    clim=(-7, 7)
)
h_coords = df_plot.query('hemis=="lh"').copy()
xyz = h_coords[['x_norm_fsav', 'y_norm_fsav', 'z_norm_fsav']].values
_, idx = l_tree.query(xyz)
infl_points = l_infl_verts[valid_l_map][idx]
plotter.add_points(
    infl_points,
    scalars=h_coords[['r', 'g', 'b', 'a']].values,
    rgb=True,
    render_points_as_spheres=True,
    point_size=7
)
plotter.subplot(0, 1)
plotter.add_mesh(
    infl_surfs['rh'],
    opacity=1,
    scalars=r_map,
    cmap='Spectral_r',
    clim=(-7, 7)
)
h_coords = df_plot.query('hemis=="rh"').copy()
xyz = h_coords[['x_norm_fsav', 'y_norm_fsav', 'z_norm_fsav']].values
_, idx = r_tree.query(xyz)
infl_points = r_infl_verts[valid_r_map][idx]
plotter.add_points(
    infl_points,
    scalars=h_coords[['r', 'g', 'b', 'a']].values,
    rgb=True,
    render_points_as_spheres=True,
    point_size=7
)
plotter.show()






# PLOT RESPONSIVNESS of GRCs, LOCs and UCs along PRINCIPAL GRADIENT 1 (FIG.1c, bottom)
gamma = gamma_all.copy()
lfp = lfp_all.copy()
merge_cols = ['unique_ch_names', 'subj', 'PC1_margulies']
gamma = gamma[merge_cols + ['acoustic', 'somatosensory', 'visual']].rename(columns={
    'acoustic': 'gamma_acoustic',
    'somatosensory': 'gamma_somatosensory',
    'visual': 'gamma_visual'
})
lfp = lfp[merge_cols + ['acoustic', 'somatosensory', 'visual']].rename(columns={
    'acoustic': 'lfp_acoustic',
    'somatosensory': 'lfp_somatosensory',
    'visual': 'lfp_visual'
})
df = pd.merge(
    gamma,
    lfp,
    on=['unique_ch_names', 'subj', 'PC1_margulies'],
    how='inner'
)
df['gamma_resp'] = (
    (df['gamma_acoustic'] == 1) |
    (df['gamma_somatosensory'] == 1) |
    (df['gamma_visual'] == 1)
)
df['lfp_resp'] = (
    (df['lfp_acoustic'] == 1) |
    (df['lfp_somatosensory'] == 1) |
    (df['lfp_visual'] == 1)
)
df['resp_class'] = np.select(
    [
        df['gamma_resp'],
        (~df['gamma_resp']) & (df['lfp_resp']),
        (~df['gamma_resp']) & (~df['lfp_resp'])
    ],
    [
        'Gamma responsive',
        'LFP only',
        'Unresponsive'
    ],
    default=np.nan
)
n_bins = 9
bin_edges = np.linspace(-6, 6, n_bins + 1)
df['PC1_margulies_clipped'] = df['PC1_margulies'].clip(-6, 6)
df['PC1_bin'] = pd.cut(
    df['PC1_margulies_clipped'],
    bins=bin_edges,
    include_lowest=True
)
all_bins = df['PC1_bin'].cat.categories
all_classes = ['Gamma responsive', 'LFP only', 'Unresponsive']
subj_counts = (
    df.groupby(['subj', 'PC1_bin', 'resp_class'], observed=False)
      .size()
      .rename('n')
      .reset_index()
)
subj_totals = (
    df.groupby(['subj', 'PC1_bin'], observed=False)
      .size()
      .rename('n_total')
      .reset_index()
)
subj_df = subj_counts.merge(
    subj_totals,
    on=['subj', 'PC1_bin'],
    how='left'
)
subj_df['percent'] = subj_df['n'] / subj_df['n_total'] * 100
all_subj = sorted(df['subj'].dropna().unique())
full_index = pd.MultiIndex.from_product(
    [all_subj, all_bins, all_classes],
    names=['subj', 'PC1_bin', 'resp_class']
)
subj_df = (
    subj_df.set_index(['subj', 'PC1_bin', 'resp_class'])
           .reindex(full_index, fill_value=0)
           .reset_index()
)
subj_df = subj_df.drop(columns=['n_total', 'percent'], errors='ignore')
subj_df = subj_df.merge(
    subj_totals,
    on=['subj', 'PC1_bin'],
    how='left'
)
subj_df['percent'] = np.where(
    subj_df['n_total'].notna() & (subj_df['n_total'] > 0),
    subj_df['n'] / subj_df['n_total'] * 100,
    np.nan
)
plot_df = (
    subj_df.groupby(['PC1_bin', 'resp_class'], observed=False)
           .agg(
               mean_percent=('percent', 'mean'),
               sem=('percent', lambda x: x.std(ddof=1) / np.sqrt(x.notna().sum()))
           )
           .reset_index()
)
plot_df['x_plot'] = plot_df['PC1_bin'].apply(lambda x: x.mid)
colors = {
    'Gamma responsive': '#f28e2b',   # arancio
    'LFP only': '#7b61ff',           # viola
    'Unresponsive': '#9e9e9e'        # grigio
}
plt.figure(figsize=(8, 5))
for cls, d in plot_df.groupby('resp_class'):
    d = d.sort_values('x_plot')
    plt.errorbar(
        d['x_plot'],
        d['mean_percent'],
        yerr=d['sem'],
        marker='o',
        linestyle='-',
        linewidth=2,
        markersize=6,
        capsize=3,
        color=colors[cls],
        label=cls
    )
plt.xlabel('PC1 Margulies')
plt.ylabel('Contacts in bin (%)')
plt.title('Responsiveness across cortical gradient')
plt.xlim(-6, 6)
plt.xticks(np.linspace(-6, 6, 7))
plt.legend(frameon=False)
plt.tight_layout()
plt.show()






# CREATE GAMMA CONTACTS FLATMAPS and SURFACES per EACH MODALITY and COMBINED (FIGURES S2-S3-S4-S8)
conds = ["acoustic", "somatosensory", "visual"]
cmaps = ['Greens', 'Reds', 'Blues']
green = mcolors.LinearSegmentedColormap.from_list("bg", ["black", "green"])
red = mcolors.LinearSegmentedColormap.from_list("br", ["black", "red"])
blue = mcolors.LinearSegmentedColormap.from_list("bb", ["black", "blue"])
cmaps_b = [green, red, blue]
for cond, cmap, cmap_b in zip(conds, cmaps, cmaps_b):
    surface_fsav(gamma_all, cond, subjects_dir=subjects_dir, cmap=cmap_b, surf='inflated', scale=10, surf_color='white')
    flatmap_fsav(gamma_all, cond, subjects_dir=subjects_dir, cmap=cmap_b)
colors = ['black', 'green', 'red', 'yellow', 'blue', 'cyan', 'magenta', 'white']
cmap_all = mcolors.ListedColormap(colors)
cmap_all.set_bad(alpha=0)
coords_gamma = gamma_all
coords_gamma["multi_plot"] = gamma_all["multi"].astype(float)
surface_fsav(gamma_all, 'multi_plot', subjects_dir=subjects_dir, cmap=cmap_all, surf='inflated', scale=10, surf_color='white')
flatmap_fsav(gamma_all, 'multi_plot', subjects_dir=subjects_dir, cmap=cmap_all)






# CREATE GAMMA AUC and OFFSET MAPs per EACH MODALITY (FIGURES S5-S6)
conds = ["acoustic", "somatosensory", "visual"]
cmaps = ['Greens', 'Reds', 'Blues']
for cond, cmap in zip(conds, cmaps):
    gamma_to_plot = gamma_stats[cond].copy()
    gamma_to_plot['fdr_sig'] = gamma_to_plot['fdr_sig'].fillna(0).astype(int)
    continuous_maps_one_cond(gamma_to_plot, 'fdr_sig', cmap, subjects_dir, lims=[0, 0.5, 1], sm=10, transparent=False, distance=0.015)
    pointplot_gamma_by_area(all_subj_coords, gamma_to_plot, 'fdr_sig')
for cond, cmap in zip(conds, cmaps):
    gamma_to_plot = gamma_stats[cond].copy()
    gamma_to_plot['fdr_ts_auc_log'] = np.log10(np.abs(gamma_to_plot['fdr_ts_auc'].fillna(0)) + 1)
    gamma_to_plot['fdr_ts_auc'] = abs(gamma_to_plot['fdr_ts_auc'].fillna(0).astype(int))
    continuous_maps_one_cond(gamma_to_plot, 'fdr_ts_auc_log', cmap, subjects_dir, lims=[0, 1.5, 3], sm=10, transparent=False, distance=0.015)
    pointplot_gamma_by_area(all_subj_coords, gamma_to_plot, 'fdr_ts_auc_log')
for cond, cmap in zip(conds, cmaps):
    gamma_to_plot = gamma_stats[cond].copy()
    gamma_to_plot['fdr_ts_lastt'] = ((gamma_to_plot['fdr_ts_lastt']-200).fillna(0).astype(int))
    continuous_maps_one_cond(gamma_to_plot, 'fdr_ts_lastt', cmap, subjects_dir, lims=[0, 100, 200], sm=10, transparent=False, distance=0.015)
    pointplot_gamma_by_area(all_subj_coords, gamma_to_plot, 'fdr_ts_lastt')






# HABITUATION ANALYSES (Fig.S7)
conds = ['acoustictask_run-01', 'somatosensorytask_run-01', 'visualtask_run-01']
conds_bil = ['acoustictask-left_run-01', 'acoustictask-right_run-01',  'somatosensorytask-left_run-01', 'somatosensorytask-right_run-01', 'visualtask-bilat_run-01']
subjs_bil = ['sub-04', 'sub-06', 'sub-12', 'sub-14', 'sub-19', 'sub-22', 'sub-24', 'sub-27', 'sub-32', 'sub-33', 'sub-37', 'sub-40', 'sub-44', 'sub-46', 'sub-47', 'sub-59']
for subj in subj_list:
    if subj in subjs_bil:
        continue
    else:
        conds_sess = conds
    for cond in conds_sess:
        subj_cond_fname = subj + '_task-' + cond
        path_original_data = op.join(path_base, 'per-seeg')
        imported_data = import_data(op.join(path_original_data, subj + '/seeg/'), subj_cond_fname)
        stat_path = op.join(path_base, 'gamma_analyses', subj_cond_fname.replace('task-', '') + '_gamma.pkl')
        with open(stat_path, "rb") as f:
            stat_data = pickle.load(f)
        path_save_gamma_hab = op.join(path_base, 'gamma_habituation')
        os.makedirs(path_save_gamma_hab, exist_ok=True)
        run_gamma_habituation(imported_data, stat_data, subj, cond, path_save_gamma_hab)
path_gamma_hab = os.path.join(path_base, 'gamma_habituation')
def build_matrix(data_dict, subj, modality):
    ch_names = data_dict["ch_names"]
    values = data_dict["gamma_hab"]
    mat = np.vstack(values)
    unique_ch_names = [f"{subj}_{ch}" for ch in ch_names]
    df = pd.DataFrame(mat, index=unique_ch_names)
    df.index.name = "unique_ch_name"
    df["subj"] = subj
    df["modality"] = modality
    return df.reset_index()
def load_all_from_folder(folder):
    dfs = []
    for path in glob.glob(os.path.join(folder, "*.pkl")):
        fname = os.path.basename(path)
        subj = fname.split("_")[0]
        if "acoustic" in fname:
            modality = "acoustic"
        elif "somatosensory" in fname:
            modality = "somatosensory"
        elif "visual" in fname:
            modality = "visual"
        else:
            modality = "unknown"
        with open(path, "rb") as f:
            data_dict = pickle.load(f)
        dfs.append(build_matrix(data_dict, subj, modality))
    return pd.concat(dfs, ignore_index=True)
hab_df = load_all_from_folder(path_gamma_hab)
colors = {"acoustic": "green", "somatosensory": "red", "visual": "blue"}
modalities = ["acoustic", "somatosensory", "visual"]
def remove_outliers_nan(arr, modality):
    arr = arr.copy()
    for i in range(arr.shape[1]):
        col = arr[:, i]
        valid = ~np.isnan(col)
        x = col[valid]
        if len(x) == 0:
            continue
        lo = np.nanquantile(x, 0.05)
        hi = np.nanquantile(x, 0.95)
        x[(x < lo) | (x > hi)] = np.nan
        col[valid] = x
        arr[:, i] = col
    return arr
def mean_sem_ignore_nan(arr):
    mean = np.nanmean(arr, axis=0)
    sem = np.nanstd(arr, axis=0) / np.sqrt(np.sum(~np.isnan(arr), axis=0))
    return mean, sem
def sliding_mean(arr, window=20, step=1):
    n_cols = arr.shape[1]
    n_bins = (n_cols - window) // step + 1
    arr_binned = np.zeros((arr.shape[0], n_bins))
    for b in range(n_bins):
        start = b * step
        stop = start + window
        arr_binned[:, b] = np.nanmean(arr[:, start:stop], axis=1)
    return arr_binned
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True, sharey='all')
for i, modality in enumerate(modalities):
    ax = axes[i]
    df_mod = hab_df[hab_df["modality"] == modality].set_index("unique_ch_name")
    arr = df_mod.iloc[:, :100].to_numpy(dtype=float)
    arr = remove_outliers_nan(arr, modality)
    arr = np.log10(arr + 0.01)
    arr = sliding_mean(arr, window=20, step=1)
    mean_across_ch = np.nanmean(arr, axis=0)
    first_block_mean = mean_across_ch[0]
    arr = ((arr - first_block_mean) / first_block_mean) * 100
    mean, sem = mean_sem_ignore_nan(arr)
    x = np.arange(1, arr.shape[1] + 1)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.errorbar(x, mean, yerr=sem, fmt='o', markersize=4, capsize=3,
                color=colors[modality], ecolor=colors[modality], elinewidth=1.2)
    ax.set_title(modality.capitalize())
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Trials (20 trials sliding win)")
    ax.set_ylabel("Δ% from mean of 1st window")
    ax.set_ylim(-5, 5)
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.show()






# CREATE GAMMA SURFACES and FLATMAPS for SPATIO-TEMPORAL BOUNDARIES per EACH MODALITY (Fig.2a)
subject = 'fsaverage'
conds = ["acoustic", "somatosensory", "visual"]
cmaps = ['Greens', 'Reds', 'Blues']
vmin = 20
vmaxs = [150, 150, 300]
green = mcolors.LinearSegmentedColormap.from_list("bg", ["darkgreen", "green", 'forestgreen', 'limegreen', 'lime'])
red = mcolors.LinearSegmentedColormap.from_list("br", ["darkred", "firebrick", 'red', 'orangered', 'orange'])
blue = mcolors.LinearSegmentedColormap.from_list("bb", ["navy", "mediumblue", 'royalblue', 'deepskyblue', 'cyan'])
cmaps_b = [green, red, blue]
for cond, cmap, cmap_b, vmax in zip(conds, cmaps, cmaps_b, vmaxs):
    gamma_to_plot = gamma_stats[cond]
    gamma_to_plot['fdr_sig'] = gamma_to_plot['fdr_sig'].astype(int)
    stc_resp, brain_resp = continuous_maps_one_cond(gamma_to_plot, 'fdr_sig', cmap, subjects_dir, lims=[0, 0.2, 1], sm=10, transparent=False, distance=0.015)
    gamma_to_plot = gamma_stats[cond].copy()
    gamma_to_plot['fdr_ts_lastt'] = ((gamma_to_plot['fdr_ts_lastt']-200).fillna(0).astype(int))
    stc_time, brain_time = continuous_maps_one_cond(gamma_to_plot, 'fdr_ts_lastt', cmap, subjects_dir, lims=[0, 100, 200], sm=10, transparent=False, distance=0.015)
    vdat_time = np.concatenate([brain_time._data['lh']['array'], brain_time._data['rh']['array']])
    vdat_up_time = cortex.freesurfer.upsample_to_fsaverage(vdat_time.squeeze(), "fsaverage5", freesurfer_subjects_dir=subjects_dir)
    vdat_resp = np.concatenate([brain_resp._data['lh']['array'], brain_resp._data['rh']['array']])
    vdat_up_resp = cortex.freesurfer.upsample_to_fsaverage(vdat_resp.squeeze(), "fsaverage5", freesurfer_subjects_dir=subjects_dir)
    vertex_data = cortex.Vertex(vdat_up_time, subject, vmin=vmin, vmax=vmax, cmap=cmap_b)
    alpha_low, alpha_high = 0.0, 0.3  # data range that maps to alpha 0..1
    resp = vdat_up_resp.squeeze()
    alpha = (resp - alpha_low) / (alpha_high - alpha_low)
    alpha = np.clip(alpha, 0.0, 1.0)  # keep alpha in [0, 1]
    vertex_data_blend = vertex_data.blend_curvature(alpha=alpha)
    cortex.webgl.show(data=vertex_data_blend, colorbar=True)







# PLOT VIOLINS for OFFSET distributions (FIGURE 2a, insert)
conds = ["acoustic", "somatosensory", "visual"]
cmaps = ['Greens', 'Reds', 'Blues']
for cond in conds:
    gamma_to_plot = gamma_stats[cond].copy()
    to_plot = (gamma_to_plot['fdr_ts_lastt'] - 200).fillna(0).astype(int)
    to_plot[to_plot == 0] = np.nan
    to_plot = np.asarray(to_plot, float)
    to_plot = to_plot[~np.isnan(to_plot)]
    p10, p50, p90 = np.percentile(to_plot, [10, 50, 90]) #clip to 10° and 90° percentile
    to_plot = to_plot[(to_plot >= p10) & (to_plot <= p90)]
    fig, ax = plt.subplots(figsize=(4, 2.5))
    parts = ax.violinplot(
        to_plot,
        vert=False,
        showmeans=False,
        showmedians=False,
        showextrema=False
    )
    for pc in parts['bodies']:
        verts = pc.get_paths()[0].vertices
        center = np.mean(verts[:, 1])
        verts[:, 1] = np.clip(verts[:, 1], center, np.inf)  # metà superiore
        pc.set_alpha(0.8)
    ax.plot([p10, p90], [1, 1], lw=2, color="k")
    ax.plot(p50, 1, "o", color="k")
    ax.set_yticks([])
    ax.set_xlabel("Gamma offset (ms)")
    ax.set_title(cond)
    ax.set_ylim(0.5, 1.5)
    ax.set_yticks([1])
    ax.set_yticklabels([cond])
    plt.tight_layout()
    plt.show()
    valid = to_plot[~np.isnan(to_plot)]
    mean_val = np.mean(valid)
    std_val = np.std(valid, ddof=1)
    print(f"{cond}: N={len(valid)}, Mean={mean_val:.2f} ms, SEM={std_val:.2f}")






# CREATE GAMMA SURFACES and FLATMAPS for SPATIO-TEMPORAL BOUNDARIES across ALL MODALITIES (Fig.2b)
gamma_to_plot = gamma_stats['acoustic']
gamma_to_plot['fdr_sig'] = gamma_to_plot['fdr_sig'].astype(int)
stc_ac, brain_ac = continuous_maps_one_cond(gamma_to_plot, 'fdr_sig', 'Greys', subjects_dir, lims=[0, 0.2, 1], sm=10, transparent=False, distance=0.015)
vdat_ac = np.concatenate([brain_ac._data['lh']['array'], brain_ac._data['rh']['array']])
vdat_up_ac = cortex.freesurfer.upsample_to_fsaverage(vdat_ac.squeeze(), "fsaverage5", freesurfer_subjects_dir=subjects_dir)
gamma_to_plot = gamma_stats['somatosensory']
gamma_to_plot['fdr_sig'] = gamma_to_plot['fdr_sig'].astype(int)
stc_ss, brain_ss = continuous_maps_one_cond(gamma_to_plot, 'fdr_sig', 'Greys', subjects_dir, lims=[0, 0.2, 1], sm=10, transparent=False, distance=0.015)
vdat_ss = np.concatenate([brain_ss._data['lh']['array'], brain_ss._data['rh']['array']])
vdat_up_ss = cortex.freesurfer.upsample_to_fsaverage(vdat_ss.squeeze(), "fsaverage5", freesurfer_subjects_dir=subjects_dir)
gamma_to_plot = gamma_stats['visual']
gamma_to_plot['fdr_sig'] = gamma_to_plot['fdr_sig'].astype(int)
stc_vi, brain_vi = continuous_maps_one_cond(gamma_to_plot, 'fdr_sig', 'Greys', subjects_dir, lims=[0, 0.2, 1], sm=10, transparent=False, distance=0.015)
vdat_vi = np.concatenate([brain_vi._data['lh']['array'], brain_vi._data['rh']['array']])
vdat_up_vi = cortex.freesurfer.upsample_to_fsaverage(vdat_vi.squeeze(), "fsaverage5", freesurfer_subjects_dir=subjects_dir)
vdat_up_max = np.max(np.stack([vdat_up_ac + vdat_up_ss + vdat_up_vi]), axis=0)
colors = ['green', 'red', 'yellow', 'blue', 'cyan', 'magenta', 'white']
cmap = mcolors.ListedColormap(colors)
cmap.set_bad(alpha=0)
coords_gamma = gamma_all
coords_gamma["multi_plot"] = gamma_all["multi"].astype(float)
coords_gamma.loc[coords_gamma["multi_plot"] == 0, "multi_plot"] = np.nan
cmap_multi = mcolors.ListedColormap([
    'white',   # 0 = none
    'green',   # 1 = acoustic
    'red',     # 2 = somatosensory
    'yellow',    # 4 = visual
    'blue',  # 3 = A+S
    'cyan',    # 5 = A+V
    'magenta', # 6 = S+V
    'dimgray'    # 7 = All
])
stc_all, brain_all = continuous_maps_multi(gamma_all, cmap_multi, fs_dir=subjects_dir, th=0.2)
vdat_all = np.concatenate([brain_all._data['lh']['array'], brain_all._data['rh']['array']])
vdat_up_all = cortex.freesurfer.upsample_to_fsaverage(vdat_all.squeeze(), "fsaverage5", freesurfer_subjects_dir=subjects_dir)
vertex_data = cortex.Vertex(vdat_up_all, subject, vmin=0, vmax=7, cmap=cmap_multi)
alpha_low, alpha_high = 0.0, 1.0  # data range that maps to alpha 0..1
resp = vdat_up_max.squeeze()
alpha = (resp - alpha_low) / (alpha_high - alpha_low)
alpha = np.clip(alpha, 0.0, 1.0)  # keep alpha in [0, 1]
vertex_data_blend = vertex_data.blend_curvature(alpha=alpha)
cortex.webgl.show(data=vertex_data_blend)
[summary_df, lobe_df, A, S, V] = print_results(gamma_stats, all_subj_coords, all_subj_coords_plot_ez, label_map, correction='fdr_sig', printt=False)
[_, tot_gamma] = print_crossmodal_per_area(A, S, V, all_subj_coords, name='TABLE S4')  # TABLE S4






# CREATE LFP CONTACTS FLATMAPS and SURFACES per EACH MODALITY and COMBINED (FIGURES S9-S10-S11-S13)
conds = ["acoustic", "somatosensory", "visual"]
cmaps = ['Greens', 'Reds', 'Blues']
green = mcolors.LinearSegmentedColormap.from_list("bg", ["black", "green"])
red = mcolors.LinearSegmentedColormap.from_list("br", ["black", "red"])
blue = mcolors.LinearSegmentedColormap.from_list("bb", ["black", "blue"])
cmaps_b = [green, red, blue]
for cond, cmap, cmap_b in zip(conds, cmaps, cmaps_b):
    surface_fsav(lfp_all, cond, subjects_dir=subjects_dir, cmap=cmap_b, surf='inflated', scale=10, surf_color='white')
    flatmap_fsav(lfp_all, cond, subjects_dir=subjects_dir, cmap=cmap_b)
colors = ['black', 'green', 'red', 'yellow', 'blue', 'cyan', 'magenta', 'white']
cmap_all = mcolors.ListedColormap(colors)
cmap_all.set_bad(alpha=0)
coords_lfp = lfp_all
coords_lfp["multi_plot"] = lfp_all["multi"].astype(float)
surface_fsav(lfp_all, 'multi_plot', subjects_dir=subjects_dir, cmap=cmap_all, surf='inflated', scale=10, surf_color='white')
flatmap_fsav(lfp_all, 'multi_plot', subjects_dir=subjects_dir, cmap=cmap_all)






# CREATE GAMMA vs LFP MAP per EACH CONDITION (FIGURE 3A-S12)
conds = ['acoustic', 'somatosensory', 'visual']
subject='fsaverage'
for cond in conds:
    gamma_vs_lfp = gamma_all[['unique_ch_names', cond]].rename(columns={cond: 'gamma'})
    gamma_vs_lfp = gamma_vs_lfp.merge(lfp_all[['unique_ch_names', cond, 'x_norm_fsav', 'y_norm_fsav', 'z_norm_fsav']], on='unique_ch_names', how='inner')
    gamma_vs_lfp = gamma_vs_lfp.rename(columns={cond: 'lfp'})
    colors = ['white', 'purple', 'orange']
    cmap = mcolors.ListedColormap(colors)
    stc_gamma, brain_gamma = continuous_maps_one_cond(gamma_all, cond, cmap, subjects_dir, lims=[0, 0.2, 1], sm=10, transparent=False, distance=0.015)
    vdat_gamma = np.concatenate([brain_gamma._data['lh']['array'], brain_gamma._data['rh']['array']])
    vdat_up_gamma = cortex.freesurfer.upsample_to_fsaverage(vdat_gamma.squeeze(), "fsaverage5", freesurfer_subjects_dir=subjects_dir)
    stc_lfp, brain_lfp = continuous_maps_one_cond(lfp_all, cond, cmap, subjects_dir, lims=[0, 0.2, 1], sm=10, transparent=False, distance=0.015)
    vdat_lfp = np.concatenate([brain_lfp._data['lh']['array'], brain_lfp._data['rh']['array']])
    vdat_up_lfp = cortex.freesurfer.upsample_to_fsaverage(vdat_lfp.squeeze(), "fsaverage5", freesurfer_subjects_dir=subjects_dir)
    vdat_up_max = np.max(np.stack([vdat_up_gamma + vdat_up_lfp]), axis=0)
    stc_gamma_vs_lfp, brain_gamma_vs_lfp = continuous_maps_gamma_vs_lfp(gamma_vs_lfp, cmap, subjects_dir)
    vdat_gamma_vs_lfp = np.concatenate([brain_gamma_vs_lfp._data['lh']['array'], brain_gamma_vs_lfp._data['rh']['array']])
    vdat_up_gamma_vs_lfp = cortex.freesurfer.upsample_to_fsaverage(vdat_gamma_vs_lfp.squeeze(), "fsaverage5", freesurfer_subjects_dir=subjects_dir)
    vertex_data = cortex.Vertex(vdat_up_gamma_vs_lfp, subject, vmin=0, vmax=3, cmap=cmap)
    alpha_low, alpha_high = 0.0, 1.0  # data range that maps to alpha 0..1
    resp = vdat_up_max.squeeze()
    alpha = (resp - alpha_low) / (alpha_high - alpha_low)
    alpha = np.clip(alpha, 0.0, 1.0)  # keep alpha in [0, 1]
    vertex_data_blend = vertex_data.blend_curvature(alpha=alpha)
    cortex.webgl.show(data=vertex_data_blend)






# CONNECTIVITY ANALYSIS BASED on CCEPs
# original data for this analysis are not provided, you can skip to the next part of the script using directly the connectivity matices, available with the original data
#
# path_ccep = op.join(path_base, 'ccep_imported')
# path_conn_save = op.join(path_base, 'connectivity')
# merged = gamma_all.merge(lfp_all, on=['unique_ch_names', 'subj', 'ch_name'], suffixes=('_gamma', '_lfp'))
# gamma_condition = ((merged['acoustic_gamma'] == 1) | (merged['somatosensory_gamma'] == 1) | (merged['visual_gamma'] == 1))
# lfp_condition = ((merged['acoustic_gamma'] == 0) & (merged['somatosensory_gamma'] == 0) & (merged['visual_gamma'] == 0) & ( (merged['acoustic_lfp'] == 1) | (merged['somatosensory_lfp'] == 1) | (merged['visual_lfp'] == 1)))
# merged['gamma_lfp'] = 'none'
# merged.loc[gamma_condition, 'gamma_lfp'] = 'gamma'
# merged.loc[lfp_condition, 'gamma_lfp'] = 'lfp'
# df_gamma_lfp = merged[['unique_ch_names', 'subj', 'ch_name', 'gamma_lfp']]
# df_gamma_lfp['ccep_exist'] = 'no'
# df_gamma_lfp['file_conn'] = None
# counts = df_gamma_lfp['gamma_lfp'].value_counts()
# print(counts)
# for sub, ch in tqdm(zip(df_gamma_lfp['subj'], df_gamma_lfp['ch_name']), total=len(df_gamma_lfp), desc='Processing'):
#     path_subj = op.join(path_ccep, sub)
#     if os.path.isdir(path_subj):
#         sess = pd.DataFrame(os.listdir(path_subj), columns=['sess_bip'])
#         sess['sess_cont'] = sess['sess_bip'].str.split('-').str[0].str.replace(
#             r"^([A-Z]'+?|\w)(\d{1})$", r"\1_0\2", regex=True).str.replace(
#             r"^([A-Z]'+?|\w)(\d{2,})$", r"\1_\2", regex=True)
#         if ch in sess['sess_cont'].values:
#             df_gamma_lfp.loc[(df_gamma_lfp['subj'] == sub) & (df_gamma_lfp['ch_name'] == ch), 'ccep_exist'] = 'yes'
#             stim_bip = sess.loc[sess['sess_cont'] == ch, 'sess_bip'].values[0]
#             file_conn_name = f"{sub}_{stim_bip}_conn_ccep.pkl"
#             df_gamma_lfp.loc[(df_gamma_lfp['subj'] == sub) & (df_gamma_lfp['ch_name'] == ch), 'file_conn'] = file_conn_name
#             df_sub = df_gamma_lfp.loc[df_gamma_lfp['subj'] == sub, ['subj', 'ch_name', 'gamma_lfp']]
#             path_sess = op.join(path_subj, stim_bip)
#             resp_gamma_lfp_ch = df_gamma_lfp.loc[(df_gamma_lfp['subj'] == sub) & (df_gamma_lfp['ch_name'] == ch), 'gamma_lfp'].values[0]
#             compute_ccep_connectivity(path_sess, df_sub, sub, stim_bip, path_conn_save, resp_gamma_lfp_ch)
# df_gamma_stim = df_gamma_lfp[(df_gamma_lfp['ccep_exist'] == 'yes') & (df_gamma_lfp['gamma_lfp'] == 'gamma')].copy()
# ---------------------------------------------------------------------------------------------------------------------




# CREATE DF for CCEP RESULTS
df_gamma_stim = pd.read_csv(op.join(path_results, 'ccep_gamma_stim.csv')) # run this to skip the previous code
path = op.join(path_base, 'connectivity')
results = []
for subj, df_sub in df_gamma_stim.groupby("subj"):
    ch_names_ref = None
    conn_list = []
    resp_list = []
    for fname in df_sub["file_conn"]:
        file_path = os.path.join(path, fname)
        if os.path.exists(file_path):
            with open(file_path, "rb") as fh:
                data = pickle.load(fh)["conn_ccep_val"]
            ch_names = list(data["ch_name"])
            conn = np.array(data["conn"])
            resp = np.array(data["n1_auc"])
            if ch_names_ref is None:
                ch_names_ref = ch_names
            else:
                if ch_names != ch_names_ref:
                    reorder_idx = [ch_names.index(ch) for ch in ch_names_ref]
                    conn = conn[reorder_idx]
                    resp = resp[reorder_idx]
        else:
            if ch_names_ref is None:
                continue
            conn = np.full(len(ch_names_ref), np.nan)
            resp = np.full(len(ch_names_ref), np.nan)
        conn_list.append(conn)
        resp_list.append(resp)
    if conn_list and resp_list:
        conn_matrix = np.column_stack(conn_list)
        resp_matrix = np.column_stack(resp_list)
        conn_sum = np.nansum(conn_matrix, axis=1)
        resp_mean = np.nanmean(resp_matrix, axis=1)
        df_out = pd.DataFrame({
            "subj": subj,
            "ch_name": ch_names_ref,
            "conn_sum": conn_sum,
            "resp_mean": resp_mean
        })
        results.append(df_out)
df_results_gamma_stim = pd.concat(results, ignore_index=True)
df_results_gamma_stim['ch_name'] = df_results_gamma_stim['ch_name'].str.replace('_', '', regex=False)
cond_map = {"acoustic": "audio", "somatosensory": "somato", "visual": "video"}
gamma_sel = gamma_all[["subj", "unique_ch_names", "ch_name"] + list(cond_map.keys())]
lfp_sel = lfp_all[["subj", "unique_ch_names", "ch_name"] + list(cond_map.keys())]
df_merged = gamma_sel.merge(lfp_sel, on=["subj", "unique_ch_names", "ch_name"], suffixes=("_gamma", "_lfp"))
df_long = df_merged.melt(id_vars=["subj", "unique_ch_names", "ch_name"], value_vars=[f"{c}_gamma" for c in cond_map] + [f"{c}_lfp" for c in cond_map], var_name="cond_resp", value_name="value")
df_long["cond"] = df_long["cond_resp"].str.replace("_gamma", "", regex=False).str.replace("_lfp", "", regex=False)
df_long["resp_type"] = df_long["cond_resp"].str.split("_").str[-1]  # gamma o lfp
df_pivot = df_long.pivot_table(index=["subj", "unique_ch_names", "ch_name", "cond"], columns="resp_type", values="value", fill_value=0).reset_index()
def get_respt(row):
    if row["gamma"] == 1:
        return "gamma"
    elif row["gamma"] == 0 and row["lfp"] == 1:
        return "lfp"
    else:
        return "noresp"
df_pivot["respt"] = df_pivot.apply(get_respt, axis=1)
df_pivot["hemi"] = df_pivot["ch_name"].apply(lambda x: "sx" if "'" in x else "dx")
df_pivot["cond"] = df_pivot["cond"].map(cond_map)
df_ccep = df_pivot[["subj", "unique_ch_names", "ch_name", "hemi", "cond", "respt"]]
df_ccep = df_ccep.sort_values(by='unique_ch_names')
df_ccep['ch_name'] = df_ccep['ch_name'].str.rsplit('_').str.join('')
df_ccep['unique_ch_names'] = df_ccep['unique_ch_names'].str.replace(r'_(\d+)$', r'\1', regex=True)
df_ccep['unique_ch_names'] = df_ccep['unique_ch_names'].str.replace(r"(\D)0(\d+)$", r"\1\2", regex=True)
def filter_contact(group):
    if group['respt'].isin(['gamma', 'lfp']).any():
        # Mantieni solo le righe con gamma o lfp
        return group[group['respt'].isin(['gamma', 'lfp'])]
    else:
        row = group.iloc[0].copy()
        row['cond'] = 'noresp'
        return pd.DataFrame([row])
df_ccep = df_ccep.groupby('unique_ch_names', group_keys=False).apply(filter_contact)
df_add = df_results_gamma_stim[['subj', 'ch_name', 'conn_sum', 'resp_mean']].rename(columns={'conn_sum': 'nresp', 'resp_mean': 'amp'})
df_ccep = df_ccep.merge(df_add, on=['subj', 'ch_name'], how='inner')
df_ccep.to_csv(op.join(path_results, 'ccep_results.csv'))






# STATISTICAL ANALYSIS for CCEP (Fig.3b)
# to be RUN in R using stats_ccep_conn.R, which loads ccep_results.csv






# CREATE LFP CONVERGENCE MAP (Fig.3c)
lfp_to_plot = lfp_stats['acoustic']
lfp_to_plot['fdr_sig'] = lfp_to_plot['fdr_sig'].astype(int)
stc_ac, brain_ac = continuous_maps_one_cond(lfp_to_plot, 'fdr_sig', 'Greys', subjects_dir, lims=[0, 0.2, 1], sm=10, transparent=False, distance=0.015)
vdat_ac = np.concatenate([brain_ac._data['lh']['array'], brain_ac._data['rh']['array']])
vdat_up_ac = cortex.freesurfer.upsample_to_fsaverage(vdat_ac.squeeze(), "fsaverage5", freesurfer_subjects_dir=subjects_dir)
lfp_to_plot = lfp_stats['somatosensory']
lfp_to_plot['fdr_sig'] = lfp_to_plot['fdr_sig'].astype(int)
stc_ss, brain_ss = continuous_maps_one_cond(lfp_to_plot, 'fdr_sig', 'Greys', subjects_dir, lims=[0, 0.2, 1], sm=10, transparent=False, distance=0.015)
vdat_ss = np.concatenate([brain_ss._data['lh']['array'], brain_ss._data['rh']['array']])
vdat_up_ss = cortex.freesurfer.upsample_to_fsaverage(vdat_ss.squeeze(), "fsaverage5", freesurfer_subjects_dir=subjects_dir)
lfp_to_plot = lfp_stats['visual']
lfp_to_plot['fdr_sig'] = lfp_to_plot['fdr_sig'].astype(int)
stc_vi, brain_vi = continuous_maps_one_cond(lfp_to_plot, 'fdr_sig', 'Greys', subjects_dir, lims=[0, 0.2, 1], sm=10, transparent=False, distance=0.015)
vdat_vi = np.concatenate([brain_vi._data['lh']['array'], brain_vi._data['rh']['array']])
vdat_up_vi = cortex.freesurfer.upsample_to_fsaverage(vdat_vi.squeeze(), "fsaverage5", freesurfer_subjects_dir=subjects_dir)
vdat_up_max = (vdat_up_ac + vdat_up_ss + vdat_up_vi) / 3
colors = ['white', 'purple', 'deeppink']
cmap = mcolors.ListedColormap(colors)
cmap.set_bad(alpha=0)
lfp_conv = lfp_all.copy()
lfp_conv['gamma'] = (lfp_conv['multi'] == 7).astype(int)
lfp_conv['lfp'] = (lfp_conv['multi'] > 0).astype(int)
stc_all, brain_all = continuous_maps_gamma_vs_lfp(lfp_conv, cmap, subjects_dir)
vdat_all = np.concatenate([brain_all._data['lh']['array'], brain_all._data['rh']['array']])
vdat_up_all = cortex.freesurfer.upsample_to_fsaverage(vdat_all.squeeze(), "fsaverage5", freesurfer_subjects_dir=subjects_dir)
vertex_data = cortex.Vertex(vdat_up_all, subject, vmin=0, vmax=3, cmap=cmap)
alpha_low, alpha_high = 0.0, 0.2  # data range that maps to alpha 0..1
resp = vdat_up_max.squeeze()
alpha = (resp - alpha_low) / (alpha_high - alpha_low)
alpha = np.clip(alpha, 0.0, 1.0)  # keep alpha in [0, 1]
vertex_data_blend = vertex_data.blend_curvature(alpha=alpha)
cortex.webgl.show(data=vertex_data_blend)
fig = quickflat_with_atlas(vertex_data_blend,
                          atlas='HCPMMP1',
                          subjects_dir=subjects_dir,
                          atlas_color='w',
                          atlas_alpha=0.65,
                          atlas_lw=0.25,
                          atlas_smooth_iter=2,
                          figsize=(12, 5), show_labels=True)
plt.show()






# CREATE DF for FUNCTIONAL MAPPING
trains = pd.read_csv(op.join(path_results, 'Tab_S5.csv'))
gamma_sel = gamma_all[['subj', 'ch_name', 'acoustic', 'somatosensory', 'visual']].rename(columns={'acoustic': 'gamma_ac', 'somatosensory': 'gamma_ss', 'visual': 'gamma_vi'})
lfp_sel = lfp_all[['subj', 'ch_name', 'acoustic', 'somatosensory', 'visual']].rename(columns={'acoustic': 'lfp_ac', 'somatosensory': 'lfp_ss', 'visual': 'lfp_vi'})
resp = gamma_sel.merge(lfp_sel, on=['subj', 'ch_name'], how='outer')
resp['resp_audio'] = np.select([resp['gamma_ac'] == 1, resp['lfp_ac'] == 1], ['gamma', 'lfp'], default='none')
resp['resp_somato'] = np.select([resp['gamma_ss'] == 1, resp['lfp_ss'] == 1], ['gamma', 'lfp'], default='none')
resp['resp_video'] = np.select([resp['gamma_vi'] == 1, resp['lfp_vi'] == 1], ['gamma', 'lfp'], default='none')
resp = resp[['subj', 'ch_name', 'resp_audio', 'resp_somato', 'resp_video']]
trains = trains.merge(resp, how='inner')
trains["audio_gamma"] = (trains["resp_audio"] == "gamma").astype(int)
trains["audio_lfp"] = trains["resp_audio"].isin(["gamma", "lfp"]).astype(int)
trains["audio_none"] = (trains["resp_audio"] == "none").astype(int)
trains["somato_gamma"] = (trains["resp_somato"] == "gamma").astype(int)
trains["somato_lfp"] = trains["resp_somato"].isin(["gamma", "lfp"]).astype(int)
trains["somato_none"] = (trains["resp_somato"] == "none").astype(int)
trains["video_gamma"] = (trains["resp_video"] == "gamma").astype(int)
trains["video_lfp"] = trains["resp_video"].isin(["gamma", "lfp"]).astype(int)
trains["video_none"] = (trains["resp_video"] == "none").astype(int)
trains = trains.loc[~((trains['somato']==1) & (~trains['body'].astype(str).str.contains('M')))]  # select contacts associated with sensations on hands or arms, exluding other body parts
trains_for_stat = trains[['subj', 'ch_name', 'video', 'video_gamma', 'video_lfp', 'video_none', 'somato', 'somato_gamma', 'somato_lfp', 'somato_none', 'audio', 'audio_gamma', 'audio_lfp', 'audio_none']]
trains_for_stat = trains_for_stat.rename(columns={
    "video": "video_resp",
    "audio": "audio_resp",
    "somato": "somato_resp"
})
cols = ["video_resp", "audio_resp", "somato_resp"]
trains_for_stat[cols] = trains_for_stat[cols].astype("Int64")
trains_for_stat.dropna(inplace=True)
trains_for_stat["unique_ch_names"] = trains_for_stat["subj"].astype(str) + "_" + trains_for_stat["ch_name"].astype(str)
trains_for_stat = trains_for_stat.merge(all_subj_coords[["unique_ch_names", "glasser"]], on="unique_ch_names", how="left")
visual_areas = [
    "L_V1_ROI", "R_V1_ROI",
    "L_V2_ROI", "R_V2_ROI",
    "L_V3_ROI", "R_V3_ROI",
]
somatomotor_areas = [
    "L_4_ROI",  "R_4_ROI",
    "L_3a_ROI", "R_3a_ROI",
    "L_3b_ROI", "R_3b_ROI",
    "L_1_ROI",  "R_1_ROI",
    "L_2_ROI",  "R_2_ROI",
]
audio_areas = [
    "L_A1_ROI", "R_A1_ROI",
    "L_LBelt_ROI", "R_LBelt_ROI",
    "L_MBelt_ROI", "R_MBelt_ROI",
    "L_PBelt_ROI", "R_PBelt_ROI",
]
trains_for_stat["video_anat"] = trains_for_stat["glasser"].isin(visual_areas).astype(int)
trains_for_stat["somato_anat"] = trains_for_stat["glasser"].isin(somatomotor_areas).astype(int)
trains_for_stat["audio_anat"] = trains_for_stat["glasser"].isin(audio_areas).astype(int)
coord_cols = ["x_norm_surf", "y_norm_surf", "z_norm_surf", "x_norm_fsav", "y_norm_fsav", "z_norm_fsav", "hemis"]
trains_for_stat = trains_for_stat.merge(
    all_subj_coords[["unique_ch_names"] + coord_cols],
    on="unique_ch_names",
    how="left"
)

conditions = [
    trains_for_stat["audio_resp"] == 1,
    trains_for_stat["somato_resp"] == 1,
    trains_for_stat["video_resp"] == 1,
]
choices = [1, 2, 3]
trains_for_stat["multi_plot"] = np.select(conditions, choices, default=0)
trains_for_stat["video_multi"] = np.select(
    [
        trains_for_stat["video_resp"] == 0,
        trains_for_stat["video_anat"] == 1,
        trains_for_stat["video_gamma"] == 1,
        (trains_for_stat["video_lfp"] == 1) & (trains_for_stat["video_gamma"] == 0),
        trains_for_stat["video_none"] == 1
    ],
    [0, 1, 2, 3, 4],
    default=np.nan
)

trains_for_stat["somato_multi"] = np.select(
    [
        trains_for_stat["somato_resp"] == 0,
        trains_for_stat["somato_anat"] == 1,
        trains_for_stat["somato_gamma"] == 1,
        (trains_for_stat["somato_lfp"] == 1) & (trains_for_stat["somato_gamma"] == 0),
        trains_for_stat["somato_none"] == 1
    ],
    [0, 1, 2, 3, 4],
    default=np.nan
)

trains_for_stat["audio_multi"] = np.select(
    [
        trains_for_stat["audio_resp"] == 0,
        trains_for_stat["audio_anat"] == 1,
        trains_for_stat["audio_gamma"] == 1,
        (trains_for_stat["audio_lfp"] == 1) & (trains_for_stat["audio_gamma"] == 0),
        trains_for_stat["audio_none"] == 1
    ],
    [0, 1, 2, 3, 4],
    default=np.nan
)
print("# of contacts eliciting visual experience:", trains_for_stat['video_resp'].sum())
print("# of contacts eliciting somatomotor experience:", trains_for_stat['somato_resp'].sum())
print("# of contacts eliciting auditory experience:", trains_for_stat['audio_resp'].sum())
trains_for_stat.to_csv(op.join(path_results, 'trains_x_stat.csv'))





# CREATE TRAINS MAP per EACH CONDITION (Fig.4b)
subject = 'fsaverage'
conds = ["audio", "somato", "video"]
conds_gamma = ["acoustic", "somatosensory", "visual"]
cmap='Greys'
for cond, cond_gamma in zip(conds, conds_gamma):
    trains_for_stat[f'{cond}_resp'] = trains_for_stat[f'{cond}_resp'].astype(int)
    trains_for_stat.drop_duplicates('unique_ch_names', inplace=True)
    stc_trains, brain_trains = continuous_maps_one_cond(trains_for_stat, f'{cond}_resp', cmap, subjects_dir, lims=[0, 0.2, 1], sm=10, transparent=False, distance=0.015)
    vdat_trains = np.concatenate([brain_trains._data['lh']['array'], brain_trains._data['rh']['array']])
    vdat_up_trains = cortex.freesurfer.upsample_to_fsaverage(vdat_trains.squeeze(), "fsaverage5", freesurfer_subjects_dir=subjects_dir)
    vertex_data = cortex.Vertex(vdat_up_trains, subject, vmin=0, vmax=vdat_up_trains.max(), cmap='copper')
    alpha_low, alpha_high = 0.1, 0.3  # data range that maps to alpha 0..1
    resp = vdat_up_trains.squeeze()
    alpha = (resp - alpha_low) / (alpha_high - alpha_low)
    alpha = np.clip(alpha, 0.0, 1.0)  # keep alpha in [0, 1]
    vertex_data_blend = vertex_data.blend_curvature(alpha=alpha)
    cortex.webgl.show(data=vertex_data_blend, colorbar=True)






# STATISTICAL ANALYSIS for FUNCTIONAL MAPPING (Fig.4c-d)
# to be RUN in R using stats_ies.R, which loads trains_x_stat.csv






# CREATE TRAINS SURFACE and FLATMAP for EACH CONDITION (Fig.S14-16)
conds = ["audio", "somato", "video"]
cmaps = ['Greens', 'Reds', 'Blues']
green = mcolors.LinearSegmentedColormap.from_list("bg", ["black", "green"])
red = mcolors.LinearSegmentedColormap.from_list("br", ["black", "red"])
blue = mcolors.LinearSegmentedColormap.from_list("bb", ["black", "blue"])
cmaps_b = [green, red, blue]
for cond, cmap, cmap_b in zip(conds, cmaps, cmaps_b):
    trains_for_stat[f'{cond}_resp'] = trains_for_stat[f'{cond}_resp'].astype(int)
    trains_for_stat.drop_duplicates('unique_ch_names', inplace=True)
    surface_fsav(trains_for_stat, f'{cond}_resp', subjects_dir=subjects_dir, cmap=cmap_b, surf='inflated', scale=10, surf_color='white')
    flatmap_fsav(trains_for_stat, f'{cond}_resp', subjects_dir=subjects_dir, cmap=cmap_b)






# CREATE GAMMA AUC and OFFSET MAPs per EACH MODALITY calculated with time win == 1ms (Fig. S18-S19)
gamma_path = op.join(path_base, 'gamma_analyses')
temporal_threshold = 1
methods = ['unc_ts', 'fdr_ts', 'bon_ts', 'cluster_ts']
gamma_1ms_stats = {}
def compute_significance(ts_array, threshold):
    ts = np.array(remove_spurious_ones(ts_array, threshold))
    return int(np.sum(ts) > threshold)
for cond in ['acoustic', 'somatosensory', 'visual']:
    files = [f for f in os.listdir(gamma_path) if f.endswith('.pkl') and cond in f]
    dfs = []
    for filename in files:
        filepath = op.join(gamma_path, filename)
        subj_id = filename.split('_')[0]
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        ch_names = data.get('ch_names', [])
        if not ch_names:
            print(f"No channel names in {filename}, skipping.")
            continue
        stat_table = pd.DataFrame({'subj': subj_id, 'ch_name': ch_names})
        if 'gamma_ts' not in data:
            print(f"No gamma_ts in {filename}, skipping.")
            continue
        gamma_ts = data['gamma_ts']
        for method in methods:
            if method not in data:
                print(f"{method} not found in {filename}")
                stat_table[method.replace('_ts', '_sig')] = np.nan
                stat_table[f'{method}_auc'] = np.nan
                stat_table[f'{method}_dur'] = np.nan
                stat_table[f'{method}_lastt'] = np.nan
                continue
            ts_list = data[method]
            sig_values = []
            amp_sum, sig_durs, sig_lasts = [], [], []
            for sig_vec, ts in zip(ts_list, gamma_ts):
                sig_vec = np.array(sig_vec)
                sig_vec = remove_spurious_ones(sig_vec, temporal_threshold)
                ts = np.array(ts)
                if len(sig_vec) < len(ts):
                    sig_vec = np.pad(sig_vec, (0, len(ts) - len(sig_vec)), constant_values=0)
                elif len(sig_vec) > len(ts):
                    sig_vec = sig_vec[:len(ts)]
                sig_values.append(compute_significance(sig_vec, temporal_threshold))
                sig_idx = np.where(sig_vec == 1)[0]
                if sig_idx.size > 0:
                    amp_sum.append(np.sum(ts[sig_idx]))
                    sig_durs.append(sig_idx.size)
                    sig_lasts.append(sig_idx.max())
                else:
                    amp_sum.append(np.nan)
                    sig_durs.append(0)
                    sig_lasts.append(np.nan)
            stat_table[method.replace('_ts', '_sig')] = sig_values
            stat_table[f'{method}_auc'] = amp_sum
            stat_table[f'{method}_dur'] = sig_durs
            stat_table[f'{method}_lastt'] = sig_lasts
        dfs.append(stat_table)
    if dfs:
        df_all = pd.concat(dfs, ignore_index=True)
        merged_df = pd.merge(
            df_all,
            all_subj_coords[['subj', 'ch_name',
                             'x_norm_surf', 'y_norm_surf', 'z_norm_surf',
                             'x_norm_mri', 'y_norm_mri', 'z_norm_mri',
                             'x_norm_fsav', 'y_norm_fsav', 'z_norm_fsav']],
            on=['subj', 'ch_name'],
            how='left'
        )
        coord_cols = [c for c in merged_df.columns if c.startswith(('x_norm_', 'y_norm_', 'z_norm_'))]
        merged_clean = merged_df.dropna(subset=coord_cols)
        merged_clean = merged_clean.groupby(['subj', 'ch_name'], as_index=False).max()
        merged_clean = merged_clean.rename(columns={'ch_name': 'ch_names'})
        gamma_1ms_stats[cond] = merged_clean
    else:
        print(f"No valid data found for {cond}.")
label_map = {'acoustic': 'Auditory', 'somatosensory': 'Somatosensory', 'visual': 'Visual'}
gamma_1ms_stats = {k: v.assign(unique_ch_names=v['subj'].astype(str) + '_' + v['ch_names'].astype(str))for k, v in gamma_1ms_stats.items()}
gamma_1ms_all = df_from_stats(gamma_1ms_stats, correction='fdr_sig', marg=False)
conds = ["acoustic", "somatosensory", "visual"]
cmaps = ['Greens', 'Reds', 'Blues']
for cond, cmap in zip(conds, cmaps):
    gamma_to_plot = gamma_1ms_stats[cond].copy()
    gamma_to_plot['fdr_ts_auc_log'] = np.log10(np.abs(gamma_to_plot['fdr_ts_auc'].fillna(0)) + 1)
    gamma_to_plot['fdr_ts_auc'] = abs(gamma_to_plot['fdr_ts_auc'].fillna(0).astype(int))
    continuous_maps_one_cond(gamma_to_plot, 'fdr_ts_auc_log', cmap, subjects_dir, lims=[0, 1.5, 3], sm=10, transparent=False, distance=0.015)
    pointplot_gamma_by_area(all_subj_coords, gamma_to_plot, 'fdr_ts_auc')
for cond, cmap in zip(conds, cmaps):
    gamma_to_plot = gamma_1ms_stats[cond].copy()
    gamma_to_plot['fdr_ts_lastt'] = ((gamma_to_plot['fdr_ts_lastt']-200).fillna(0).astype(int))
    continuous_maps_one_cond(gamma_to_plot, 'fdr_ts_lastt', cmap, subjects_dir, lims=[0, 150, 300], sm=10, transparent=False, distance=0.015)
    pointplot_gamma_by_area(all_subj_coords, gamma_to_plot, 'fdr_ts_lastt')






# CALCULATE PHASE SYNCHRONY MEASURES from ORIGINAL DATA (wPLI)
bands = {'BETA': [13, 30], 'GAMMA': [50, 150]}  # declaration of parameters used by mne connectivity
con_methods = 'wpli'
mode = 'multitaper'
# Subjects' data directories and files
dir_seeg_data = path_original_data
subjects = [name for name in os.listdir(dir_seeg_data) if op.isdir(op.join(dir_seeg_data, name))]
bad_channels = pd.read_csv(op.join(dir_seeg_data,'bad_contacts.csv'))
bad_channels["ez_ch"] = bad_channels["ez_ch"].apply(ast.literal_eval)
results_path = op.join(path_base, 'wpli')
if not os.path.exists(results_path):
    os.makedirs(results_path)
for s, sub in enumerate(subjects):
    filelist = glob.glob(op.join(dir_seeg_data, sub, 'seeg')+"/*.mat") #extract only csv files which contain seeg data

    for file in filelist:
        sub_id = file.split('/')[-1].split('_')[0]

        channels = pd.read_csv(file.split('epochs')[0] + 'channels.tsv', sep='\t')
        mat_data = pym.read_mat(op.join(file))
        eeg_data = mat_data['data']
        eeg_data = eeg_data.transpose((2, 0, 1))
        eeg_data_info = mne.create_info(list(channels['name']), 1000, 'seeg', None)

        epo = mne.EpochsArray(eeg_data, eeg_data_info, tmin=-0.3)
        bads = [bad for bad in bad_channels[bad_channels['subj_id']==sub_id]['ez_ch'].values[0] if bad in epo.ch_names]
        epo.info['bads'] = bads
        epo.drop_channels(epo.info['bads'])

        bl = epo.copy().crop(tmin=-0.3, tmax=0)
        post1 = epo.copy().crop(tmin=0, tmax=0.3)  # 300 ms from 0
        post2 = epo.copy().crop(tmin=0.15, tmax=0.45) #300 ms from 150 ms post stimulus

        for band, freq_band in bands.items():
            fmin = freq_band[0]
            fmax = freq_band[1]

            #compute connectivity
            con_bl = mne_connectivity.spectral_connectivity_epochs(bl, method=con_methods, mode=mode, sfreq=None,
                                                                       fmin=fmin, fmax=fmax, faverage=True,
                                                                       verbose=False)

            con_post1= mne_connectivity.spectral_connectivity_epochs(post1, method=con_methods, mode=mode, sfreq=None,
                                                                        fmin=fmin, fmax=fmax, faverage=True,
                                                                        verbose=False)

            con_post2= mne_connectivity.spectral_connectivity_epochs(post2, method=con_methods, mode=mode, sfreq=None,
                                                                        fmin=fmin, fmax=fmax, faverage=True,
                                                                        verbose=False)

            rfilename = file.split('/')[-1].split('epochs')[0] + 'wPLI_' + band.lower()

            np.savez(
                op.join(results_path, rfilename),
                baseline=con_bl.get_data('dense').squeeze(),
                post1=con_post1.get_data('dense').squeeze(),
                post2=con_post2.get_data('dense').squeeze(),
                ch_names=epo.ch_names
            )






# CALCULATE wPLI between UCs vs GRCs and UCs vs LRCs
band = ["beta", "gamma"]
path_base_conn = op.join(path_base, 'wpli')
os.makedirs(path_results, exist_ok=True)
contacts_table = all_subj_coords
erp_clust = lfp_all
gamma_clust = gamma_all
posterior = ['parietal', 'occipital', 'temporal']
erp_dict = {s: df.reset_index(drop=True) for s, df in erp_clust.groupby("subj")}
gamma_dict = {s: df.reset_index(drop=True) for s, df in gamma_clust.groupby("subj")}
for b in band:
    file_list = [f for f in os.listdir(path_base_conn) if b in f]
    file_list = natsorted(file_list)
    records = []
    for f in file_list:
        sbj_name = f.split('_')[0]

        # Skip excluded subjects
        if sbj_name not in erp_dict:
            continue

        contacts_sub = contacts_table[contacts_table['subj'] == sbj_name]
        task_string = f.split('_')[1]

        # modality
        task = [t for t in ["acoustic", "visual", "somatosensory"] if t in task_string][0]

        erp_clust_sb = erp_dict[sbj_name]
        gamma_clust_sb = gamma_dict.get(sbj_name, pd.DataFrame())
        all_ch_names = erp_clust_sb["ch_name"].values

        # --- stim & hemi ---
        if "left" in task_string:
            stim, hemi = "R", "B"
        elif "right" in task_string:
            stim, hemi = "L", "B"
        elif "bilat" in task_string:
            stim, hemi = "B", "B"
        else:
            hemi = "L" if len(erp_clust_sb['ch_name'].iloc[0].split('_')[0]) == 2 else "R"
            stim = "B" if task == "visual" else ("L" if hemi == "R" else "R")

        # Load connectivity data
        conn = np.load(op.join(path_base_conn, f), allow_pickle=True)
        baseline, post1, post2, ch_names = conn["baseline"], conn["post1"], conn["post2"], conn["ch_names"]

        # Lower triangle indices
        i, j = np.tril_indices(len(ch_names), k=-1)
        pairs = [(ch_names[x], ch_names[y]) for x, y in zip(i, j)]

        #connectivity arrays
        bl_values = baseline[i, j]
        early_values = post1[i, j]
        late_values = post2[i, j]

        # Loop over channel pairs
        for (chan1, chan2), bl_val, early_val, late_val in zip(pairs, bl_values, early_values, late_values):

            if chan1 not in all_ch_names or chan2 not in all_ch_names:
                continue

            lobe1 = contacts_sub.loc[contacts_sub['ch_name'] == chan1, 'lobe'].values[0]
            lobe2 = contacts_sub.loc[contacts_sub['ch_name'] == chan2, 'lobe'].values[0]

            # --- cluster assignment chan1 ---
            if not gamma_clust_sb.empty and gamma_clust_sb.loc[gamma_clust_sb['ch_name'] == chan1, task].any():
                cluster1, lb1 = "G", ""
            elif erp_clust_sb.loc[erp_clust_sb['ch_name'] == chan1, task].any():
                cluster1, lb1 = "L", ""
            else:
                cluster1, lb1 = "U", ""
                if lobe1 == "frontal":
                    lb1 = "F"
                elif lobe1 in posterior:
                    lb1 = "P"

            # --- cluster assignment chan2 ---
            if not gamma_clust_sb.empty and gamma_clust_sb.loc[gamma_clust_sb['ch_name'] == chan2, task].any():
                cluster2, lb2 = "G", ""
            elif erp_clust_sb.loc[erp_clust_sb['ch_name'] == chan2, task].any():
                cluster2, lb2 = "L", ""
            else:
                cluster2, lb2 = "U", ""
                if lobe2 == "frontal":
                    lb2 = "F"
                elif lobe2 in posterior:
                    lb2 = "P"

            cluster = ''.join(sorted(cluster1 + cluster2))
            cluster_lobe = cluster + ''.join(sorted(lb1 + lb2))

            # Compute distance
            pos_ch1 = contacts_sub.loc[contacts_sub['ch_name'] == chan1, ['x_norm_mri', 'y_norm_mri', 'z_norm_mri']].values
            pos_ch2 = contacts_sub.loc[contacts_sub['ch_name'] == chan2, ['x_norm_mri', 'y_norm_mri', 'z_norm_mri']].values
            dist = np.linalg.norm(pos_ch1 - pos_ch2)

            # Save record
            records.append({
                "subject": sbj_name,
                "ch1": chan1,
                "ch2": chan2,
                "dist": dist,
                "clust1": cluster1,
                "clust2": cluster2,
                "clust": cluster,
                "clust_lobe": cluster_lobe,
                "lobe1": lobe1,
                "lobe2": lobe2,
                "hemi": hemi,
                "stim": stim,
                "conn_bl": bl_val,
                "conn_post_early": early_val,
                "conn_post_late": late_val,
                "task": task,
            })
    subj_stk = pd.DataFrame.from_records(records)
    subj_stk.to_csv(op.join(path_results, f"table_connectivity_all_lobes_{b}.csv"), index=False)

# STATISTICAL ANALYSIS for wPLI
# to be RUN in R using stats_wpli_conn.R, which loads table_connectivity_all_lobes_beta.csv &  table_connectivity_all_lobes_gamma.csv