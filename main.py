import pandas as pd
import os.path as op
import os
from fx_calculation import (import_data, run_gamma, run_gamma_habituation, run_lfp, mni2fsav_coords, map_annot,
                            reject_outliers, print_stats_per_area, print_results, df_from_stats, remove_spurious_ones,
                            run_gamma_and_lfp_examples,)
from fx_plot import (surface_fsav, flatmap_fsav, continuous_maps_one_cond, pointplot_gamma_by_area)


#run_lfp_habituation, run_lfp_bipo,
#                      , continuous_maps_multi, continuous_maps_gamma_vs_lfp,
#                      compute_ccep_connectivity,  flatmap_fsav, surface_fsav,
#                      , print_crossmodal_per_area, compute_gamma_stats_by_area,
#                       , df_from_stats, quickflat_with_atlas)


from itcfpy.plot import surface, flatmap, surface_by_area
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





# CREATE GAMMA SURFACES and FLATMAPS for SPATIO-TEMPORAL BOUNDARIES per EACH MODALITY (FIGURE 2a)
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