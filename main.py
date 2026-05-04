import pandas as pd
import os.path as op
import os
from fx_calculation import (import_data, run_gamma, run_gamma_habituation, run_lfp, mni2fsav_coords, map_annot)

#run_lfp_habituation, run_lfp_bipo,
#                      continuous_maps_one_cond, continuous_maps_multi, continuous_maps_gamma_vs_lfp, reject_outliers,
#                      compute_ccep_connectivity, run_gamma_and_lfp_examples, flatmap_fsav, surface_fsav,
#                      print_stats_per_area, print_results, print_crossmodal_per_area, compute_gamma_stats_by_area,
#                      remove_spurious_ones, pointplot_gamma_by_area, df_from_stats, quickflat_with_atlas)


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
path_base = '/home/andrea/data/Pigorini_DelVecchio_et_al'  # please download the data from DOI:
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
