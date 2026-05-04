import pandas as pd
import os.path as op
import os
from fx_calculation import (import_data, run_gamma, run_gamma_habituation, run_lfp)

#run_lfp_habituation, run_lfp_bipo,
#                      continuous_maps_one_cond, continuous_maps_multi, continuous_maps_gamma_vs_lfp, reject_outliers,
#                      compute_ccep_connectivity, run_gamma_and_lfp_examples, flatmap_fsav, surface_fsav,
#                      print_stats_per_area, print_results, print_crossmodal_per_area, compute_gamma_stats_by_area,
#                      remove_spurious_ones, pointplot_gamma_by_area, df_from_stats, quickflat_with_atlas)


from itcfpy.plot import surface, flatmap, surface_by_area
from itcfpy.spatial import mni2fsav_coords, map_annot
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




# -----------------------------------------------------------------------------------------------------------------------------------------------------------
# CALCULATION of GAMMA and LFP ACTIVITY
# iterations over subjects and conditions takes a lot of computations (up to 3 days on a single PC) - we run them in parallel on https://www.indaco.unimi.it/
# -----------------------------------------------------------------------------------------------------------------------------------------------------------

# LISTS and DEFINE CONDITIONS for MONO- and BI-LATERALLY IMPLANTED SUBJECTS
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

# end of iteration
