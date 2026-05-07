from pymatreader import read_mat
import os
from itcfpy.spatial import make_bip_lists, mni2fsav_coords
from scipy.stats import ttest_ind
from scipy import stats
from statsmodels.stats.multitest import multipletests
import pickle
import pandas as pd
import os.path as op
import numpy as np
import mne
import cortex
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from cortex.polyutils import Surface
from itcfpy.spatial import find_closest_vert
from scipy.sparse import csr_matrix
from mne.stats import permutation_cluster_1samp_test
from functools import reduce
import seaborn as sns






def import_data(fpath, fname):
    """
    Load epoched SEEG data and channel metadata, and convert them into an MNE Epochs object.

    The function reads stimulation-locked epochs from a MATLAB file and the corresponding
    channel information from a BIDS-like channels.tsv file. Data are rearranged to match
    the MNE convention: epochs × channels × time samples. All channels are assigned the
    SEEG channel type, and the resulting data structure is returned as an MNE EpochsArray.

    Parameters
    ----------
    fpath : str
        Path to the folder containing the epoch and channel files.
    fname : str
        Base filename used to identify the corresponding *_epochs.mat and *_channels.tsv files.

    Returns
    -------
    epo : mne.EpochsArray
        Epoched SEEG data with channel names, sampling frequency, and SEEG channel types.
    """

    # Read epoched data from the MATLAB file.
    epo = read_mat(op.join(fpath, fname + '_epochs.mat'))['data']

    # Reorder dimensions to match the MNE format: epochs x channels x time samples.
    epo = np.transpose(epo, (2, 0, 1))

    # Read channel metadata from the corresponding TSV file.
    chs = pd.read_csv(op.join(fpath, fname + '_channels.tsv'), sep='\t')

    # Create the MNE info structure using channel names, sampling frequency, and channel types.
    ch_names = chs['name'].tolist()
    ch_types = ['seeg'] * len(ch_names)
    sfreq = chs['sampling_frequency'][0]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

    # Create an MNE EpochsArray aligned to stimulus onset, with epochs starting at -300 ms.
    epo = mne.EpochsArray(epo, info, tmin=-0.3)

    return epo





def run_gamma(imported_data, subj, cond, path_save_gamma):
    """
    Compute stimulus-locked gamma-band responses for all SEEG contacts.

    This function implements the gamma-responsiveness analysis used to identify
    Gamma Responding Contacts (GRCs). After stimulation-artifact removal, single-trial
    time-frequency representations are computed with Morlet wavelets in the 50–150 Hz
    range. Gamma power is baseline-normalized, averaged across frequencies, and tested
    at each time point against the pre-stimulus baseline. For each contact, the function
    stores the average gamma time series, sample-wise p-values, and binary significance
    vectors obtained with uncorrected, Bonferroni-corrected, and Benjamini–Yekutieli
    FDR-corrected statistics.

    Parameters
    ----------
    imported_data : mne.Epochs
        Epoched SEEG data aligned to stimulus onset.
    subj : str
        Subject identifier.
    cond : str
        Stimulation condition or sensory modality.
    path_save_gamma : str
        Output directory where the gamma analysis pickle file is saved.

    Returns
    -------
    None
        Saves a pickle file containing gamma time series, p-values, significance masks,
        and channel names for the specified subject and condition.
    """

    # Remove stimulation artifacts before time-frequency decomposition.
    epo = remove_stim_artifact(imported_data, kind='median')

    # Define gamma-band Morlet wavelet parameters and baseline interval.
    freqs_seeg = np.arange(50, 150, 10)
    n_cycles_seeg = 4
    bl_min = -0.20
    bl_max = -0.05

    # Initialize containers for contact-wise time series and statistical results.
    all_ch_sig = pd.DataFrame(columns=['ch_names', 'uncorrected', 'bonferroni', 'fdr'])
    all_ch_sig['ch_names'] = epo.info['ch_names']
    all_pval = []
    all_sig_unc = []
    all_sig_bon = []
    all_sig_fdr = []
    all_avg_gamma = []

    for ch in epo.info['ch_names']:

        print('running gamma analyses for ' + subj + ', condition: ' + cond + ', contact: ' + ch)

        # Compute single-trial Morlet time-frequency power for the current contact.
        power_seeg = epo.copy().pick(ch).apply_baseline((bl_min, bl_max)).compute_tfr(
            method='morlet',
            freqs=freqs_seeg,
            n_cycles=n_cycles_seeg,
            use_fft=True,
            return_itc=False,
            decim=1,
            n_jobs=1,
            average=False
        )

        # Z-score gamma power relative to the pre-stimulus baseline.
        power_seeg_z = power_seeg.copy().apply_baseline((bl_min, bl_max), mode="zscore")

        # Average gamma power across frequencies and restrict the analysis window to -200/+500 ms.
        trials_gamma = np.mean(power_seeg_z, axis=2).squeeze(axis=1).T
        trials_gamma = trials_gamma[(power_seeg.times > -0.2) & (power_seeg.times <= 0.5), :]
        time_gamma = power_seeg.times[(power_seeg.times > -0.2) & (power_seeg.times <= 0.5)]

        # Compute the trial-averaged gamma time series for the current contact.
        avg_gamma = np.mean(trials_gamma, axis=1)
        all_avg_gamma.append(avg_gamma)

        # Perform sample-wise statistical comparisons against the pre-stimulus baseline.
        pvals = np.empty(np.shape(trials_gamma)[0])
        baseline = np.mean(trials_gamma[0:150, :], axis=0)
        for t in range(0, np.shape(trials_gamma)[0]):
            t_dist = trials_gamma[t, :]
            _, p = ttest_ind(t_dist, baseline, equal_var=False)
            pvals[t] = p
        all_pval.append(pvals)

        # Store uncorrected significance at p < 0.05.
        sig_unc = (pvals < 0.05).astype(int)
        all_sig_unc.append(sig_unc)

        # Apply Bonferroni correction to the post-stimulus time window.
        n_samples = len(pvals)
        sig_bon = np.zeros(n_samples, dtype=int)
        _, pvals_bonf, _, _ = multipletests(pvals[200:], alpha=0.05, method='bonferroni')
        sig_bon[200:] = (pvals_bonf < 0.05).astype(int)
        all_sig_bon.append(sig_bon)

        # Apply Benjamini-Yekutieli FDR correction to the post-stimulus time window.
        n_samples = len(pvals)
        sig_fdr = np.zeros(n_samples, dtype=int)
        _, pvals_fdr, _, _ = multipletests(pvals[200:], alpha=0.05, method='fdr_by')
        sig_fdr[200:] = (pvals_fdr < 0.05).astype(int)
        all_sig_fdr.append(sig_fdr)

    # Save gamma time series and statistical outputs for the current subject and condition.
    session_data = {
        'gamma_ts': all_avg_gamma,
        'pval_ts': all_pval,
        'unc_ts': all_sig_unc,
        'bon_ts': all_sig_bon,
        'fdr_ts': all_sig_fdr,
        'ch_names': epo.info['ch_names'],
    }

    with open(op.join(path_save_gamma, subj + '_' + cond + '_gamma.pkl'), 'wb') as f:
        pickle.dump(session_data, f)






def run_lfp(imported_data, subj, cond, path_save_lfp):
    """
    Compute stimulus-locked broadband LFP responses for all SEEG contacts.

    This function implements the LFP-responsiveness analysis used to identify contacts
    showing significant broadband evoked responses. After stimulation-artifact removal,
    single-trial SEEG signals are z-scored relative to the pre-stimulus baseline and
    averaged across trials to obtain the evoked LFP response. Sample-wise statistical
    comparisons are then performed against the baseline. For each contact, the function
    stores the average LFP time series, p-values, and binary significance vectors obtained
    with uncorrected, Bonferroni-corrected, and Benjamini–Yekutieli FDR-corrected statistics.

    Parameters
    ----------
    imported_data : mne.Epochs
        Epoched SEEG data aligned to stimulus onset.
    subj : str
        Subject identifier.
    cond : str
        Stimulation condition or sensory modality.
    path_save_lfp : str
        Output directory where the LFP analysis pickle file is saved.

    Returns
    -------
    None
        Saves a pickle file containing LFP time series, p-values, significance masks,
        and channel names for the specified subject and condition.
    """

    # Remove stimulation artifacts before estimating broadband evoked responses.
    epo = remove_stim_artifact(imported_data, kind='median')

    # Initialize containers for contact-wise time series and statistical results.
    all_ch_sig = pd.DataFrame(columns=['ch_names', 'uncorrected', 'bonferroni', 'fdr'])
    all_ch_sig['ch_names'] = epo.info['ch_names']
    all_pval = []
    all_sig_unc = []
    all_sig_bon = []
    all_sig_fdr = []
    all_avg_lfp = []

    for ch in epo.info['ch_names']:

        print('running lfp analyses for ' + subj + ', condition: ' + cond + ', contact: ' + ch)

        # Extract single-trial SEEG data for the current contact.
        seeg = epo.get_data(copy=True)
        ch_idx = epo.ch_names.index(ch)
        times = epo.times

        # Define the pre-stimulus baseline used for trial-wise z-scoring.
        bl_min = -0.20
        bl_max = -0.05
        bl_mask = (times >= bl_min) & (times <= bl_max)

        # Z-score each trial relative to its own pre-stimulus baseline.
        seeg_ch = seeg[:, ch_idx, :]
        bl_mean = np.mean(seeg_ch[:, bl_mask], axis=1, keepdims=True)
        bl_std = np.std(seeg_ch[:, bl_mask], axis=1, ddof=0, keepdims=True)
        trials_lfp = (seeg_ch - bl_mean) / bl_std
        trials_lfp = trials_lfp.T

        # Restrict the analysis window to -200/+500 ms to match the gamma analysis.
        trials_lfp = trials_lfp[100:800, :]

        # Compute the trial-averaged broadband LFP response for the current contact.
        avg_lfp = np.mean(trials_lfp, axis=1)
        all_avg_lfp.append(avg_lfp)

        # Perform sample-wise statistical comparisons against the pre-stimulus baseline.
        pvals = np.empty(np.shape(trials_lfp)[0])
        baseline = np.mean(trials_lfp[0:150, :], axis=0)
        for t in range(0, np.shape(trials_lfp)[0]):
            t_dist = trials_lfp[t, :]
            _, p = ttest_ind(t_dist, baseline, equal_var=False)
            pvals[t] = p
        all_pval.append(pvals)

        # Store uncorrected significance at p < 0.05.
        sig_unc = (pvals < 0.05).astype(int)
        all_sig_unc.append(sig_unc)

        # Apply Bonferroni correction to the post-stimulus time window.
        n_samples = len(pvals)
        sig_bon = np.zeros(n_samples, dtype=int)
        _, pvals_bonf, _, _ = multipletests(pvals[200:], alpha=0.05, method='bonferroni')
        sig_bon[200:] = (pvals_bonf < 0.05).astype(int)
        all_sig_bon.append(sig_bon)

        # Apply Benjamini-Yekutieli FDR correction to the post-stimulus time window.
        n_samples = len(pvals)
        sig_fdr = np.zeros(n_samples, dtype=int)
        _, pvals_fdr, _, _ = multipletests(pvals[200:], alpha=0.05, method='fdr_by')
        sig_fdr[200:] = (pvals_fdr < 0.05).astype(int)
        all_sig_fdr.append(sig_fdr)

    # Save LFP time series and statistical outputs for the current subject and condition.
    session_data = {
        'lfp_ts': all_avg_lfp,
        'pval_ts': all_pval,
        'unc_ts': all_sig_unc,
        'bon_ts': all_sig_bon,
        'fdr_ts': all_sig_fdr,
        'ch_names': epo.info['ch_names'],
    }

    with open(op.join(path_save_lfp, subj + '_' + cond + '_lfp.pkl'), 'wb') as f:
        pickle.dump(session_data, f)






def run_gamma_and_lfp_examples(imported_data, ch, cmap, vlim=(0, 3), show_lfp=False, ylim=(-4, 4), correction='fdr_by', min_len=20):
    """
    Compute and visualize gamma-band or LFP responses for a single SEEG contact.

    This function provides a compact visualization tool for illustrating example responses
    at the single-contact level. It computes time-frequency representations (gamma) or
    broadband LFP responses, performs statistical testing against baseline, and displays
    the resulting time courses along with significant activation periods.

    Parameters
    ----------
    imported_data : mne.Epochs
        Epoched SEEG data aligned to stimulus onset.
    ch : str
        Channel name to analyze.
    cmap : str
        Colormap used for time-frequency visualization.
    vlim : tuple, optional
        Color limits for the time-frequency plot.
    show_lfp : bool, optional
        If False, display gamma results; if True, display LFP results.
    ylim : tuple, optional
        Y-axis limits for LFP plot.
    correction : str, optional
        Multiple comparison correction method (e.g., 'fdr_by', 'bonferroni').
    min_len : int, optional
        Minimum length (in samples) for contiguous significant segments.

    Returns
    -------
    None
        Displays plots of gamma or LFP responses for the selected contact.
    """

    # Remove stimulation artifact.
    epo = remove_stim_artifact(imported_data, kind='median')

    # Define gamma analysis parameters.
    freqs_seeg = np.arange(50, 150, 10)
    n_cycles_seeg = 4
    bl_min = -0.20
    bl_max = -0.05

    # Compute time-frequency representation.
    power_seeg = epo.copy().pick(ch).apply_baseline((bl_min, bl_max)).compute_tfr(
        method='morlet', freqs=freqs_seeg,
        n_cycles=n_cycles_seeg, use_fft=True,
        return_itc=False, decim=1, n_jobs=1, average=False
    )

    # Z-score normalization.
    power_seeg_z = power_seeg.copy().apply_baseline((bl_min, bl_max), mode="zscore")

    # Plot average time-frequency map.
    power_seeg_z.average().plot(tmin=-0.1, tmax=0.5, cmap=cmap, vlim=vlim)

    if show_lfp is False:

        # Compute gamma time series.
        trials_gamma = np.mean(power_seeg_z, axis=2).squeeze(axis=1).T
        trials_gamma = trials_gamma[100:800, :]
        avg_gamma = np.mean(trials_gamma, axis=1)

        # Statistical comparison vs baseline.
        pvals = np.empty(np.shape(trials_gamma)[0])
        baseline = np.mean(trials_gamma[0:150, :], axis=0)
        for t in range(len(pvals)):
            _, pvals[t] = ttest_ind(trials_gamma[t, :], baseline, equal_var=False)

        # Multiple comparison correction.
        _, pvals_corr, _, _ = multipletests(pvals, alpha=0.05, method=correction)
        sig_corr = (pvals_corr < 0.05).astype(int)
        sig_corr = remove_spurious_ones(sig_corr, min_len=min_len)

        # Plot gamma time series with significant segments.
        fig, ax = plt.subplots(figsize=(10, 4))
        times = np.linspace(-200, 500, 700)
        sig_corr[times < 11] = 0

        ax.plot(times, avg_gamma, color='black', lw=1.2)
        ax.axhline(0, color='gray', ls='--', lw=0.8)
        ax.set_xlim(-100, 500)
        ax.set_xlabel('time (ms)')
        ax.set_ylabel('z-score')
        ax.set_title(ch)

        # Shade significant regions.
        ax.fill_between(times, avg_gamma, 0, where=sig_corr > 0, color='grey', alpha=0.25)

        # Draw horizontal bars indicating significant intervals.
        y_min = ax.get_ylim()[0]
        for i in range(1, len(sig_corr)):
            if sig_corr[i] and not sig_corr[i - 1]:
                start = times[i]
            if sig_corr[i - 1] and not sig_corr[i]:
                end = times[i]
                ax.hlines(y=y_min, xmin=start, xmax=end, color='black', lw=2)

        plt.tight_layout()
        plt.show()

    else:

        # Compute LFP responses for a single contact.
        seeg = epo.get_data(verbose=False)
        ch_idx = epo.ch_names.index(ch)
        times = epo.times

        bl_mask = (times >= -0.20) & (times <= -0.05)
        seeg_ch = seeg[:, ch_idx, :]

        # Trial-wise z-scoring.
        bl_mean = np.mean(seeg_ch[:, bl_mask], axis=1, keepdims=True)
        bl_std = np.std(seeg_ch[:, bl_mask], axis=1, keepdims=True)
        trials_lfp = (seeg_ch - bl_mean) / bl_std
        trials_lfp = trials_lfp.T

        # Restrict time window.
        times_ms = times * 1000
        mask = (times_ms >= -200) & (times_ms <= 500)
        trials_lfp = trials_lfp[mask, :]
        times_ms = times_ms[mask]

        # Compute average and SEM.
        avg_lfp = np.mean(trials_lfp, axis=1)
        sem_lfp = np.std(trials_lfp, axis=1) / np.sqrt(trials_lfp.shape[1])

        # Plot LFP.
        plt.figure(figsize=(7, 4))
        plt.plot(times_ms, avg_lfp, color="black", lw=2)
        plt.fill_between(times_ms, avg_lfp - sem_lfp, avg_lfp + sem_lfp, color="black", alpha=0.3)
        plt.axvline(0, color="k", ls="--")
        plt.axhline(0, color="gray", ls=":")
        plt.xlim(-100, 500)
        plt.ylim(ylim)
        plt.xlabel("Time (ms)")
        plt.ylabel("LFP (z-score)")
        plt.title(f"Average LFP - {ch}")
        plt.tight_layout()
        plt.show()






def run_gamma_habituation(imported_data, stat_data, subj, cond, path_save_gamma_hab):
    """
    Compute trial-wise gamma activity to assess habituation effects.

    This function quantifies changes in gamma responses across trials by integrating
    gamma power over significant time points only. It uses precomputed statistical masks
    (e.g., FDR significance) to restrict the analysis to meaningful activations, and
    compares post-stimulus gamma activity to baseline.

    Parameters
    ----------
    imported_data : mne.Epochs
        Epoched SEEG data.
    stat_data : dict
        Dictionary containing statistical outputs (e.g., 'fdr_ts').
    subj : str
        Subject identifier.
    cond : str
        Experimental condition.
    path_save_gamma_hab : str
        Output directory.

    Returns
    -------
    None
        Saves gamma habituation metrics per contact.
    """

    epo = remove_stim_artifact(imported_data, kind='median')

    freqs_seeg = np.arange(50, 150, 10)
    n_cycles_seeg = 4
    bl_min = -0.20
    bl_max = -0.05

    gamma_hab_all = []
    gamma_hab_bl_all = []

    for ch in epo.info['ch_names']:

        print(f'running gamma habituation for {subj}, {cond}, {ch}')

        # Compute gamma power.
        power_seeg = epo.copy().pick(ch).apply_baseline((bl_min, bl_max)).compute_tfr(
            method='morlet', freqs=freqs_seeg,
            n_cycles=n_cycles_seeg, use_fft=True,
            return_itc=False, decim=1, n_jobs=1, average=False
        )

        power_seeg_z = power_seeg.copy().apply_baseline((bl_min, bl_max), mode="zscore")

        trials_gamma = np.mean(power_seeg_z, axis=2).squeeze(axis=1).T
        trials_gamma = trials_gamma[100:800, :]

        # Mask non-significant samples using precomputed FDR results.
        ts = np.array(stat_data["fdr_ts"])[stat_data["ch_names"].index(ch), :]
        trials_gamma[ts == 0, :] = np.nan

        # Compute post-stimulus and baseline gamma integration per trial.
        gamma_hab = np.nansum(trials_gamma[210:], axis=0)
        gamma_hab_bl = np.nansum(trials_gamma[0:190], axis=0)

        gamma_hab_all.append(gamma_hab)
        gamma_hab_bl_all.append(gamma_hab_bl)

    # Save results.
    session_data = {
        'gamma_hab': gamma_hab_all,
        'gamma_hab_bl': gamma_hab_bl_all,
        'ch_names': epo.info['ch_names']
    }

    with open(op.join(path_save_gamma_hab, subj + '_' + cond + '_gamma_hab.pkl'), 'wb') as f:
        pickle.dump(session_data, f)






def remove_spurious_ones(sig, min_len):
    """
    Remove short spurious sequences of ones from a binary significance vector.

    This function enforces temporal continuity by removing clusters of significant
    samples shorter than a specified minimum length.

    Parameters
    ----------
    sig : array-like
        Binary array indicating significant samples.
    min_len : int
        Minimum length required for a valid cluster.

    Returns
    -------
    sig : ndarray
        Cleaned binary array.
    """

    sig = sig.copy()
    start = None

    for i in range(len(sig)):
        if sig[i] == 1 and start is None:
            start = i
        elif sig[i] == 0 and start is not None:
            if i - start < min_len:
                sig[start:i] = 0
            start = None

    if start is not None and len(sig) - start < min_len:
        sig[start:] = 0

    return sig






def reject_outliers(ch_names_ts, time_series, abs_clip=1000, plot=True):
    """
    Identify and optionally visualize outlier SEEG channels based on amplitude.

    Channels exceeding a fixed absolute amplitude threshold are flagged as outliers.
    The function provides diagnostic plots showing good vs bad channels and the
    distribution of amplitudes.

    Parameters
    ----------
    ch_names_ts : object
        Object containing channel names.
    time_series : array-like
        Data array (channels x time).
    abs_clip : float
        Absolute amplitude threshold.
    plot : bool
        Whether to display diagnostic plots.

    Returns
    -------
    bads : list
        List of channel names classified as outliers.
    """

    data = np.atleast_2d(np.asarray(time_series))
    ch_names = list(ch_names_ts.unique_ch_names)

    if data.shape[0] != len(ch_names):
        raise ValueError("Mismatch between data and channel metadata")

    max_abs = np.max(np.abs(data), axis=1)
    bad_mask = max_abs > abs_clip
    bads = [ch for ch, bad in zip(ch_names, bad_mask) if bad]

    print(f"Identified {len(bads)} bad channels out of {len(ch_names)}")

    if plot:
        fig, axs = plt.subplots(3, 1, figsize=(10, 8), constrained_layout=True)
        times = np.arange(data.shape[1])

        axs[0].plot(times, data[~bad_mask].T, alpha=0.4)
        axs[0].set_title("Good channels")

        if np.any(bad_mask):
            axs[1].plot(times, data[bad_mask].T, color='red')
            axs[1].set_title("Outlier channels")

        axs[2].hist(max_abs, bins=60)
        axs[2].axvline(abs_clip, color='r', linestyle='--')

        plt.show()

    return bads






def mni2fsav_coords(mni_coords, fname_affine, fname_warp, dir_tmp='/tmp'):
    '''
    mni_coords = pandas (n_coords, 3) [x_norm_mri, y_norm_mri, z_norm_mri]
    fname_affine = *0GenericAffine.mat
    fname_warp = *1InverseWarp.nii.gz
    '''
    import mne
    import os.path as op
    import pandas as pd
    from nipype.interfaces.ants import ApplyTransformsToPoints

    coords_lps = mni_coords.copy()
    coords_lps['x_norm_mri'] *= -1
    coords_lps['y_norm_mri'] *= -1

    fname_lps = op.join(dir_tmp, 'tmp_lps_coords.csv')
    coords_lps[['x_norm_mri', 'y_norm_mri', 'z_norm_mri']].to_csv(fname_lps, index=False)

    fname_lps_norm = op.join(dir_tmp, 'tmp_lps_coords_norm.csv')

    at = ApplyTransformsToPoints()
    at.inputs.dimension = 3
    at.inputs.input_file = fname_lps
    at.inputs.output_file = fname_lps_norm
    at.inputs.transforms = [fname_affine, fname_warp]
    at.inputs.invert_transform_flags = [True, False]
    # at.cmdline
    at.run()

    coords_lps_norm = pd.read_csv(fname_lps_norm)
    coords_ras_norm = coords_lps_norm.copy()
    coords_ras_norm['x_norm_mri'] *= -1
    coords_ras_norm['y_norm_mri'] *= -1
    coords_ras_norm.rename(columns={'x_norm_mri': 'x_norm_fsav', 'y_norm_mri': 'y_norm_fsav', 'z_norm_mri': 'z_norm_fsav'}, inplace=True)

    return coords_ras_norm






def map_annot(coords, subjects_dir, kind='lobe'):
    """
    Find the corresponding annotation (lobe or yeo) of a set of electrode coordinates.

    Parameters
    ----------
    coords: pandas.DataFrame
        The dataframe with the electrode coordinates
    subjects_dir: str
        Freesurfer's subjects dir
    kind: str
        Annotation type (lobe or yeo)
    plot: Bool
        Whether or not to plot the surface and the electrodes (color-coded by lobe)

    Returns
    -------
    lobes_df: pandas.DataFrame
        A dataframe with electrode names and corresponding lobes.

    """

    import mne
    import pandas as pd
    import os.path as op
    import numpy as np
    from nibabel.freesurfer import read_annot

    subject = 'mni152'

    if kind == 'lobe':
        fname_annot = op.join(subjects_dir, subject, 'label', '%s.lobes_file.annot')
    elif kind == 'yeo':
        fname_annot = op.join(subjects_dir, subject, 'label', '%s.Yeo2011_7Networks_N1000.annot')
    elif kind == 'glasser':
        fname_annot = op.join(subjects_dir, subject, 'label', '%s.glasser.annot')
    elif kind == 'destrieux':
        fname_annot = op.join(subjects_dir, subject, 'label', '%s.aparc.a2009s.annot')
    elif kind == 'desikan':
            fname_annot = op.join(subjects_dir, subject, 'label', '%s.aparc.annot')
    else:
        print('Unknown Annotation')
        return

    labels = {h: mne.read_labels_from_annot(annot_fname=fname_annot % h, subjects_dir=
                                            subjects_dir, subject=subject) for h in ['rh', 'lh']}

    annot = {h: read_annot(fname_annot % h) for h in ['rh', 'lh']}

    fname_surf = op.join(subjects_dir, subject, 'surf', '%s.pial')
    surfs = {h: mne.read_surface(fname_surf % h) for h in ['rh', 'lh']}

    names = []
    lobes = []
    codes = []

    for ix, r in coords.iterrows():
        print(r['name'])
        pos = r[['x_norm_surf', 'y_norm_surf', 'z_norm_surf']].tolist()
        if any(np.isnan(pos)):
            print('NaN Found - Aborting')
            return 'NaN Found'
        hemi = 'rh' if pos[0] > 0 else 'lh'
        dist_all = np.sqrt(np.sum((surfs[hemi][0] - pos) ** 2, axis=1))

        # lobe_code = annot[hemi][0][dist_all.argmin()]
        min_ind = -1
        lobe_code = -1
        while lobe_code == -1:
            min_ind += 1
            lobe_code = annot[hemi][0][np.argpartition(dist_all, min_ind)[min_ind]]

        codes.append(lobe_code)
        lobes.append(annot[hemi][2][lobe_code].decode('utf-8'))
        names.append(r['name'])

    lobes_df = pd.DataFrame({'name': names, kind: lobes})

    coords[kind] = lobes
    colors = {k.decode('utf-8'): tuple(annot['rh'][1][c, :3]/256) for c, k in enumerate(annot['rh'][2])}

    return lobes_df






def print_results(stats_dict, all_subj_coords, all_subj_coords_plot_ez, label_map,
                  name="", correction='fdr', plot=False, printt=True):
    """
    Summarize responsive contacts across modalities and cortical lobes.

    This function computes the number and percentage of responsive gray-matter contacts
    for each sensory modality, after excluding contacts located in the seizure-onset zone
    and contacts rejected as outliers. It also reports cross-modal overlaps and provides
    a lobe-wise breakdown of responsiveness. Optionally, it plots the percentage of
    responsive contacts per lobe and modality.

    Parameters
    ----------
    stats_dict : dict
        Dictionary containing one statistical dataframe per modality.
    all_subj_coords : pandas.DataFrame
        Contact-level dataframe after exclusion of pathological/artifactual contacts.
        Must contain unique_ch_names and lobe columns.
    all_subj_coords_plot_ez : pandas.DataFrame
        Contact-level dataframe before final exclusions, including gray-matter contacts
        and, when available, seizure-onset-zone information.
    label_map : dict
        Dictionary mapping modality names to display labels.
    name : str, optional
        Analysis label used in printed outputs and plot titles.
    correction : str, optional
        Column name used to define statistical significance.
    plot : bool, optional
        Whether to plot lobe-wise responsiveness.
    printt : bool, optional
        Whether to print summary tables.

    Returns
    -------
    summary_df : pandas.DataFrame
        Summary table with overall responsiveness and cross-modal overlaps.
    lobe_df : pandas.DataFrame
        Lobe-wise table reporting responsive and total contacts per modality.
    A : set
        Set of acoustic responsive contacts.
    S : set
        Set of somatosensory responsive contacts.
    V : set
        Set of visual responsive contacts.
    """

    # Keep modalities in a fixed order when available.
    ordered = [c for c in ['acoustic', 'somatosensory', 'visual'] if c in stats_dict]

    def _mk_uid(df, subj_col='subj', ch_col='ch_names'):
        """
        Create subject-specific contact identifiers.
        """

        if ch_col not in df.columns and 'ch_name' in df.columns:
            ch_col = 'ch_name'

        return df[subj_col].astype(str) + '_' + df[ch_col].astype(str)

    # Define the initial gray-matter denominator.
    gm_df = all_subj_coords_plot_ez[['subj', 'ch_name']].drop_duplicates().copy()
    uids_gm = set((gm_df['subj'].astype(str) + '_' + gm_df['ch_name'].astype(str)).tolist())

    # Identify contacts belonging to the seizure-onset zone, when available.
    if 'ez' in all_subj_coords_plot_ez.columns:
        soz_df = all_subj_coords_plot_ez.loc[
            all_subj_coords_plot_ez['ez'] == 1,
            ['subj', 'ch_name']
        ].drop_duplicates()

        uids_soz = set(
            (soz_df['subj'].astype(str) + '_' + soz_df['ch_name'].astype(str)).tolist()
        ) & uids_gm
    else:
        uids_soz = set()

    # Identify outlier contacts as gray-matter contacts absent from the final coordinate table.
    uids_after = set(all_subj_coords['unique_ch_names'].astype(str).tolist())
    uids_outliers = (uids_gm - uids_soz) - uids_after

    # Final denominator: gray-matter contacts excluding seizure-onset-zone contacts and outliers.
    uids_den = uids_gm - (uids_soz | uids_outliers)
    n_den = len(uids_den)

    def _corr_mask(df):
        """
        Return a boolean significance mask from either binary or p-value columns.
        """

        if correction not in df.columns:
            raise KeyError(f"Column '{correction}' not found in dataframe.")

        s = pd.to_numeric(df[correction], errors='coerce')
        vals = set(s.dropna().unique().tolist())

        if vals.issubset({0, 1}) and len(vals) > 0:
            return s.astype(bool).to_numpy()

        return (s < 0.05).to_numpy()

    # Build modality-specific sets of responsive contacts.
    resp_sets = {}

    for cond in ordered:
        df = stats_dict[cond].copy()

        if 'ch_names' not in df.columns and 'ch_name' in df.columns:
            df = df.rename(columns={'ch_name': 'ch_names'})

        mask = _corr_mask(df)
        df_resp = df.loc[mask, ['subj', 'ch_names']].copy()
        uids_resp = set(_mk_uid(df_resp).tolist()) if len(df_resp) else set()

        # Restrict responsive contacts to the final denominator.
        resp_sets[cond] = uids_resp & uids_den

    def _pct(n, den):
        """
        Compute percentage while avoiding division by zero.
        """

        return (n / den * 100.0) if den > 0 else 0.0

    # Build overall summary table.
    rows = []
    rows.append(['Denominator (GM − SOZ − outliers)', n_den, '100.00'])

    any_count = len(set().union(*resp_sets.values())) if resp_sets else 0
    rows.append(['Any responding contact (any modality)', any_count, f"{_pct(any_count, n_den):.2f}"])

    for cond in ordered:
        n_mod = len(resp_sets[cond])
        rows.append([label_map.get(cond, cond), n_mod, f"{_pct(n_mod, n_den):.2f}"])

    # Compute lobe-wise responsiveness.
    lobe_summary = []

    df_lobes = all_subj_coords[['unique_ch_names', 'lobe']].drop_duplicates().copy()
    df_lobes = df_lobes[df_lobes['unique_ch_names'].isin(uids_den)]

    for cond in ordered:
        resp = resp_sets[cond]

        for lobe, group in df_lobes.groupby('lobe'):
            uids_lobe = set(group['unique_ch_names'])
            n_lobe = len(uids_lobe)
            n_resp = len(uids_lobe & resp)
            pct = _pct(n_resp, n_lobe)

            lobe_summary.append([cond, lobe, n_resp, n_lobe, f"{pct:.2f}"])

    lobe_df = pd.DataFrame(
        lobe_summary,
        columns=['Modality', 'Lobe', 'n_resp', 'n_total', '%']
    )

    # Compute cross-modal overlaps.
    A = resp_sets.get('acoustic', set())
    S = resp_sets.get('somatosensory', set())
    V = resp_sets.get('visual', set())

    if len(A) and len(S):
        rows.append(['Overlap A∩S', len(A & S), f"{_pct(len(A & S), n_den):.2f}"])

    if len(A) and len(V):
        rows.append(['Overlap A∩V', len(A & V), f"{_pct(len(A & V), n_den):.2f}"])

    if len(S) and len(V):
        rows.append(['Overlap S∩V', len(S & V), f"{_pct(len(S & V), n_den):.2f}"])

    if len(A) and len(S) and len(V):
        rows.append(['Overlap A∩S∩V', len(A & S & V), f"{_pct(len(A & S & V), n_den):.2f}"])

    summary_df = pd.DataFrame(
        rows,
        columns=['Item', 'n', '% of contacts in grey matter']
    )

    # Print summary tables.
    if printt:
        print(f"\nSummary of responsive contacts — {name}:")
        print(summary_df.to_string(index=False))

        print("\nBreakdown per lobe:")
        print(lobe_df.to_string(index=False))

    # Optionally plot lobe-wise responsiveness for each modality.
    if plot:
        sns.set(style='whitegrid')
        fig, axes = plt.subplots(1, len(ordered), figsize=(15, 5), sharey=True)

        palette = {
            'occipital': 'blue',
            'parietal': 'red',
            'temporal': 'green',
            'frontal': 'black',
            'insula': 'yellow',
            'cingulate': 'magenta'
        }

        for i, cond in enumerate(ordered):
            ax = axes[i]
            dfp = lobe_df[lobe_df['Modality'] == cond]

            sns.barplot(
                data=dfp,
                x='Lobe',
                y=dfp['%'].astype(float),
                ax=ax,
                palette=palette
            )

            ax.set_title(label_map.get(cond, cond), fontsize=12)
            ax.set_ylabel('% responsive' if i == 0 else "")
            ax.set_xlabel('Lobe')
            ax.set_ylim(0, 100)
            ax.tick_params(axis='x', rotation=45)

        plt.suptitle(f"Percentage of responsive contacts per lobe — {name}", fontsize=14)
        plt.tight_layout()
        plt.show()

    return summary_df, lobe_df, A, S, V






def print_stats_per_area(stats_dict, all_subj_coords, all_subj_coords_plot_ez,
                         label_map, name="", correction='fdr_sig'):
    """
    Compute area-wise counts and percentages of responsive contacts for each modality.

    This function summarizes responsiveness at the anatomical-area level. For each
    cortical area, it reports the total number of valid gray-matter contacts and the
    number and percentage of contacts responding to each available sensory modality.
    The denominator excludes seizure-onset-zone contacts and outliers, consistently
    with the global responsiveness summaries.

    Parameters
    ----------
    stats_dict : dict
        Dictionary containing one statistical dataframe per modality.
    all_subj_coords : pandas.DataFrame
        Contact-level dataframe after exclusion of pathological/artifactual contacts.
        Must contain unique_ch_names and area columns.
    all_subj_coords_plot_ez : pandas.DataFrame
        Contact-level dataframe before final exclusions, including gray-matter contacts
        and, when available, seizure-onset-zone information.
    label_map : dict
        Dictionary mapping modality names to display labels.
    name : str, optional
        Analysis label used in printed output.
    correction : str, optional
        Column name used to define statistical significance.

    Returns
    -------
    area_table : pandas.DataFrame
        Area-wise table containing total contacts and responsive contacts per modality.
    """

    # Keep modalities in a fixed order when available.
    ordered = [c for c in ['acoustic', 'somatosensory', 'visual'] if c in stats_dict]

    def _mk_uid(df, subj_col='subj', ch_col='ch_names'):
        """
        Create subject-specific contact identifiers.
        """

        if ch_col not in df.columns and 'ch_name' in df.columns:
            ch_col = 'ch_name'

        return df[subj_col].astype(str) + '_' + df[ch_col].astype(str)

    # Define the initial gray-matter contact set.
    gm_df__ = all_subj_coords_plot_ez[['subj', 'ch_name']].drop_duplicates().copy()
    uids_gm__ = set((gm_df__['subj'].astype(str) + '_' + gm_df__['ch_name'].astype(str)).tolist())

    # Identify seizure-onset-zone contacts when available.
    if 'ez' in all_subj_coords_plot_ez.columns:
        soz_df__ = all_subj_coords_plot_ez.loc[
            all_subj_coords_plot_ez['ez'] == 1,
            ['subj', 'ch_name']
        ].drop_duplicates()

        uids_soz__ = set(
            (soz_df__['subj'].astype(str) + '_' + soz_df__['ch_name'].astype(str)).tolist()
        ) & uids_gm__

    else:
        uids_soz__ = set()

    # Identify outlier contacts as gray-matter contacts absent from the final table.
    uids_after__ = set(all_subj_coords['unique_ch_names'].astype(str).tolist())
    uids_outliers__ = (uids_gm__ - uids_soz__) - uids_after__

    # Final denominator: gray-matter contacts excluding seizure-onset-zone contacts and outliers.
    uids_den = uids_gm__ - (uids_soz__ | uids_outliers__)

    def _sig_mask(df):
        """
        Return a boolean significance mask from either binary or p-value columns.
        """

        col = correction
        s = pd.to_numeric(df[col], errors='coerce')
        vals = set(s.dropna().unique().tolist())

        if vals.issubset({0, 1}) and len(vals) > 0:
            return s.astype(bool).to_numpy()

        return (s < 0.05).to_numpy()

    # Build modality-specific sets of responsive contacts.
    resp_sets = {}

    for cond in ordered:
        df = stats_dict[cond].copy()

        if 'ch_names' not in df.columns and 'ch_name' in df.columns:
            df = df.rename(columns={'ch_name': 'ch_names'})

        mask = _sig_mask(df)
        df_resp = df.loc[mask, ['subj', 'ch_names']].copy()
        uids_resp = set(_mk_uid(df_resp).tolist()) if len(df_resp) else set()

        # Restrict responsive contacts to the final denominator.
        resp_sets[cond] = uids_resp & uids_den

    # Ensure that a unique contact identifier is available.
    if 'unique_ch_names' not in all_subj_coords.columns:
        all_subj_coords = all_subj_coords.copy()
        all_subj_coords['unique_ch_names'] = (
            all_subj_coords['subj'].astype(str) + '_' +
            all_subj_coords['ch_name'].astype(str)
        )

    # Map contacts to anatomical areas.
    map_area = all_subj_coords[['unique_ch_names', 'area']].drop_duplicates()

    # Compute the total number of valid contacts per area.
    den_df = pd.DataFrame({'unique_ch_names': list(uids_den)}).merge(
        map_area,
        on='unique_ch_names',
        how='left'
    )

    area_tot = (
        den_df['area']
        .value_counts()
        .rename_axis('Area')
        .rename('N in area')
        .reset_index()
    )

    def _area_counts_for(uids):
        """
        Count responsive contacts per anatomical area.
        """

        if not uids:
            return pd.Series(dtype=int)

        df_u = pd.DataFrame({'unique_ch_names': list(uids)})

        return (
            df_u
            .merge(map_area, on='unique_ch_names', how='left')['area']
            .value_counts()
        )

    # Count responsive contacts per area and modality.
    cnt_by_mod = {}

    for cond in ordered:
        cnt_by_mod[cond] = _area_counts_for(resp_sets[cond]).rename(label_map[cond])

    # Merge total contacts and modality-specific responsive counts.
    area_table = area_tot.copy()

    for cond in ordered:
        area_table = area_table.merge(
            cnt_by_mod[cond],
            left_on='Area',
            right_index=True,
            how='left'
        )

    # Add percentages of responsive contacts within each area.
    for cond in ordered:
        col_n = label_map[cond]

        if col_n not in area_table.columns:
            area_table[col_n] = 0

        area_table[col_n] = area_table[col_n].fillna(0).astype(int)

        area_table[f'{label_map[cond]} %'] = np.where(
            area_table['N in area'] > 0,
            (area_table[col_n] / area_table['N in area'] * 100).round(2),
            0.00
        ).astype(float)

    # Reorder columns for readability.
    cols = ['Area', 'N in area']

    for cond in ordered:
        lab = label_map[cond]

        if lab not in area_table.columns:
            area_table[lab] = 0
            area_table[f'{lab} %'] = 0.00

        cols += [lab, f'{lab} %']

    area_table = area_table[cols]
    area_table = area_table.sort_values(by=['Area']).reset_index(drop=True)

    print(f"\nArea-wise responsive contacts by modality — {name}:")
    print(area_table.to_string(index=False))

    return area_table






def remove_spurious_ones(sig, min_len):
    """
    Remove short spurious sequences of ones from a binary significance vector.

    This function enforces temporal continuity by removing clusters of significant
    samples shorter than a specified minimum duration. It is used to avoid classifying
    isolated or very brief significant samples as valid electrophysiological responses.

    Parameters
    ----------
    sig : array-like
        Binary array indicating significant samples, where 1 denotes significance
        and 0 denotes non-significance.
    min_len : int
        Minimum number of consecutive samples required for a significant segment
        to be retained.

    Returns
    -------
    sig : ndarray
        Cleaned binary significance vector in which short significant segments
        have been removed.
    """

    # Work on a copy to avoid modifying the input array in place.
    sig = sig.copy()

    # Track the beginning of each contiguous significant segment.
    start = None

    for i in range(len(sig)):

        # Mark the start of a new significant segment.
        if sig[i] == 1 and start is None:
            start = i

        # When the segment ends, remove it if it is shorter than min_len.
        elif sig[i] == 0 and start is not None:
            if i - start < min_len:
                sig[start:i] = 0
            start = None

    # Handle a significant segment that continues until the end of the array.
    if start is not None and len(sig) - start < min_len:
        sig[start:] = 0

    return sig






def df_from_stats(stats, correction='fdr_sig', marg=True):
    """
    Build a contact-level multimodal responsiveness dataframe from modality-specific statistics.

    This function merges acoustic, somatosensory, and visual responsiveness tables into
    a single contact-level dataframe. For each contact, it stores binary or numerical
    significance values for each modality and computes a compact multimodal code
    indicating which combination of modalities elicited a significant response.
    Contact coordinates in fsaverage and surface-normalized space are then appended.

    Parameters
    ----------
    stats : dict
        Dictionary containing modality-specific statistical dataframes.
        Expected keys are 'acoustic', 'somatosensory', and 'visual'.
    correction : str, optional
        Column name used to define responsiveness for each modality.
    marg : bool, optional
        Whether to include the Margulies principal gradient value
        (`PC1_margulies`) among the coordinate columns.

    Returns
    -------
    merged : pandas.DataFrame
        Contact-level dataframe containing modality-specific responsiveness,
        multimodal response code, coordinates, and unique contact identifiers.
    """

    # Ensure consistent naming of the channel column across modality-specific dataframes.
    for key in ['acoustic', 'somatosensory', 'visual']:
        if 'ch_names' in stats[key].columns:
            stats[key] = stats[key].rename(columns={'ch_names': 'ch_name'})

    # Extract modality-specific responsiveness columns.
    ac = stats['acoustic'][['subj', 'ch_name', correction]].rename(
        columns={correction: 'acoustic'}
    )

    ss = stats['somatosensory'][['subj', 'ch_name', correction]].rename(
        columns={correction: 'somatosensory'}
    )

    vis = stats['visual'][['subj', 'ch_name', correction]].rename(
        columns={correction: 'visual'}
    )

    # Merge acoustic, somatosensory, and visual responsiveness at the contact level.
    merged = pd.merge(ac, ss, on=['subj', 'ch_name'], how='outer')
    merged = pd.merge(merged, vis, on=['subj', 'ch_name'], how='outer')

    # Fill missing responsiveness values and encode cross-modal response combinations.
    # Coding scheme:
    # 0 = no response
    # 1 = acoustic only
    # 2 = somatosensory only
    # 3 = acoustic + somatosensory
    # 4 = visual only
    # 5 = acoustic + visual
    # 6 = somatosensory + visual
    # 7 = acoustic + somatosensory + visual
    merged[['acoustic', 'somatosensory', 'visual']] = (
        merged[['acoustic', 'somatosensory', 'visual']].fillna(0)
    )

    merged['multi'] = (
        (merged['acoustic'] > 0).astype(int) * 1 +
        (merged['somatosensory'] > 0).astype(int) * 2 +
        (merged['visual'] > 0).astype(int) * 4
    )

    # Define coordinate columns to retrieve from modality-specific tables.
    if marg is True:
        coord_cols = [
            'subj', 'ch_name',
            'x_norm_surf', 'y_norm_surf', 'z_norm_surf',
            'x_norm_fsav', 'y_norm_fsav', 'z_norm_fsav',
            'PC1_margulies'
        ]
    else:
        coord_cols = [
            'subj', 'ch_name',
            'x_norm_surf', 'y_norm_surf', 'z_norm_surf',
            'x_norm_fsav', 'y_norm_fsav', 'z_norm_fsav'
        ]

    # Collect coordinates from all modality-specific dataframes.
    ac_coords = stats['acoustic'][coord_cols]
    ss_coords = stats['somatosensory'][coord_cols]
    vis_coords = stats['visual'][coord_cols]

    coords_all = pd.concat([ac_coords, ss_coords, vis_coords], ignore_index=True)

    # Retain one coordinate entry per subject-contact pair.
    coords_all = coords_all.drop_duplicates(subset=['subj', 'ch_name'])

    # Append coordinates to the merged responsiveness dataframe.
    merged = pd.merge(merged, coords_all, on=['subj', 'ch_name'], how='left')

    # Create a subject-specific unique contact identifier.
    merged['unique_ch_names'] = merged['subj'] + '_' + merged['ch_name']

    return merged






def remove_stim_artifact(epo, win=0.0125, size=5, plot=False, kind='spline'):
    """
    Attenuate stimulation artifacts around stimulus onset in epoched SEEG data.

    This function replaces the signal within a short time window centered on stimulus
    onset using either a median-filtered estimate or a spline-based interpolation.
    A Tukey window is used to blend the corrected segment with the original signal,
    minimizing sharp discontinuities at the edges of the artifact-correction window.

    Parameters
    ----------
    epo : mne.Epochs
        Epoched SEEG data aligned to stimulation onset.
    win : float, optional
        Half-width of the artifact-correction window in seconds.
    size : int, optional
        Kernel size used for median filtering.
    plot : bool, optional
        Whether to plot the original and corrected signal for visual inspection.
    kind : {'median', 'spline'}, optional
        Artifact-correction method. If 'median', the artifact window is replaced
        with a Tukey-weighted median-filtered signal. If 'spline', the window is
        replaced with a Tukey-weighted spline interpolation.

    Returns
    -------
    epo : mne.Epochs
        Epoched SEEG data after stimulation-artifact attenuation.
    """

    import numpy as np
    from scipy.ndimage import median_filter
    from scipy.signal.windows import tukey
    import matplotlib.pyplot as plt
    from scipy.interpolate import UnivariateSpline

    # Access epoch data directly: trials x channels x time samples.
    d = epo._data

    # Define the time window centered on stimulation onset.
    artifact_mask = (epo.times > 0 - win) & (epo.times < 0 + win)
    win_times = epo.times[artifact_mask]

    # Build a Tukey window and its inverse to smoothly blend corrected and original data.
    tuk = tukey(len(win_times))
    tuk_inv = np.abs(tuk - 1)

    # Iterate across trials and channels.
    for ix_tr, tr in enumerate(d):
        for ix_ch, ch_dat in enumerate(tr):

            # Extract the original signal inside the artifact window.
            old_dat = ch_dat[artifact_mask]

            # Estimate the artifact-corrected signal using median filtering.
            medf_dat = median_filter(old_dat, size=size)
            new_dat = tuk * medf_dat + tuk_inv * old_dat

            # Estimate the artifact-corrected signal using spline interpolation.
            spl = UnivariateSpline(win_times, old_dat, w=None)
            spl_dat = tuk * spl(win_times) + tuk_inv * old_dat

            # Optionally display correction diagnostics.
            if plot:
                plt.plot(epo.times, ch_dat)
                plt.plot(win_times, old_dat, label='old')
                plt.plot(win_times, medf_dat, label='median filter')
                plt.plot(win_times, new_dat, label='median-corrected')
                plt.plot(win_times, spl_dat, label='spline-corrected')
                plt.legend()
                plt.show()

            # Replace the artifact window using the selected correction method.
            if kind == 'median':
                d[ix_tr, ix_ch, artifact_mask] = new_dat
            elif kind == 'spline':
                d[ix_tr, ix_ch, artifact_mask] = spl_dat

    # Write corrected data back into the Epochs object.
    epo._data = d

    return epo






def print_crossmodal_per_area(A, S, V, all_subj_coords, name=""):
    """
    Summarize cross-modal responsive contacts by cortical area.

    This function computes the anatomical distribution of contacts responding to more
    than one sensory modality. Given the sets of acoustic-, somatosensory-, and
    visual-responsive contacts, it calculates pairwise and trimodal overlaps and reports
    their counts for each cortical area.

    Parameters
    ----------
    A : set
        Set of acoustic-responsive contact identifiers.
    S : set
        Set of somatosensory-responsive contact identifiers.
    V : set
        Set of visual-responsive contact identifiers.
    all_subj_coords : pandas.DataFrame
        Contact-level dataframe containing subject IDs, channel names, lobes,
        cortical areas, and optionally unique contact identifiers.
    name : str, optional
        Analysis label used in printed output.

    Returns
    -------
    area_table : pandas.DataFrame
        Area-wise table containing pairwise and trimodal overlap counts.
    tot_cont : dict
        Dictionary containing total overlap counts across all cortical areas.
    """

    # Ensure that a subject-specific unique contact identifier is available.
    if 'unique_ch_names' not in all_subj_coords.columns:
        all_subj_coords = all_subj_coords.copy()
        all_subj_coords['unique_ch_names'] = (
            all_subj_coords['subj'].astype(str) + '_' +
            all_subj_coords['ch_name'].astype(str)
        )

    # Build a contact-to-area lookup table.
    map_area = all_subj_coords[['unique_ch_names', 'lobe', 'area']].drop_duplicates()

    def _area_count(uid_set):
        """
        Count the number of contacts in a given set within each cortical area.
        """

        if not uid_set:
            return pd.Series(dtype=int)

        df_u = pd.DataFrame({'unique_ch_names': list(uid_set)})
        merged = df_u.merge(map_area, on='unique_ch_names', how='left')

        return merged['area'].value_counts()

    # Compute pairwise and trimodal overlaps.
    AS = (A & S) if (len(A) and len(S)) else set()
    AV = (A & V) if (len(A) and len(V)) else set()
    SV = (S & V) if (len(S) and len(V)) else set()
    ASV = (A & S & V) if (len(A) and len(S) and len(V)) else set()

    # Count overlap contacts per cortical area.
    cnt_AS = _area_count(AS)
    cnt_AV = _area_count(AV)
    cnt_SV = _area_count(SV)
    cnt_ASV = _area_count(ASV)

    # Collect all areas containing at least one overlapping contact.
    areas_all = set(cnt_AS.index) | set(cnt_AV.index) | set(cnt_SV.index) | set(cnt_ASV.index)

    # Build the area-wise overlap table.
    area_table = pd.DataFrame({'Area': sorted(list(areas_all))})

    area_table = area_table.merge(cnt_AS.rename('A∩S'), left_on='Area', right_index=True, how='left')
    area_table = area_table.merge(cnt_AV.rename('A∩V'), left_on='Area', right_index=True, how='left')
    area_table = area_table.merge(cnt_SV.rename('S∩V'), left_on='Area', right_index=True, how='left')
    area_table = area_table.merge(cnt_ASV.rename('A∩S∩V'), left_on='Area', right_index=True, how='left')

    # Replace missing values with zero and enforce integer counts.
    area_table = area_table.fillna(0).astype({
        'A∩S': int,
        'A∩V': int,
        'S∩V': int,
        'A∩S∩V': int
    })

    # Compute global overlap totals.
    tot_cont = {
        'A∩S': area_table['A∩S'].sum(),
        'A∩V': area_table['A∩V'].sum(),
        'S∩V': area_table['S∩V'].sum(),
        'A∩S∩V': area_table['A∩S∩V'].sum()
    }

    # Append total row.
    total_row = pd.DataFrame([{'Area': 'Total', **tot_cont}])
    area_table = pd.concat([area_table, total_row], ignore_index=True)

    # Sort areas by their maximum overlap count, keeping the total row at the bottom.
    area_table_non_total = area_table[area_table['Area'] != 'Total']

    order_vals = area_table_non_total[['A∩S', 'A∩V', 'S∩V', 'A∩S∩V']].max(axis=1)

    area_table_non_total = (
        area_table_non_total
        .assign(_sort=order_vals)
        .sort_values(by=['_sort', 'Area'], ascending=[False, True])
        .drop(columns=['_sort'])
        .reset_index(drop=True)
    )

    # Re-append total row at the end.
    area_table = pd.concat(
        [area_table_non_total, area_table[area_table['Area'] == 'Total']],
        ignore_index=True
    )

    # Print the area-wise overlap table.
    print(f"\nContacts responding to two or three modalities (counts) per area — {name}:")
    print(area_table.to_string(index=False))

    return area_table, tot_cont






def compute_ccep_connectivity(path_sess, df_sub, sub, stim_bip, path_conn_save, resp_gamma_lfp_ch):
    """
    Compute CCEP-based effective connectivity from one bipolar SPES stimulation session.

    This function loads cleaned cortico-cortical evoked potentials (CCEPs) elicited by
    single-pulse electrical stimulation, z-scores each recorded channel relative to the
    pre-stimulus baseline, and extracts early and late response metrics. The early N1
    response, measured between 10 and 30 ms after stimulation, is used as an index of
    putative monosynaptic effective connectivity. Results are saved together with the
    full CCEP data and the response class of the stimulated contact.

    Parameters
    ----------
    path_sess : str
        Path to the stimulation session folder containing CCEP files.
    df_sub : pandas.DataFrame
        Subject-level contact dataframe containing channel names and gamma/LFP response
        classifications.
    sub : str
        Subject identifier.
    stim_bip : str
        Name of the stimulated bipolar contact pair.
    path_conn_save : str
        Output directory where CCEP connectivity results are saved.
    resp_gamma_lfp_ch : str or int
        Response class of the stimulated contact, typically indicating whether the
        stimulated site is gamma-responsive, LFP-only, or unresponsive.

    Returns
    -------
    None
        Saves a pickle file containing CCEP metrics, the z-scored Evoked object,
        and the response class of the stimulated contact.
    """

    # Define the expected cleaned CCEP file for the current bipolar stimulation session.
    fname_ccep = op.join(path_sess, 'Bipolar', 'evoked_autoclean.mat')

    if os.path.exists(fname_ccep):

        # Parse the stimulated bipolar channel name.
        root = stim_bip.split('-')

        if len(root[1].split("'")) > 1:
            ch_name = root[0] + '-' + root[1].split("'")[1]
        else:
            ch_name = root[0] + '-' + root[1][1:]

        # Load CCEP data, time vector, and recorded channel labels.
        ccep_base = read_mat(fname_ccep)
        fname_times = op.join(path_sess, 'Ts.mat')
        ccep_times = read_mat(fname_times)['Ts']
        ccep_ch = ccep_base['EVOKED']['labels']

        # Retrieve channels marked as bad by the preprocessing pipeline.
        ccep_bads = [
            ccep_base['EVOKED']['labels'][ix]
            for ix, i in enumerate(ccep_base['EVOKED']['bad_channels'])
            if i == 1
        ]

        # Create an MNE Evoked object from z-scored CCEP data.
        info = mne.create_info(ccep_ch, 1000, ch_types=['seeg'] * len(ccep_ch))
        ccep = ccep_base['EVOKED']['data']

        # Z-score each channel relative to the pre-stimulus baseline.
        med = np.nanmean(ccep[:, ccep_times < -20], axis=1, keepdims=1)
        std = np.nanstd(ccep[:, ccep_times < -20], axis=1, keepdims=1)
        zscore = (ccep - med) / std

        ccep = mne.EvokedArray(zscore, info, tmin=ccep_times[0] / 1000)

        # Mark bad channels and exclude contacts with indices above 18.
        ccep_bads = ccep_bads + [c for c in ccep_ch if int(c.split('-')[-1]) > 18]
        ccep.info['bads'] = ccep_bads

        ccep.drop_channels([c for c in ccep_ch if int(c.split('-')[-1]) > 18])

        ch_drop_ix = [ix for ix, c in enumerate(ccep.ch_names) if c in ccep_bads]
        n_ccep_ch = len(ccep.ch_names)

        # Extract early N1 response metrics between 10 and 30 ms.
        half_win = 5
        early_crop = ccep.copy().crop(0.01, 0.03)
        early_data = abs(early_crop.get_data())

        early_pk = early_data.argmax(1)
        early_latency = np.array([early_crop.times[ep] for ep in early_pk])
        early_latency[ch_drop_ix] = np.nan

        pk_value = np.array([
            early_crop.get_data()[r, early_pk[r]]
            for r in range(0, n_ccep_ch)
        ])
        pk_value[ch_drop_ix] = np.nan

        # Compute the mean absolute response around the early peak.
        start_early = early_pk - half_win
        start_early[start_early < 0] = 0

        end_early = early_pk + half_win
        end_early[end_early > early_data.shape[1]] = early_data.shape[1]

        early_resp = np.array([
            early_data[ch, start_early[ch]:end_early[ch]].mean()
            for ch, idx in enumerate(ccep.ch_names)
        ])
        early_resp[ch_drop_ix] = np.nan

        # Extract late CCEP response metrics between 80 and 500 ms.
        late_crop = ccep.copy().crop(0.08, 0.5)
        late_data = abs(late_crop.get_data())

        late_pk = late_data.argmax(1)
        late_latency = np.array([late_crop.times[ep] for ep in late_pk])
        late_latency[ch_drop_ix] = np.nan

        # Compute the mean absolute response around the late peak.
        start_late = late_pk - half_win
        start_late[start_late < 0] = 0

        end_late = late_pk + half_win
        end_late[end_late > late_data.shape[1]] = late_data.shape[1]

        late_resp = np.array([
            late_data[ch, start_late[ch]:end_late[ch]].mean()
            for ch, idx in enumerate(ccep.ch_names)
        ])
        late_resp[ch_drop_ix] = np.nan

        # Store the full z-scored CCEP data.
        conn_ccep_data = ccep

        # Build a channel-level connectivity table.
        conn_ccep_val = pd.DataFrame(
            columns=[
                'ch_name_bip',
                'ch_name',
                'conn',
                'n1_peak',
                'n1_auc',
                'n1_lat',
                'n2_auc',
                'n2_lat'
            ]
        )

        conn_ccep_val['ch_name_bip'] = ccep.ch_names

        # Convert bipolar channel names to monopolar contact names used in the main dataframe.
        conn_ccep_val['ch_name'] = (
            conn_ccep_val['ch_name_bip']
            .str.split('-')
            .str[0]
            .str.replace(r"^([A-Z]'+?|\w)(\d{1})$", r"\1_0\2", regex=True)
            .str.replace(r"^([A-Z]'+?|\w)(\d{2,})$", r"\1_\2", regex=True)
        )

        # Add early and late CCEP response metrics.
        conn_ccep_val['n1_peak'] = pk_value
        conn_ccep_val['n1_auc'] = early_resp

        # Define effective connectivity based on suprathreshold early N1 response.
        conn_ccep_val['conn'] = (conn_ccep_val['n1_auc'] > 5).astype(int)

        conn_ccep_val['n1_lat'] = early_latency
        conn_ccep_val['n2_auc'] = late_resp
        conn_ccep_val['n2_lat'] = late_latency

        # Append gamma/LFP response classification of each recorded contact.
        conn_ccep_val = conn_ccep_val.merge(
            df_sub[['ch_name', 'gamma_lfp']],
            on='ch_name',
            how='left'
        )

        # Store the response class of the stimulated contact.
        resp_stim_cont = resp_gamma_lfp_ch

        # Save connectivity table, full CCEP data, and stimulated-contact class.
        with open(op.join(path_conn_save, sub + '_' + stim_bip + '_conn_ccep.pkl'), 'wb') as f:
            pickle.dump(
                {
                    'conn_ccep_val': conn_ccep_val,
                    'conn_ccep_data': conn_ccep_data,
                    'resp_stim_cont': resp_stim_cont
                },
                f
            )

    else:
        print(fname_ccep + ' does not exist')