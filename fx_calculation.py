from pymatreader import read_mat
import os
from itcfpy.process import remove_stim_artifact
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