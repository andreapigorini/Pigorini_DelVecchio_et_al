# Pigorini_DelVecchio_et_al
Code for data analysis for
The boundaries of cortical recruitment during simple conscious perception
by
Andrea Pigorini*, Maria Del Vecchio*, Ezequiel P. Mikulan, Flavia Zauli, Alessandra Calcagno, Davide Albertini, Simone Russo, Piergiorgio D’Orio, Ivana Sartori, Pietro Avanzini@, Marcello Massimini@

* equal contribution
@ shared senior authorship

---

## Overview

This repository contains the analysis code associated with the manuscript:

> The boundaries of cortical recruitment during simple conscious perception

The repository was developed to ensure full reproducibility of the analyses and figures presented in the paper using the accompanying SEEG database.

The project combines:

- intracranial SEEG recordings
- broadband evoked responses (LFPs)
- high-frequency gamma activity
- cortico-cortical evoked potentials (CCEPs)
- intracranial electrical stimulation (iES)
- phase-synchrony analyses (wPLI)
- cortical surface and flatmap visualizations on fsaverage

The pipeline reproduces the main and supplementary figures of the manuscript, including:

- gamma and LFP responsiveness maps
- spatio-temporal boundary analyses
- cross-modal convergence maps
- connectivity analyses
- functional mapping analyses
- principal-gradient analyses
- habituation analyses

This repository is intended as a companion reproducibility resource for the manuscript and associated database, rather than a general-purpose software package.

---

## Associated dataset

The repository is associated with a publicly available SEEG database.

Dataset DOI: TBD

The dataset contains:

- epoched SEEG recordings in BIDS format
- channel metadata
- electrode coordinates
- derived gamma/LFP statistical outputs
- connectivity matrices
- functional mapping tables

Some raw datasets (e.g. original CCEP recordings) are not distributed because of storage constraints; derived connectivity matrices are provided instead.

---

## Repository

GitHub repository:

https://github.com/andreapigorini/Pigorini_DelVecchio_et_al

---

## Citation

If you use this repository or dataset, please cite:

> Pigorini A., Del Vecchio M., Mikulan E.P. et al.  
> The boundaries of cortical recruitment during simple conscious perception.  
> DOI: TBD

---

## License

This repository and associated dataset are distributed under the:

Creative Commons Attribution 4.0 International (CC BY 4.0)

https://creativecommons.org/licenses/by/4.0/

---

## Operating system

Analyses were developed and tested under:

- Ubuntu 22.04.5 LTS (Jammy)

---

## Python environment

The full analysis environment is provided as:

Pigorini_DelVecchio_env.yml

Exported using:

conda env export

### Recreate the environment

conda env create -f Pigorini_DelVecchio_env.yml
conda activate Pigorini_DelVecchio_env

---

## Main dependencies

The pipeline relies on the following scientific Python ecosystem:

- Python
- MNE-Python
- mne-connectivity
- NumPy
- SciPy
- Pandas
- Matplotlib
- Seaborn
- NiBabel
- PyVista
- PyCortex
- Neuromaps
- Statsmodels

In addition, the repository depends on a custom package:

itcfpy

used for:
- stimulation artifact removal
- coordinate transforms
- cortical projections
- electrode utilities

This package is not distributed in this repository and must be available in the Python environment.


---

## Main scripts

### main.py

Main analysis pipeline coordinating:

- gamma analyses
- LFP analyses
- coordinate transformations
- anatomical labeling
- responsiveness statistics
- cortical mapping
- habituation analyses
- connectivity analyses
- functional mapping analyses
- wPLI analyses

---

### fx_calculation.py

Core computational functions for:

- gamma-band analyses
- LFP analyses
- habituation analyses
- statistical thresholding
- contact classification
- connectivity computation
- coordinate processing

---

### fx_plot.py

Visualization utilities for:

- cortical surfaces
- fsaverage flatmaps
- continuous cortical maps
- multimodal overlays
- atlas projections
- PyCortex visualizations

---

### R scripts

#### stats_ccep_conn.R

Statistical analyses for CCEP connectivity analyses.

#### stats_ies.R

Statistical analyses for intracranial electrical stimulation (iES) mapping.

#### stats_wpli_conn.R

Statistical analyses for phase synchrony (wPLI) analyses.

---

## Dataset organization

The repository assumes a specific BIDS-derived directory structure.

Example:

per-seeg/
└── sub-01/
    └── seeg/
        ├── sub-01_task-acoustictask_run-01_epochs.mat
        ├── sub-01_task-acoustictask_run-01_channels.tsv
        ├── sub-01_task-acoustictask_run-01_electrodes.tsv
        └── ...

The pipeline is tightly coupled to this organization.

Paths are intentionally explicit and should be manually adapted inside main.py.

---

## Coordinate systems and anatomical mapping

The pipeline uses:

- MNI coordinates
- fsaverage projections
- FreeSurfer surfaces
- HCP-MMP1/Glasser atlas
- Desikan-Killiany atlas
- Margulies principal cortical gradient

Surface projections are generated using:

- FreeSurfer
- PyCortex
- Neuromaps

---

## Computational requirements

Several parts of the pipeline are computationally demanding.

The full gamma/LFP analyses across all subjects may require:

- multiple days on a single workstation

The original analyses were run in parallel on the INDACO HPC cluster (University of Milan):

https://www.indaco.unimi.it/

Users attempting full reruns are strongly encouraged to use:
- HPC infrastructure
- parallel execution
- cached intermediate outputs

---

## Reproducibility notes

The repository is designed primarily for:

- figure reproducibility
- verification of statistical analyses
- transparency of processing steps

It is not optimized as:
- a standalone software package
- an installable toolbox
- a GUI-based framework
- a generalized SEEG analysis suite

Several assumptions are hard-coded, including:
- dataset structure
- naming conventions
- stimulation conditions
- coordinate spaces
- subject/session organization

This design was intentionally chosen to maximize transparency and exact reproducibility of the published analyses.

---

## Figures reproduced by the pipeline


The scripts reproduce analyses associated with:

| Figure | Content |
|---|---|
| Fig. 1 | Experimental procedure, representative responses, contact coverage, and distribution along Principal Gradient 1 |
| Fig. 2 | Spatio-temporal boundaries of gamma recruitment |
| Fig. 3 | Gamma vs LFP convergence and connectivity |
| Fig. 4 | Functional mapping analyses |
| Fig. S1-S19 | Supplementary analyses and maps |

The corresponding code blocks are explicitly annotated throughout main.py.


---

## Notes on excluded data

The repository excludes:

- raw CCEP recordings
- some intermediate heavy files
- subject-specific clinical imaging data

However, all derived measures required to reproduce manuscript-level results are included or generated by the pipeline.

---

## Contact

Andrea Pigorini
Università degli Studi di Milano
Milan, Italy
andrea.pigorini@unimi.it

GitHub:
https://github.com/andreapigorini/Pigorini_DelVecchio_et_al


---

## Disclaimer

This repository is provided for scientific reproducibility purposes.

Because of the complexity of intracranial SEEG analyses, successful execution requires:
- familiarity with MNE/Python neurophysiology workflows
- FreeSurfer/PyCortex installations
- correct filesystem organization
- manual adaptation of paths and environment variables

The code reflects the exact analysis framework used for the manuscript and prioritizes reproducibility over software abstraction or general usability.

