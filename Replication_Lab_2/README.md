# Replication Laboratory #2 | Can the Model Keep Learning?

## Overview

This repository contains a proxy replication for ANLY 735. The reference paper is Klein et al. (2026), *Plasticity Loss in Deep Reinforcement Learning: A Survey*. Because the reference article is a survey rather than one executable study, this project does not directly reproduce the survey's data, code, or deep reinforcement learning results.

The project tests a narrower behavioral question: after the same input-environment change, do two models with similar pre-change performance differ in their rate and completeness of post-change learning?

## Repository structure

```text
.
├── analysis/
│   ├── lab02_analysis.py
│   ├── model_summary_metrics.csv                 # Created by the analysis script
│   ├── post_change_learning_trajectories.csv     # Created by the analysis script
│   ├── seed_level_metrics.csv                    # Created by the analysis script
│   ├── post_change_learning_curve.png            # Created by the analysis script
│   └── retention_curve.png                       # Created by the analysis script
├── replication-lab-2.qmd
├── reference_plasticity.bib
├── requirements.txt
└── README.md
```

## Data

The analysis uses the public scikit-learn Digits data set, loaded through `sklearn.datasets.load_digits()`. The data set is bundled with scikit-learn; no download, authentication, API key, or restricted data access is required.

- Observations: 1,797 handwritten digits
- Features: 64 pixel values per image (an 8 × 8 gray scale image)
- Target: digit class from 0 to 9
- Train/test split: stratified 70/30 split using seed 2026

The environmental change is a fixed permutation of the 64 input-pixel columns. Labels remain unchanged.

## Requirements

Use Python 3.10+ and install the dependencies:

```bash
python -m pip install -r requirements.txt
```

If you do not use `requirements.txt`, install the packages directly:

```bash
python -m pip install numpy pandas matplotlib scikit-learn
```

## Reproduce the analysis

1. Clone the repository and enter the project folder.

```bash
git clone <YOUR-REPOSITORY-URL>
cd anly735-lab02-Parm1486
```

2. Install packages.

```bash
python -m pip install -r requirements.txt
```

3. Run the analysis script from the project root.

```bash
python Replication_Lab_2/replication_lab2_plasticity.py
```

4. Confirm that the script creates these files in `analysis/`:

```text
model_summary_metrics.csv
post_change_learning_trajectories.csv
seed_level_metrics.csv
post_change_learning_curve.png
retention_curve.png
```

5. Render the report after the analysis files exist:

```bash
quarto render replication_lab_2.qmd
```

This creates a DOCX report named `replication_lab_2.docx`.

## Experiment settings

| Setting | Value |
|---|---:|
| Data split seed | 2026 |
| Pixel-permutation seed | 517 |
| Model seeds | 11, 22, 33, 44, 55 |
| Pre-change epochs | 20 |
| Post-change epochs | 30 |
| Recovery threshold | 90% changed-environment accuracy |
| MLP hidden units | 32 |
| Learning rate | 0.03 |
| Batch size | 64 |
| Model A L2 alpha | 0.001 |
| Model B L2 alpha | 0.0 |

## Outputs and interpretation

The primary behavioral measures are:

- Pre-change accuracy on the original representation.
- Immediate accuracy decline after the pixel-permutation change.
- Post-change learning curve on the changed representation.
- Recovery rate during the first ten post-change epochs.
- Epochs to 90% changed-environment accuracy.
- Final changed-environment accuracy after 30 post-change epochs.
- Accuracy retained on the original representation after adaptation.

This is a behavioral proxy investigation. It does not identify internal causal mechanisms such as gradient pathology, rank collapse, or dormant neurons.

## Reference

Klein, T., Luther, C., McAuliffe, M., Miklautz, L., Plant, C., & Tschiatschek, S. (2026). *Plasticity Loss in Deep Reinforcement Learning: A Survey*. arXiv:2411.04832. https://doi.org/10.48550/arXiv.2411.04832

