# Graph Surgery: Tuning the Connectome for Numerosity Dynamics
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

Code and data for the paper **"Graph Surgery: Tuning the Connectome to Predict Distinct Neural Dynamics of Subitizing and Estimation"**.

This repository contains the Python implementation of the **"Graph Surgery" framework**, a bio-computational approach that embeds cognitive constraints (monotonicity) into a static resting-state connectome to predict event-related potentials (ERPs) for numerosity tasks.

---

## 🧠 Theoretical Background

The simulation bridges the gap between structural connectivity and dynamic cognition by:
1.  **Mapping** fMRI-identified enumeration regions (Parietal/Frontal) to a 638-node resting-state connectome.
2.  **Calibrating** lateral inhibition ($\beta^*$) to ensure the network satisfies the *monotonicity constraint* (distinct activations for quantities 1-6).
3.  **Simulating** emergent neural dynamics using a Recurrent On-Center Off-Surround (OCOS) model.
4.  **Comparing** the virtual EEG output against empirical data (Subitizing vs. Estimation).

**Key Finding:** Estimation dynamics align with empirical EEG immediately ($\approx 0$ms lag, P2 component), while Subitizing dynamics align only during late-stage processing ($\approx 376$ms lag, P3 component), suggesting distinct temporal mechanisms.

---

## 📂 Repository Structure

```text
.
├── code/
│   └── simulation.py       # Main script: Calibrates beta, runs OCOS dynamics, plots results
├── data/
│   ├── GroupAverage_rsfMRI_matrix.mat  # Structural scaffold (See Data Description)
│   └── numerosity_EEG.mat              # Empirical validation data
├── plots/
│   └── python_results_plot.png         # Generated output figure
└── README.md
