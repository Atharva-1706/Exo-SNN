# Exo-SNN
AI-enabled Detection of Exoplanets from Noisy Astronomical Light Curves 


# ExoSNN v1 — TESS Exoplanet Candidate Detection and Vetting

An end-to-end AI-assisted pipeline for detecting, classifying, explaining, and astrophysically vetting transit-like signals in NASA TESS light curves.

The system combines **Box Least Squares (BLS) detection**, a **three-branch neural architecture**, **Spiking Neural Network (SNN) temporal analysis**, **Grad-CAM explainability**, and independent astrophysical validation checks to distinguish promising planetary transit candidates from false positives and unreliable detections.

> **Important:** ExoSNN is a candidate detection and vetting system. Its AI confidence or final pipeline status must not be interpreted as formal exoplanet confirmation.

---

## Table of Contents

- [Overview](#overview)
- [Key Results](#key-results)
- [Scientific Motivation](#scientific-motivation)
- [Pipeline Architecture](#pipeline-architecture)
- [Pipeline Workflow](#pipeline-workflow)
- [1. TESS Data Acquisition](#1-tess-data-acquisition)
- [2. Light Curve Preprocessing](#2-light-curve-preprocessing)
- [3. BLS Transit Detection](#3-bls-transit-detection)
- [4. Phase Folding](#4-phase-folding)
- [5. Three-Branch Neural Classifier](#5-three-branch-neural-classifier)
- [6. Synthetic Pretraining](#6-synthetic-pretraining)
- [7. Real-TESS Fine-Tuning](#7-real-tess-fine-tuning)
- [8. Physical Vetting](#8-physical-vetting)
- [9. Explainable AI](#9-explainable-ai)
- [10. Conservative Status Logic](#10-conservative-status-logic)
- [False-Positive Rejection](#false-positive-rejection)
- [Odd-Even Transit Test](#odd-even-transit-test)
- [Eclipsing Binary Detection](#eclipsing-binary-detection)
- [Boundary-Peak Detection](#boundary-peak-detection)
- [Transit-Count Validation](#transit-count-validation)
- [Example Candidates](#example-candidates)
- [Evaluation](#evaluation)
- [Confusion Matrix](#confusion-matrix)
- [Project Structure](#project-structure)
- [Important Files](#important-files)
- [Installation](#installation)
- [Running the Project](#running-the-project)
- [Running the Dashboard](#running-the-dashboard)
- [Running a Specific TESS Target](#running-a-specific-tess-target)
- [Training](#training)
- [Benchmarking](#benchmarking)
- [Unseen Evaluation](#unseen-evaluation)
- [Data Sources](#data-sources)
- [Model Checkpoint](#model-checkpoint)
- [Reproducibility](#reproducibility)
- [Limitations](#limitations)
- [Scientific Interpretation](#scientific-interpretation)
- [Future Improvements](#future-improvements)
- [References](#references)
- [Disclaimer](#disclaimer)
- [Project Status](#project-status)

---

# Overview

**ExoSNN v11.1** is an AI-assisted astronomical pipeline designed to analyze noisy stellar light curves from NASA's **Transiting Exoplanet Survey Satellite (TESS)**.

The fundamental problem is that a planetary transit can appear as a very small decrease in stellar brightness. Real astronomical observations also contain:

- Instrumental noise
- Stellar variability
- Data gaps
- Detrending artifacts
- Cosmic-ray contamination
- Isolated dips
- Eclipsing binaries
- Periodic stellar phenomena
- Insufficient numbers of observed transits

Consequently, a high signal-to-noise ratio alone is not sufficient to establish that a detected signal is planetary.

ExoSNN therefore uses multiple independent stages:

```text
TESS Light Curve
       |
       v
Data Cleaning & Detrending
       |
       v
BLS Period Search
       |
       v
Candidate Period Selection
       |
       v
Global + Local Phase Folding
       |
       +---------------------------+
       |                           |
       v                           v
Global CNN                   Local CNN
       |                           |
       +-------------+-------------+
                     |
                     v
              SNN Temporal Branch
                     |
                     v
              Neural Ensemble
                     |
                     v
             AI Morphology Score
                     |
                     v
       +-----------------------------+
       | Independent Physical Vetting|
       +-----------------------------+
          |      |       |       |
          v      v       v       v
       Transit Odd-Even Secondary BLS
       Count   Test     Eclipse   Quality
          \      |       |       /
           \     |       |      /
            +----+-------+-----+
                     |
                     v
            Conservative Status
                     |
                     v
             Final Candidate
                Assessment
