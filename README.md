# Exo-SNN
AI-enabled Detection of Exoplanets from Noisy Astronomical Light Curves 


Astro: TESS Exoplanet Detection & Vetting Pipeline

An automated, end-to-end machine learning pipeline for discovering, classifying, and vetting exoplanet candidates from NASA's TESS (Transiting Exoplanet Survey Satellite) mission light curves.

This project integrates traditional astronomical detection methods (like Box-fitting Least Squares - BLS) with a cutting-edge Tri-Branch Ensemble classification system, featuring Spiking Neural Networks (SNNs) and Explainable AI (XAI) to provide physical verification of transit signals.

🌟 Key Features

Data Ingestion & Synthesis: Automated downloading of TESS Full Frame Image (FFI) light curves from the MAST archive, coupled with synthetic data generation for robust model training.

Dual-Stream Detection: Utilizes standard BLS (Box-fitting Least Squares) alongside custom detection algorithms to identify periodic dips in stellar brightness.

Tri-Branch Ensemble Classification: A powerful composite model architecture that includes:

SNN Branch: Spiking Neural Networks for temporally-aware pattern recognition.

Feature Extraction: Deep feature analysis of transit shapes.

Delta Modulation: Signal encoding for high-fidelity anomaly detection.

Physical Verification & XAI: Ensures that detected candidates physically make sense as exoplanets and provides Explainable AI metrics to interpret the neural network's decisions.

Automated Vetting Reports: Generates comprehensive visual reports (PNG/PDF) for individual TESS Input Catalog (TIC) targets.

Interactive UX Dashboard: A Streamlit-based web application to interactively explore light curves, run the pipeline, and view vetting reports.

📁 Project Structure

astro/
├── mastDownload/          # Downloaded TESS FITS files and light curves
├── root/
│   ├── benchmark_results/ # Output metrics and CSVs from model benchmarking
│   ├── classification/    # Core ML models (SNN, Tri-branch, XAI, Verification)
│   ├── data/              # Ingestion scripts and synthetic data generators
│   ├── detection/         # Transit search algorithms (BLS, Dual-Stream)
│   ├── real_data_cache/   # Cached NPZ files of processed TIC targets
│   ├── reporting/         # Automated vetting report generators
│   ├── reports/           # Generated output reports (PNGs)
│   ├── unseen_evaluation_200/ # Manifests and metrics for unseen target validation
│   ├── ux/                # User Interface code
│   │   └── dashboard.py   # Main Streamlit dashboard application
│   ├── weights/           # Pre-trained PyTorch model weights (.pt files)
│   ├── main_pipeline.py   # End-to-end execution script
│   ├── train_real.py      # Script for training models on real TESS data
│   ├── train_all.py       # Script for training on composite datasets
│   ├── evaluate_real.py   # Evaluation scripts for real data
│   ├── evaluate_unseen.py # Evaluation scripts for unseen validation sets
│   └── Requirements.txt   # Python package dependencies


🚀 Installation

Clone the repository:

git clone <repository-url>
cd astro


Set up a virtual environment (recommended):

python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate


Install the dependencies:

pip install -r root/Requirements.txt


🖥️ Usage

Running the Interactive Dashboard

The easiest way to explore the data and model results is through the interactive Streamlit dashboard. To launch the UX, run the following command from the project root:

python -m streamlit run root/ux/dashboard.py 


This will open the dashboard in your default web browser, usually at http://localhost:8501.

Running the Pipeline via CLI

You can also run specific parts of the pipeline directly from the command line:

1. Run the full detection and classification pipeline:

python root/main_pipeline.py


2. Train the Tri-Branch Network:

# Train on real cached data
python root/train_real.py

# Train on all data (including synthetic)
python root/train_all.py


3. Evaluate the Model:

# Evaluate on real benchmarking targets
python root/evaluate_real.py

# Evaluate against the unseen 200 validation set
python root/evaluate_unseen.py


📊 Data Management

MAST Downloads: The pipeline automatically stores raw TESS data inside the mastDownload/HLSP/ directory.

Caching: To speed up processing, extracted and normalized light curves are cached as .npz files in root/real_data_cache/ (e.g., TIC_127315102_S11.npz).

🧠 Pre-trained Models

Pre-trained weights for the Tri-Branch SNN ensemble are included in the repository. They are located in root/weights/. The pipeline will automatically load tri_branch_tess_net_pretrained.pt during inference if training from scratch is not specified.
