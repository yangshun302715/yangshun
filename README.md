# BiTCN-RGM-ACLPSO: Time Series Forecasting Model

[!\[Python](https://img.shields.io/badge/Python-3.x-blue)](https://www.python.org/)
[!\[PyTorch](https://img.shields.io/badge/PyTorch-2.6.0-EE4C2C)](https://pytorch.org/)

**Citation**: Yang et al., Physics-Plausible Probabilistic Photovoltaic Power Forecasting Using an Optimally Regulated Gaussian Mixture Deep Framework,2026 (in sumitting) (Contaction: ys302715@163.com）

This repository contains the official implementation of the **BiTCN-RGM-ACLPSO** framework, a novel deep learning approach designed for **multi-step time series forecasting**. The model specifically targets the complex characteristics of Photovoltaic (PV) power generation, offering high-precision probabilistic predictions.

## Core Innovations

This framework integrates three key components to enhance forecasting robustness and accuracy:

* **BiTCN Architecture:** A Bidirectional Temporal Convolutional Network that captures comprehensive temporal features from both past and future contexts relative to the prediction point.
* **RGM (Regularized Gaussian Mixture) Loss:** A physics-plausible loss function that models the output as a mixture of Gaussians with regularization, ensuring stable probabilistic forecasting.
* **ACLPSO Optimization:** An Adaptive Chaotic Levy Particle Swarm Optimization algorithm is employed for automatic hyperparameter tuning, maximizing model performance without manual grid search.

## Project Structure

The core logic is contained within the `model code` directory. Ensure your local file structure matches the following:

```
└── model code/
    ├── main.py           # Entry point: Training and inference pipeline
    ├── loss.py           # Implementation of the RGM Loss function
    ├── aclpso.py         # Implementation of the ACLPSO optimization algorithm
    └── data.csv          # Dataset for PV power forecasting
```

## Dependencies \& Installation

The project requires **PyTorch 2.6.0 (CUDA 12.4)**. Please ensure your environment matches the following specifications.

You can install all dependencies by running the following commands in your terminal:

```bash
# Install PyTorch and core dependencies
pip install torch==2.6.0+cu124 pytorch-lightning==2.5.2 torchmetrics==1.7.3

# Install data processing and utility libraries
pip install utilsforecast==0.2.12 coreforecast==0.0.16 pandas==2.2.3 numpy==1.26.4

# Install visualization and ML tools
pip install matplotlib==3.7.2 scikit-learn==1.2.2 properscoring==0.1

# Install NeuralForecast
pip install neuralforecast==3.0.2
```

## How to Run

1. **Data Preparation:**
Place your time-series data in `model code/data.csv`. Ensure the file contains the necessary features (e.g., historical power, meteorological data).
2. **Configuration:**
Open `main.py` to configure the random seed (fixed to `42` for reproducibility) and other hyperparameters if needed.
3. **Execution:**
Navigate to the project directory and run the main script:

```bash
   cd "model code"
   python main.py
   ```

## Experimental Setup

* **Random Seed:** `42` (Fixed to ensure reproducible results)
* **Task:** Multi-step Probabilistic Forecasting
* **Hardware:** Recommended to run on a GPU with CUDA  GPU P100 support for optimal training speed

