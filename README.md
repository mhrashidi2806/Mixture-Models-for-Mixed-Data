# Overview
This project implements and experiments with a mixture model for mixed data, specifically applied to an echocardiogram (ECG) dataset. The dataset consists of ECG waveforms (x) and binary labels (y), which indicate the presence of various cardiac arrhythmias. Each data instance corresponds to one heartbeat.

The model is trained using mini-batch negative log marginal likelihood minimization with PyTorch and automatic differentiation. The implementation involves parameter re-parameterization to ensure proper constraints during training.

# Key Features
Mixture Model for Mixed Data: Handles both continuous (ECG waveforms) and categorical (arrhythmia labels) data.

### Re-parameterization Techniques: 
#### Ensures model parameters remain within valid ranges using appropriate transformations:

Softmax for mixture weights (πz)

Softplus for standard deviations (σdz)

Sigmoid for Bernoulli parameters (φdz)


Mini-Batch Training: Uses stochastic optimization to minimize the negative log marginal likelihood efficiently.
PyTorch Automatic Differentiation: Enables efficient gradient-based optimization.

# Requirements
Python 3.x
PyTorch
NumPy
Matplotlib (for visualization)
