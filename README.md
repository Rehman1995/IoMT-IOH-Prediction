# 🩺 IoMT-IOH-Prediction

## Early Prediction of Intraoperative Hypotension Using Multi-Scale Temporal Deep Learning and Continuous MAP Signals

<p align="center">

🚑 Early IOH Prediction • 📈 Time-Series Learning • 🏥 Smart Operating Rooms • 🌐 IoMT-Enabled Healthcare

</p>

<p align="center">
  <img src="figures/Picture1_new.png" width="100%">
</p>

<p align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)
![Smart Healthcare](https://img.shields.io/badge/Domain-Smart%20Healthcare-success)
![IoMT](https://img.shields.io/badge/IoMT-Enabled-purple)
![Time Series](https://img.shields.io/badge/Task-Time%20Series%20Forecasting-orange)
![Status](https://img.shields.io/badge/Status-Research-green)
![License](https://img.shields.io/badge/License-MIT-purple)

</p>

---

# 📖 Overview

Intraoperative Hypotension (IOH) is a common yet critical perioperative complication associated with:

- Acute Kidney Injury (AKI)
- Myocardial Injury
- Organ Hypoperfusion
- Increased Postoperative Morbidity and Mortality
- Extended Hospital Stay

Current monitoring systems are largely reactive and identify hypotension only after it occurs.

This repository presents **IoMT-IOH-Prediction**, an **IoMT-enabled Multi-Scale Temporal Deep Learning Framework** capable of forecasting intraoperative hypotension **5 minutes before onset** using only continuously monitored Mean Arterial Pressure (MAP) signals.

The framework integrates:

✅ Leakage-aware temporal segmentation

✅ Multi-scale temporal convolutions

✅ Residual temporal learning

✅ Dilated temporal modeling

✅ Temporal attention pooling

✅ IoMT-oriented clinical decision support

---

# 🏥 IoMT-Enabled Monitoring Framework

The proposed framework bridges continuous physiological monitoring with intelligent clinical decision support.

<p align="center">
  <img src="figures/Picture1_new.png" width="100%">
</p>

## Framework Components

### 1️⃣ Intraoperative Environment (IoMT Sensing Layer)

- Continuous MAP acquisition
- Arterial pressure transducer monitoring
- Bedside physiological monitoring
- IoMT-enabled sensing infrastructure

### 2️⃣ Temporal Data Processing

- Leakage-aware temporal segmentation
- Case-level dataset splitting
- Observation, prediction, and labeling windows

### 3️⃣ Intelligent Analytics Layer

- MAP feature augmentation
- Multi-scale temporal convolutions
- Residual refinement
- Dilated temporal modeling
- Temporal attention pooling

### 4️⃣ Clinical Decision Support

- IOH risk estimation
- Early warning alerts
- Smart operating room dashboards
- Clinician-assisted intervention

---

# ⏱ Temporal Segmentation Strategy

Each training sample follows a strict leakage-aware temporal structure.

<p align="center">
  <img src="figures/MAP.png" width="90%">
</p>

## Window Configuration

| Segment | Duration |
|----------|----------|
| Observation Window | 20 Minutes |
| Prediction Horizon | 5 Minutes |
| Future Label Window | 10 Minutes |

### Hypotension Definition

```text
MAP < 65 mmHg
for ≥ 50% of the future labeling window
```

This design ensures complete temporal separation between model inputs and outcome labels, preventing information leakage.

---

# 🧠 Proposed MAP-TCN Architecture

The proposed model learns hierarchical temporal representations directly from continuous MAP signals.

```text
Continuous MAP Signal
          │
          ▼
Feature Augmentation
(MAP + ΔMAP)
          │
          ▼
Multi-Scale Convolutional Blocks
(k = 3, 5, 9)
          │
          ▼
Residual Refinement
          │
          ▼
Dilated Temporal Modeling
          │
          ▼
Temporal Attention Pooling
          │
          ▼
Dense Layers
          │
          ▼
Predicted IOH Risk
```

## Architectural Highlights

- Multi-scale temporal feature extraction
- Long-range dependency modeling
- Attention-guided temporal aggregation
- Leakage-aware evaluation protocol
- Clinically interpretable predictions

---

# 📊 Experimental Results

## Performance Comparison

| Model | AUROC | AUPRC | Accuracy |
|---------|---------|---------|---------|
| Logistic Regression | 0.8342 | 0.8227 | 0.7363 |
| Random Forest | 0.8493 | 0.8137 | 0.6167 |
| XGBoost | 0.8119 | 0.7917 | 0.6391 |
| CNN | 0.8179 | 0.8181 | 0.7236 |
| GRU | 0.9002 | 0.8918 | 0.8314 |
| CNN + LSTM | 0.9130 | 0.8999 | 0.8258 |
| CNN + GRU | 0.9054 | 0.8917 | 0.8126 |
| CNN + Attention | 0.8764 | 0.8462 | 0.8155 |
| Transformer Encoder | 0.8477 | 0.7830 | 0.7775 |
| **MAP-TCN (Proposed)** | **0.9367** | **0.9284** | **0.8559** |

---

# 📈 Sensitivity vs. Specificity Trade-Off

<p align="center">
  <img src="figures/Trade-off.png" width="75%">
</p>

The proposed MAP-TCN achieves the highest sensitivity while maintaining clinically meaningful specificity, making it suitable for proactive perioperative monitoring.

---

# 🔍 Risk Stratification Analysis

The learned representations provide strong separation between hypotensive and non-hypotensive cases.

<p align="center">
  <img src="figures/Distribution.png" width="70%">
</p>

### Key Observations

- Clear class separation
- Minimal overlap near decision thresholds
- Robust risk calibration
- Effective early-warning capability

---

# 🧬 Multi-Scale Feature Learning

To better understand the learned representations, feature activations from each temporal scale are visualized.

<p align="center">
  <img src="figures/Multiscale_feature.png" width="100%">
</p>

### Interpretation

#### Kernel = 3

- Captures abrupt MAP fluctuations
- Learns short-term physiological dynamics

#### Kernel = 5

- Captures intermediate temporal transitions
- Learns evolving hypotensive trends

#### Kernel = 9

- Captures long-range hemodynamic patterns
- Learns sustained physiological deterioration

Together, these scales enable robust anticipation of impending hypotension.

---

# 📂 Dataset

## VitalDB

This study utilizes the publicly available VitalDB database:

🔗 https://vitaldb.net

### Inclusion Criteria

- Adult patients (>18 years)
- Surgery duration ≥ 1 hour
- Valid invasive arterial pressure recordings

### Dataset Summary

| Metric | Value |
|----------|----------|
| Cases | 94 |
| Total Segments | 14,518 |
| Sampling Interval | 2 seconds |
| Sampling Frequency | 0.5 Hz |
| Signal Type | Mean Arterial Pressure (MAP) |

---

# 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/Rehman1995/IoMT-IOH-Prediction.git

cd IoMT-IOH-Prediction
```

Create environment and install dependencies:

```bash
pip install -r requirements.txt
```

---

# 🏃 Training

Train the proposed MAP-TCN model:

```bash
python train.py
```

---

# 🧪 Evaluation

Evaluate on the independent test set:

```bash
python evaluate.py
```

---

# 📁 Repository Structure

```text
IoMT-IOH-Prediction
│
├── figures/
│   ├── Picture1_new.png
│   ├── MAP.png
│   ├── Distribution.png
│   ├── Multiscale_feature.png
│   └── Trade-off.png
│
├── data/
│
├── preprocessing/
│   ├── segmentation.py
│   ├── augmentation.py
│
├── models/
│   ├── multiscale_tcn.py
│   ├── attention.py
│
├── training/
│   ├── train.py
│
├── evaluation/
│   ├── evaluate.py
│
├── requirements.txt
│
└── README.md
```

---

# 💡 Key Contributions

### Methodological Contributions

- Leakage-aware temporal segmentation framework
- Multi-scale temporal convolutional architecture
- Attention-guided temporal aggregation
- MAP-only prediction strategy
- IoMT-oriented deployment architecture

### Clinical Contributions

- Early prediction of intraoperative hypotension
- Improved perioperative risk assessment
- Clinician-centered decision support
- Smart operating room integration

---

# 🔮 Future Directions

- Multi-modal physiological monitoring
- Federated healthcare learning
- Edge-AI deployment
- Real-time hospital integration
- Prospective clinical validation
- Personalized hypotension prediction

---

# 📚 Citation

If you find this work useful, please cite:

```bibtex
@article{abbas2025iomtioh,
  title={An IoT-Enabled Multi-Scale Temporal Deep Learning Framework for Early Prediction of Intraoperative Hypotension Using Continuous Arterial Pressure Signals},
  author={Abbas, Zeeshan and Abbas, Zeeshan and Rehman, Mobeen Ur},
  journal={Internet of Things},
  year={2025}
}
```

---

# 🤝 Acknowledgements

- VitalDB Project
- United Arab Emirates University
- Smart Healthcare & IoMT Research Community

---

# ⭐ Support This Work

If this repository helps your research:

⭐ Star the repository

🍴 Fork the repository

📖 Cite our work

🤝 Contribute improvements

---

## 🔗 Repository

https://github.com/Rehman1995/IoMT-IOH-Prediction

---

<p align="center">

### Smart Healthcare • IoMT • Deep Learning • Time-Series Forecasting

</p>
