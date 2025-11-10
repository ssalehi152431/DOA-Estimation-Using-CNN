# Covariance Matrix-Based DOA Estimation Using CNN
**Performance Across Narrow and Wide Grids**

## Overview
This repository contains the implementation of the research paper **“Covariance Matrix-Based DOA Estimation Using CNN: Performance Across Narrow and Wide Grids”**, presented at *IEEE DevIC 2025*.  
The work demonstrates a deep learning approach to **Direction of Arrival (DOA)** estimation using **Convolutional Neural Networks (CNNs)** trained on synthetic covariance matrix data.

The CNN-based method provides improved accuracy and robustness over classical subspace algorithms such as MUSIC and ESPRIT, across both **narrow (-60° to 60°)** and **wide (-90° to 90°)** grids under varying noise levels.

---

## Key Highlights
- CNN learns DOA patterns from signal covariance matrices.
- Works under low SNR conditions (up to -20 dB).
- Narrow grid RMSE ≈ 3.5°, Wide grid RMSE ≈ 24°.
- Low inference latency (3.2 ms).
- Full dataset generation and model training code provided.

---

## Repository Structure
```

├── dataset_generator.py         # Dataset creation script
├── DatasetGeneration.ipynb      # Notebook for dataset generation and visualization
├── DOAEstimation.ipynb          # CNN training and evaluation notebook
├── narrow_grid_dataset.mat      # Can be generated using script-Narrow grid covariance matrix dataset (-60° to 60°)
├── wide_grid_dataset.mat        # Can be generated using script- Wide grid covariance matrix dataset (-90° to 90°)
├── Results/                     # Output plots and numerical results
└── README.md                    # This documentation file
```

---

## Methodology

### Dataset Generation
- **Array:** 16-element Uniform Linear Array (ULA)
- **Angle Range:** Narrow (-60° to 60°) and Wide (-90° to 90°)
- **Grid Resolution:** 1°
- **SNR Values:** 0, -5, -10, -15, -20 dB
- **Samples:** ~36k (narrow), ~81k (wide)
- **Input Tensor Shape:** 16×16×3 (real, imaginary, phase channels)


### Model Architecture
- **Input:** 16×16×3 covariance matrices
- **Conv Layers:** 4 layers with 256 filters (3×3 kernel)
- **Dense Layers:** 4096 → 2048 → 1024 → output grid
- **Activation:** ReLU (hidden), Sigmoid (output)
- **Regularization:** BatchNorm + Dropout (0.3)
- **Optimizer:** Adam
- **Loss:** Binary Cross-Entropy

---

## Training Setup
| Parameter | Value |
|------------|-------|
| Framework | TensorFlow |
| Epochs | 200 |
| Batch Size | 128 |
| Learning Rate | Adaptive (Adam) |
| Split | 90% train / 10% validation |
| Hardware | GPU (CUDA Recommended) |

---

## Results

### Narrow Grid (-60° to 60°)
| SNR (dB) | RMSE (°) | Accuracy (%) |
|-----------|-----------|--------------|
| 0 | 1.2 | 98.7 |
| -5 | 1.8 | 96.4 |
| -10 | 2.6 | 94.9 |
| -15 | 3.2 | 93.1 |
| -20 | 3.4 | 94.13 |

### Wide Grid (-90° to 90°)
| SNR (dB) | RMSE (°) | Accuracy (%) |
|-----------|-----------|--------------|
| 0 | 7.2 | 91.3 |
| -5 | 9.8 | 87.2 |
| -10 | 15.3 | 84.6 |
| -15 | 20.1 | 82.9 |
| -20 | 24.0 | 82.45 |

### Comparison with Traditional Methods
| Method | RMSE (°) | Inference Time (ms) |
|:--------|:----------:|:-------------------:|
| MUSIC | 5.3 | 5.3 |
| ESPRIT | 4.7 | 4.7 |
| **CNN (Proposed)** | **3.5 / 24** | **3.2** |

---

## Usage

### 1. Dataset Generation
```bash
python dataset_generator.py
```

### 2. Model Training
```bash
jupyter notebook DOAEstimation.ipynb
```

### 3. Evaluation and Visualization
Run the final notebook cells to view RMSE vs. SNR plots and estimated DOA distributions.

---

## Findings
- The CNN model outperforms MUSIC and ESPRIT in low-SNR conditions.
- Incorporating phase and magnitude data enhances angular precision.
- The model generalizes effectively across both narrow and wide angular grids.
- Efficient for real-time applications on GPU or embedded platforms.

---

## Citation
If you use this repository, please cite:

> @inproceedings{salehin2025covariance,
  title={Covariance Matrix-Based DOA Estimation Using CNN: Performance Across Narrow and Wide Grids},
  author={Salehin, Sultanus and Islam, Kamrul and Islam, Akib Jayed and Barua, Nirzar and Ananna, Kaniz F and Pavel, TA and Uddin, A and Noor-A-Alahee, SAM},
  booktitle={2025 Devices for Integrated Circuit (DevIC)},
  pages={313--318},
  year={2025},
  organization={IEEE}
> }


> S. Salehin, K. Islam, A. J. Islam, N. Barua, K. F. Ananna, T. A. Pavel, A. Uddin, and S. A. M. Noor-A-Alahee,  
> *“Covariance Matrix-Based DOA Estimation Using CNN: Performance Across Narrow and Wide Grids,”*  
> 2025 IEEE Devices for Integrated Circuit (DevIC), Kalyani, India, pp. 313–318, Apr. 2025.

---

## Contact
**Sultanus Salehin**  
Email: [salehin.iut@gmail.com](mailto:salehin.iut@gmail.com)  
GitHub: [https://github.com/ssalehi152431](https://github.com/ssalehi152431)

