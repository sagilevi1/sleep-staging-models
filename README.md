# On Improving PPG-Based Sleep Staging: A Pilot Study

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official implementation of "On Improving PPG-Based Sleep Staging: A Pilot Study" - exploring dual-stream cross-attention architectures for enhanced sleep stage classification using PPG signals.
>  This work has been published as a workshop paper to the *Companion of the 2025 ACM International Joint Conference on Pervasive and Ubiquitous Computing (UbiComp/ISWC 2025)*.

## 📋 Abstract

This repository contains the implementation of our pilot study on improving PPG-based sleep staging through dual-stream cross-attention architectures. We demonstrate that substantial performance gains can be achieved by combining PPG with auxiliary modalities (Augmented PPG, Synthetic ECG, Real ECG) under a dual-stream cross-attention framework, achieving up to **83.3% accuracy** and **0.745 Cohen's kappa** on the MESA dataset.

## 🔑 Key Findings

- **PPG + Augmented PPG** achieves the best performance (κ=0.745, Acc=83.3%), improving accuracy by 5% over single-stream baseline
- Cross-attention mechanism effectively extracts complementary information from signal variations
- Augmented PPG strategy performs comparably to PPG + Real ECG, while being more practical (no additional sensors needed)
- Synthetic ECG shows promise but requires sleep-specific training for optimal performance

## 🏗️ Architecture Overview

### Single-Stream Model (Baseline)
- **SleepPPG-Net**: Processes 10-hour PPG recordings through residual convolutional blocks
- We have replicated the code of SleepPPG-Net
<div align="center">
  <img src="https://raw.githubusercontent.com/DavyWJW/sleep-staging-models/main/single-ppg.jpg" alt="Single-Stream Architecture" width="300"/>
</div>

### Dual-Stream Models

1. **PPG + Augmented PPG**: Combines PPG and noise-augmented PPG signals

2. **PPG + Synthetic ECG**: Uses RDDM-generated ECG as auxiliary modality
  
3. **PPG + Real ECG**: Upper bound performance using actual ECG recordings

<div align="center">
  <img src="https://raw.githubusercontent.com/DavyWJW/sleep-staging-models/main/dual.jpg" alt="Dual-Stream Architecture" width="260"/>
</div>

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/DavyWJW/sleep-staging-models.git
cd sleep-staging-models

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python ≥ 3.8
- PyTorch ≥ 1.9.0
- NumPy ≥ 1.19.0
- scikit-learn ≥ 0.24.0
- h5py ≥ 3.0.0
- See `requirements.txt` for complete list

### Data Preparation

1. Download the <a target="_blank" href="https://sleepdata.org/datasets/mesa/files/polysomnography" data-view-component="true" class="Link mr-2">polysomnography</a> data of MESA Sleep Study dataset.

2. Data Processing: Extract PPG and ECG data from MESA data.
```bash
python extract_mesa_data.py 
```

## 🏃‍♂️ Training

### Train Single-Stream Baseline (SleepPPG-Net)

```bash
python train_ppg_only.py --config configs/config_cloud.yaml --model ppg_only --runs 5
```

### Train Dual-Stream Models

```bash
# PPG + Augmented PPG (Best Performance)
python train_ppg_unfiltered.py --config configs/config_ppg_unfiltered.yaml --runs 5

# PPG + Synthetic ECG
python train_crossattn_gen.py --config configs/config_crossattn_generated.yaml --model_type generated_ecg --runs 5

# PPG + Real ECG
python train_crossattn.py --config configs/config_crossattn_v2.yaml --runs 5
```

### Multi-GPU Training (DDP)

```bash
python train_crossattn_gen.py --config configs/config_crossattn_generated.yaml --gpus 3 --runs 5
```

## 📝 Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{wang2025improving,
  title={On Improving PPG-Based Sleep Staging: A Pilot Study},
  author={Wang, Jiawei and Guan, Yu and Chen, Chen and Zhou, Ligang and Yang, Laurence T and Gu, Sai},
  booktitle={Companion of the 2025 ACM International Joint Conference on Pervasive and Ubiquitous Computing},
  pages={1640--1644},
  year={2025}
}
```
Please read our paper on arxiv here: https://arxiv.org/abs/2508.02689

Further Reading:

SleepPPG-Net: A deep learning algorithm for robust sleep staging from continuous photoplethysmography
```bibtex
@article{kotzen2022sleepppg,
  title={SleepPPG-Net: A deep learning algorithm for robust sleep staging from continuous photoplethysmography},
  author={Kotzen, Kevin and Charlton, Peter H and Salabi, Sharon and Amar, Lea and Landesberg, Amir and Behar, Joachim A},
  journal={IEEE Journal of Biomedical and Health Informatics},
  volume={27},
  number={2},
  pages={924--932},
  year={2022},
  publisher={IEEE}
}
```

RDDM: Region-disentangled diffusion model for high-fidelity ppg-to-ecg translation
```bibtex
@inproceedings{shome2024region,
  title={Region-disentangled diffusion model for high-fidelity ppg-to-ecg translation},
  author={Shome, Debaditya and Sarkar, Pritam and Etemad, Ali},
  booktitle={Proceedings of the AAAI conference on artificial intelligence},
  volume={38},
  number={13},
  pages={15009--15019},
  year={2024}
}
```

## 🤝 Acknowledgments

- MESA Sleep Study dataset 
- SleepPPG-Net baseline architecture 
- RDDM for synthetic ECG generation 

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


<p align="center">
<i>Advancing accessible sleep monitoring through innovative computational approaches</i>
</p>
