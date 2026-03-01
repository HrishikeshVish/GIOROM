<div align="center">

<div id="user-content-toc">
  <ul align="center" style="list-style: none;">
    <summary>
      <h1>GIOROM: Online Phase & ROM Benchmarking</h1>
    </summary>
  </ul>
</div>

<a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10%2B-598BE7?style=for-the-badge&logo=python&logoColor=598BE7&labelColor=F0F0F0"/></a> &emsp;
<a href="https://github.com/google/jax"><img src="https://img.shields.io/badge/JAX-Supported-9A52BA?style=for-the-badge&logo=jupyter&logoColor=9A52BA&labelColor=F0F0F0"/></a> &emsp;
<a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=EE4C2C&labelColor=F0F0F0"/></a> &emsp;
<a href="https://developer.nvidia.com/cuda-toolkit"><img src="https://img.shields.io/badge/CUDA-12.1-76B900?style=for-the-badge&logo=nvidia&logoColor=76B900&labelColor=F0F0F0"/></a>

<div id="toc">
  <ul align="center" style="list-style: none;">
    <summary>
      <h2><a href="../README.md">Back to Main</a> &emsp; <a href="https://hrishikeshvish.github.io/projects/giorom.html">Project Page</a> &emsp; <a href="https://drive.google.com/drive/folders/1CWMdqKaCtLy8KhA-DIBpS07M6CSuMjgQ?usp=sharing">Online Phase Data & Weights</a></h2>
    </summary>
  </ul>
</div>

</div>

# Overview

This directory contains the **core algorithmic contributions** of the GIOROM paper, alongside a highly modular, unified benchmarking suite for state-of-the-art Reduced Order Models (ROMs).

> [!NOTE]
> **A Note on Methodology:** In a practical, real-world deployment, the sparse inputs for the online phase would be generated iteratively by the offline model. However, for the sake of simplicity, computational efficiency, and exact reproducibility in this repository, **we sample the sparse observations directly from the ground-truth dataset** to train and evaluate the online versions.

# 💾 Data Preparation & Weights

To run the online phase models and benchmarks, you will need the pre-processed datasets and the offline-trained model weights. 

1. **Download Data & Weights:** All required datasets and offline CROM weights are hosted on our [Google Drive](https://drive.google.com/drive/folders/1CWMdqKaCtLy8KhA-DIBpS07M6CSuMjgQ?usp=sharing). 
2. **Directory Structure:** By default, our execution scripts expect the downloaded data to be extracted to a `/data` directory at the root of your system. Ensure your paths match the following structure (or update the path variables at the top of the shell scripts to point to your custom locations):
   * `H5_DATA_BASE="online/data/CROM_dataset/CROM_Ready_Data"` *(Contains the `.h5` simulation sequences for baseline benchmarking)*
   * `PT_DATA_BASE="online/data/pt_dataset"` *(Contains the `rollout_full.pt` files required for GIOROM training/eval)*
   * `OFFLINE_BASE="online/data/CROM_offline_training"` *(Contains the offline-trained CROM weights required by several baseline ROMs)*

# 📂 Architecture

We have strictly decoupled research logic from execution scripts to ensure fair, apples-to-apples comparisons across all models.

```text
online_phase/
├── checkpoints/                # Saved weights for all online models
├── results/                    # Aggregated CSV metrics (L2, Chamfer, Time, VRAM)
├── visualizations/             # Rendered Blender frames and .mp4 videos
├── src/                        # Core Logic
│   ├── giorom/                 # JAX-based GIOROM architectures
│   ├── baselines/              # PyTorch implementations (PCA, GNO, LiCROM, DINo, CORAL, CoLoRA)
│   ├── metrics/                # Shared evaluation utilities
│   ├── utils/                  # Hardware tracking and HDF5/PT data loaders
│   └── visualizations/         # Blender API rendering scripts
└── scripts/                    # Entry Points
    ├── train_giorom.py         # Main training and eval script for GIOROM
    ├── benchmark_baselines.py  # Unified evaluation loop for all PyTorch baselines
    ├── run_experiments.sh      # Master bash script for full pipeline execution
    ├── run_ablations.sh        # Bash script for grid/sparsity ablations
    └── generate_latex_table.py # Aggregates CSV metrics into a LaTeX table
```

# 🚀 Execution Guide
We provide automated bash scripts to run the entire experimental pipeline, from training to metrics aggregation and 3D rendering.

**1. Master Experiment Pipeline**
The ```run_experiments.sh``` script handles the training, benchmarking, .obj mesh generation, and Blender rendering for GIOROM and all baselines.

```Shell

chmod +x run_experiments.sh
./run_experiments.sh
```

Rendering Note: By default, the Blender rendering step (DO_RENDER) is set to false to save time. Open the script and set it to true to generate .mp4 visual comparisons.

**2. Ablation Studies**
To recreate the grid resolution and sparsity ablation charts from the paper:

```Shell

chmod +x run_ablations.sh
./run_ablations.sh
```
**3. Individual Benchmarking**
If you want to evaluate a specific model manually, you can use the unified benchmarking Python script:

```Shell

# Evaluate PCA
python scripts/benchmark_baselines.py -model_type pca -data_root /data/CROM_dataset/CROM_Ready_Data/owl

# Evaluate DINo
python scripts/benchmark_baselines.py -model_type dino -ckpt checkpoints/dino/weights.ckpt
```

# 📊 Implemented Baselines
This repository includes standardized inference pipelines for the following methods:

**PCA / Gappy POD**: Classical linear spatial basis reduction.

**GNO**: Graph Neural Operator.

**LiCROM**: Latent space interpolation via Gauss-Newton optimization.

**DINo**: Dynamics-Informed Neural Operator (Neural ODE).

**CORAL**: Hybrid Latent Dynamics.

**CoLoRA**: Continuous Low-Rank Adaptation.