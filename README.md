<div align="center">

<div id="user-content-toc">
  <ul align="center" style="list-style: none;">
    <summary>
      <h1>Learning Lagrangian Interaction Dynamics with Sampling-Based Model Order Reduction</h1>
    </summary>
  </ul>
</div>

<a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10%2B-598BE7?style=for-the-badge&logo=python&logoColor=598BE7&labelColor=F0F0F0"/></a> &emsp;
<a href="https://github.com/google/jax"><img src="https://img.shields.io/badge/JAX-Supported-9A52BA?style=for-the-badge&logo=jupyter&logoColor=9A52BA&labelColor=F0F0F0"/></a> &emsp;
<a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=EE4C2C&labelColor=F0F0F0"/></a> &emsp;
<a href="https://developer.nvidia.com/cuda-toolkit"><img src="https://img.shields.io/badge/CUDA-12.1-76B900?style=for-the-badge&logo=nvidia&logoColor=76B900&labelColor=F0F0F0"/></a>

<br><br>
<img src="Res/giorom.png" width="850px" alt="GIOROM Pipeline"/>
<br><br>

<div id="toc">
  <ul align="center" style="list-style: none;">
    <summary>
      <h2><a href="https://arxiv.org/pdf/2407.03925">Paper</a> &emsp; <a href="https://hrishikeshvish.github.io/projects/giorom.html">Project Page</a> &emsp; <a href="https://drive.google.com/drive/folders/1CWMdqKaCtLy8KhA-DIBpS07M6CSuMjgQ?usp=sharing">Weights</a> &emsp; <a href="https://sites.google.com/view/learning-to-simulate">Data</a></h2>
    </summary>
  </ul>
</div>

</div>

# Overview

**GIOROM** Simulating physical systems governed by Lagrangian dynamics often entails solving partial differential equations (PDEs) over high-resolution spatial domains, leading to significant computational expense. Reduced-order modeling (ROM) mitigates this cost by evolving low-dimensional latent representations of the underlying system. While neural ROMs enable querying solutions from latent states at arbitrary spatial points, their latent states typically represent the global domain and struggle to capture localized, highly dynamic behaviors such as fluids. We propose a sampling-based reduction framework that evolves Lagrangian systems directly in physical space, over the particles themselves, reducing the number of active degrees of freedom via data-driven neural PDE operators. To enable querying at arbitrary spatial locations, we introduce a learnable kernel parameterization that uses local spatial information from time-evolved sample particles to infer the underlying solution manifold. Empirically, our approach achieves a 6.6–32x
 reduction in input dimensionality while maintaining high-fidelity evaluations across diverse Lagrangian regimes, including fluid flows, granular media, and elastoplastic dynamics. We refer to this framework as GIOROM (\textbf{G}eometry-\textbf{I}nf\textbf{O}rmed \textbf{R}educed-\textbf{O}rder \textbf{M}odeling).

### Features

- **25× Speedup**: Vastly outperforms existing neural network-based physics simulators while delivering high-fidelity predictions.
- **Massive Upsampling**: Infers dense point clouds of ~100,000 points from highly sparse sensor graphs of ~1,000 points with negligible computational overhead. 
- **Discretization-Agnostic**: Geometry-aware architecture that seamlessly generalizes to new initial conditions, velocities, and unseen geometries post-training.
- **5 Complex Physical Systems**: Empirically validated on elastic solids, Newtonian fluids, Non-Newtonian fluids, Drucker-Prager granular flows, and von Mises elastoplasticity.
- **Comprehensive ROM Benchmark Suite**: Includes clean, well-tuned, modular implementations of 6 state-of-the-art baseline Reduced Order Models (PCA, GNO, LiCROM, DINo, CORAL, and CoLoRA).
- **Fully Automated Pipelines**: Single-command execution for model training, metrics aggregation (Chamfer, Rel L2, VRAM), and professional Blender 3D video rendering.

# Repository Structure

To ensure modularity and clean benchmarking, this repository is divided into two distinct phases. **Our core algorithmic contributions lie in the Online Phase.**

> [!NOTE]
> **A Note on Methodology:** In a practical deployment, the sparse inputs for the online phase are generated iteratively by a slow, high-fidelity offline model. However, to ensure computational efficiency and exact reproducibility for this benchmark, we sample the sparse observations directly from the ground-truth offline datasets to train and evaluate the online ROMs.

### 1. `online/` (Core Contribution & Benchmarks)
Contains the JAX-based GIOROM architecture alongside PyTorch implementations of all baseline ROMs. It includes a unified evaluation engine that guarantees fair, apples-to-apples comparisons across all models regarding inference time, memory footprint, and reconstruction accuracy. See the [`online_phase/README.md`](./online_phase/README.md) for full execution details.

### 2. `offline/` (Data Prep & Baselines)
Contains the code for processing raw simulation data (GNS / NCLAW) and training the offline full-order and time-stepper models. While we provide our specific offline training code for reproducibility, **GIOROM is strictly agnostic to the offline model**. Any suitable high-fidelity physics simulator can be used to generate the prior trajectories. See the [`offline_phase/README.md`](./offline_phase/README.md) for details.

# Quick Start

### Installation

We recommend using Conda to manage dependencies. GIOROM utilizes both JAX (for the core online method) and PyTorch (for the offline model and baseline benchmark suite).

```shell
# Clone the repository
git clone [https://github.com/HrishikeshVish/GIOROM.git](https://github.com/HrishikeshVish/GIOROM.git)
cd GIOROM

# Create the environment
conda create --name giorom_env python=3.10
conda activate giorom_env

# Install dependencies
pip install -r requirements.txt
```

### Data & Weights
All necessary datasets (NCLAW/GNS formats) and pre-trained offline weights for baseline ROM models are available on our Google Drive.

For the automated scripts to work out-of-the-box, extract the downloaded data into a /data/ directory. The master execution script expects the following path variables (which you can modify at the top of run_experiments.sh):

```bash
H5_DATA_BASE="/data/CROM_dataset/CROM_Ready_Data"

PT_DATA_BASE="/data/pt_dataset"

OFFLINE_BASE="/data/CROM_offline_training"
```

Running the Online Benchmarks
Navigate to the online_phase directory to run the master experimental pipeline. This script trains the baselines, evaluates GIOROM, aggregates all metrics into CSVs, and optionally stitches 3D Blender renders into .mp4 comparison videos.

```Shell

cd online_phase
chmod +x run_experiments.sh

# Run the master sweep across all models and datasets
./run_experiments.sh
```

To run ablation studies on GIOROM's grid resolution and sparsity tolerances:

```Shell

chmod +x run_ablations.sh
./run_ablations.sh
```
#### Citation
If you find this codebase or benchmark suite useful in your research, please consider citing our paper:

Code snippet

@article{viswanath2024reduced,
  title={Reduced-Order Neural Operators: Learning Lagrangian Dynamics on Highly Sparse Graphs},
  author={Viswanath, Hrishikesh and Chang, Yue and Berner, Julius and Chen, Peter Yichen and Bera, Aniket},
  journal={arXiv preprint arXiv:2407.03925},
  year={2024}
}