# Reduced-Order Neural Operators: Learning Lagrangian Dynamics on Highly Sparse Graphs, 2024
## [Hrishikesh Viswanath](https://hrishikeshvish.github.io), [Chang Yue](https://changy1506.github.io/), [Julius Berner](https://jberner.info/), [Peter Yichen Chen](https://peterchencyc.com/), [Aniket Bera](https://www.cs.purdue.edu/homes/ab/)

#### [Arxiv](https://arxiv.org/pdf/2407.03925) | [Project Page](https://hrishikeshvish.github.io/assets/projects/giorom.html) | [Medium]() | [Saved Weights]() | [Data]()

![GIOROM Pipeline\label{pipeline}](Res/giorom_pipeline_plasticine.png)

![Python](https://img.shields.io/badge/Python-3.8,%203.10-red?style=for-the-badge&logo=python)
![Pytorch](https://img.shields.io/badge/Pytorch-2.1.0-yellow?style=for-the-badge&logo=pytorch)
![Cuda](https://img.shields.io/badge/Cuda-12.1-green?style=for-the-badge&logo=nvidia)
![Torch Geometric](https://img.shields.io/badge/Torch%20Geometric-2.5.3-blue?style=for-the-badge&logo=PyG)
![Neural Operator](https://img.shields.io/badge/Neural%20Operator-0.3.0-brown?style=for-the-badge&logo=nvidia)
![Blender](https://img.shields.io/badge/Blender-4.0.0-orange?style=for-the-badge&logo=blender)

> We present a neural operator architecture to simulate Lagrangian dynamics, such as fluid flow, granular flows, and elastoplasticity. Traditional numerical methods, such as the finite element method (FEM), suffer from long run times and large memory consumption. On the other hand, approaches based on graph neural networks are faster but still suffer from long computation times on dense graphs, which are often required for high-fidelity simulations. Our model, GIOROM or Graph Interaction Operator for Reduced-Order Modeling, learns temporal dynamics within a reduced-order setting, capturing spatial features from a highly sparse graph representation of the input and generalizing to arbitrary spatial locations during inference. The model is geometry-aware and discretization-agnostic and can generalize to different initial conditions, velocities, and geometries after training. We show that point clouds of the order of 100,000 points can be inferred from sparse graphs with $\sim$1000 points, with negligible change in computation time. We empirically evaluate our model on elastic solids, Newtonian fluids, Non-Newtonian fluids, Drucker-Prager granular flows, and von Mises elastoplasticity. On these benchmarks, our approach results in a 25$\times$ speedup compared to other neural network-based physics simulators while delivering high-fidelity predictions of complex physical systems and showing better performance on most benchmarks. The code and the demos are provided at [here](https://github.com/HrishikeshVish/GIOROM).




