# Reduced-Order Neural Operators: Learning Lagrangian Dynamics on Highly Sparse Graphs, 2024
## [Hrishikesh Viswanath](https://hrishikeshvish.github.io), [Chang Yue](https://changy1506.github.io/), [Julius Berner](https://jberner.info/), [Peter Yichen Chen](https://peterchencyc.com/), [Aniket Bera](https://www.cs.purdue.edu/homes/ab/)

#### [Arxiv](https://arxiv.org/pdf/2407.03925) | [Project Page](https://hrishikeshvish.github.io/projects/giorom.html) | [Saved Weights](https://drive.google.com/drive/folders/1CWMdqKaCtLy8KhA-DIBpS07M6CSuMjgQ?usp=sharing) | [Data](https://sites.google.com/view/learning-to-simulate)

![GIOROM Pipeline\label{pipeline}](Res/giorom_pipeline_plasticine.png)

![Python](https://img.shields.io/badge/Python-3.10-red?style=for-the-badge&logo=python)
![Pytorch](https://img.shields.io/badge/Pytorch-2.1.0-yellow?style=for-the-badge&logo=pytorch)
![Cuda](https://img.shields.io/badge/Cuda-12.1-green?style=for-the-badge&logo=nvidia)
![Torch Geometric](https://img.shields.io/badge/Torch%20Geometric-2.5.3-blue?style=for-the-badge&logo=PyG)
![Neural Operator](https://img.shields.io/badge/Neural%20Operator-0.3.0-brown?style=for-the-badge&logo=nvidia)
![Blender](https://img.shields.io/badge/Blender-4.0.0-orange?style=for-the-badge&logo=blender)

    @article{viswanath2024reduced,
      title={Reduced-Order Neural Operators: Learning Lagrangian Dynamics on Highly Sparse Graphs},
      author={Viswanath, Hrishikesh and Chang, Yue and Berner, Julius and Chen, Peter Yichen and Bera, Aniket},
      journal={arXiv preprint arXiv:2407.03925},
      year={2024}
    }



<p align="center">
  <img src="Res/neuraloperator.gif" width="25%" />

  <img src="Res/neuralfield.gif" width="25%" />
  <img src="Res/rendered.gif" width="36%" /> <br>
  
</p>
<p><em>&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;Neural Operator Inference &emsp;&emsp;&emsp;&emsp; Discretization Agnostic FOM Inference &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;Final Rendering</em></p>


# Abstract
> We present a neural operator architecture to simulate Lagrangian dynamics, such as fluid flow, granular flows, and elastoplasticity. Traditional numerical methods, such as the finite element method (FEM), suffer from long run times and large memory consumption. On the other hand, approaches based on graph neural networks are faster but still suffer from long computation times on dense graphs, which are often required for high-fidelity simulations. Our model, GIOROM or Graph Interaction Operator for Reduced-Order Modeling, learns temporal dynamics within a reduced-order setting, capturing spatial features from a highly sparse graph representation of the input and generalizing to arbitrary spatial locations during inference. The model is geometry-aware and discretization-agnostic and can generalize to different initial conditions, velocities, and geometries after training. We show that point clouds of the order of 100,000 points can be inferred from sparse graphs with $\sim$1000 points, with negligible change in computation time. We empirically evaluate our model on elastic solids, Newtonian fluids, Non-Newtonian fluids, Drucker-Prager granular flows, and von Mises elastoplasticity. On these benchmarks, our approach results in a 25$\times$ speedup compared to other neural network-based physics simulators while delivering high-fidelity predictions of complex physical systems and showing better performance on most benchmarks. The code and the demos are provided at (https://github.com/HrishikeshVish/GIOROM).

# Instructions to Use the Code

#### After downloading the repo, and from the parent directory. Install dependencies:

    conda create --name <env> --file requirements.txt

#### Download the dataset and Create a folder:

    mkdir giorom_datasets
    mkdir <datasetname>

We provide code to process datasets provided by GNS [1] and NCLAW [2]

- [1] Sanchez-Gonzales+ Learning to Simulate Complex Physics with Graph Neural Networks
- [2] Ma+ Learning neural constitutive laws from motion observations for generalizable pde dynamics

#### Dataset preprocessing:

    cd Dataset\ Parsers/
    python parseData.py --data_config nclaw_Sand
    python parseData.py --data_config WaterDrop2D

#### Train a model from the config:

    python train.py --train_config train_configs_nclaw_Water

#### Train a model with args:

    python train.py --batch_size 2 --epoch 100 --lr 0.0001 --noise 0.0003 --eval_interval 1500 --rollout_interval 1500 --sampling true --sampling_strategy fps --graph_type radius --connectivity_radius 0.032 --model giorom2d_large --dataset WaterDrop2D --load_checkpoint true --ckpt_name giorom2d_large_WaterDrop2D --dataset_rootdir giorom_datasets/

#### Evaluate a model (Untested code):

    python eval.py --eval_config train_configs_Water2D

We have not provided the code to save the rollout output as a pytorch tensor. However, this snippet can be found at ```eval_3d.ipynb```
To render the results with Polyscope or Blender Use the following. This part of the code needs to be modified and appropriate paths need to be provided in the code

    cd Viz
    python createObj.py
    python blender_rendering.py

### Neural Fields Inference: We follow the code and training strategy provided by [LiCROM](https://github.com/Changy1506/LiCROM_all) 

# Datasets
We support existing datasets provided by GNS and have new dataset curated from NCLAW framework. We shall upload the new dataset soon...

### Currently tested datasets from Learning to Simulate Complex Physics with Graph Neural Networks (GNS)

* `{DATASET_NAME}` one of the datasets following the naming used in the paper:
  * `WaterDrop`
  * `Water`
  * `Sand`
  * `Goop`
  * `MultiMaterial`
  * `Water-3D`
  * `Sand-3D`
  * `Goop-3D`
 
Datasets are available to download via:

* Metadata file with dataset information (sequence length, dimensionality, box bounds, default connectivity radius, statistics for normalization, ...):

  `https://storage.googleapis.com/learning-to-simulate-complex-physics/Datasets/{DATASET_NAME}/metadata.json`

* TFRecords containing data for all trajectories (particle types, positions, global context, ...):

  `https://storage.googleapis.com/learning-to-simulate-complex-physics/Datasets/{DATASET_NAME}/{DATASET_SPLIT}.tfrecord`

Where:

* `{DATASET_SPLIT}` is one of:
  * `train`
  * `valid`
  * `test`

##### Note: We have included the code to convert tfrecord to pytorch tensor in our repository, within ```Dataset Parsers```

# Discretization Invariance and Generalizability

![Discretization Invariance](Res/disc_invar.png)
![Generalizability](Res/3d_sims.png)

# How GIOROM is different from GNS and GINO (Neural Operator for 3D PDEs)
![Diff Archs](Res/diff_archs.png)

# Different Graph Construction methods and Sampling Strategies supported in the codebase
![Graphs](Res/graph_techniques.png)
