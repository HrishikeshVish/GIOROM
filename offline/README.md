<div align="center">

<div id="user-content-toc">
  <ul align="center" style="list-style: none;">
    <summary>
      <h1>GIOROM: Offline Phase & Data Preparation</h1>
    </summary>
  </ul>
</div>

<a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10%2B-598BE7?style=for-the-badge&logo=python&logoColor=598BE7&labelColor=F0F0F0"/></a> &emsp;
<a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=EE4C2C&labelColor=F0F0F0"/></a> &emsp;
<a href="https://developer.nvidia.com/cuda-toolkit"><img src="https://img.shields.io/badge/CUDA-12.1-76B900?style=for-the-badge&logo=nvidia&logoColor=76B900&labelColor=F0F0F0"/></a> &emsp;
<a href="https://huggingface.co/datasets/hrishivish23/MPM-Verse-MaterialSim-Small"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Datasets-F8D521?style=for-the-badge&labelColor=F0F0F0"/></a>

<div id="toc">
  <ul align="center" style="list-style: none;">
    <summary>
      <h2><a href="../README.md">Back to Main</a> &emsp; <a href="https://hrishikeshvish.github.io/projects/giorom.html">Project Page</a> &emsp; <a href="https://drive.google.com/drive/folders/1CWMdqKaCtLy8KhA-DIBpS07M6CSuMjgQ?usp=sharing">Data & Weights</a></h2>
    </summary>
  </ul>
</div>

</div>

# Overview

This directory contains the code for processing raw simulation data and training the offline full-order and time-stepper models.

> [!NOTE]
> **Disclaimer:** The core contribution of our paper is the **Online Phase** algorithm. While we provide this offline training code for reproducibility, our pipeline is highly flexible. In practice, *any* high-fidelity physics simulator or neural network can be utilized for the offline phase to generate the foundational trajectories.


#### ⚙️ Setup & Installation

From the parent directory, install the required dependencies:

```bash
conda create --name giorom_env --file requirements.txt
conda activate giorom_env
```

#### 📂 Datasets
We support existing datasets provided by GNS and new datasets curated from the NCLAW framework. You can either download our pre-processed datasets directly from Hugging Face or process them manually.

**Option 1: Download from Hugging Face (Recommended)**
We host our ready-to-use datasets on Hugging Face at hrishivish23/MPM-Verse-MaterialSim-Small. You can use the provided Python script to selectively download specific materials or fetch the entire suite.

Usage Examples:

```bash

# Download a specific dataset
python download_hf_data.py -s Water2D
```

# Download all available datasets
```bash
python download_hf_data.py -s all
```

**Option 2: Manual Directory Setup & Preprocessing**
If you are generating your own data or bypassing Hugging Face, create a directory for your datasets manually:

```bash
mkdir giorom_datasets
mkdir giorom_datasets/<datasetname>
```

For each dataset, there should be 4 files within the dataset directory. In some cases there's a fifth file rollout_full, which is the full pointcloud (not sampled). Typically, rollout.pt contains sampled points for space efficiency. For small datasets such as WaterDrop2D, the full point cloud has ~2000 points and does not contain a rollout_full.pt. For example, nclaw_Water would look something like this

    giorom_datasets/nclaw_Water/metadata.json
    giorom_datasets/nclaw_Water/rollout_full.pt
    giorom_datasets/nclaw_Water/test.pt
    giorom_datasets/nclaw_Water/train.obj
    giorom_datasets/nclaw_Water/rollout.pt

We provide code to process datasets provided by GNS [1] and NCLAW [2]

- [1] Sanchez-Gonzales+ Learning to Simulate Complex Physics with Graph Neural Networks
- [2] Ma+ Learning neural constitutive laws from motion observations for generalizable pde dynamics

#### Dataset preprocessing:

    cd Dataset\ Parsers/
    python parseData.py --data_config nclaw_Sand
    python parseData.py --data_config WaterDrop2D

#### Preparing NCLAW datasets (Optional)

> **Dataset Requirement:** The datasets used in our experiments must be generated using the official implementation from the paper *Learning Neural Constitutive Laws from Motion Observations for Generalizable PDE Dynamics* (Ma et al., ICML 2023). Specifically:
>
> Ma, P., Chen, P. Y., Deng, B., Tenenbaum, J. B., Du, T., Gan, C., & Matusik, W. (2023). *Learning Neural Constitutive Laws from Motion Observations for Generalizable PDE Dynamics*. International Conference on Machine Learning (ICML), PMLR, pp. 23279–23300.
>
> If you wish to  re-build the dataset using their repository and simulation framework before running our model, follow the procedure described below.

The dt used for the simulations is 5e-3. During generation, this can be set in ```configs/sim/high.yaml```, ```configs/sim/low.yaml```. Alternatively, after generating, every tenth frame can be used in the train dataset

Each material has a hard-coded geometry. This can be found in ```configs/env/blob/armadillo.yaml```, ```configs/env/blob/bunny.yaml``` etc. The defaults can be changed to ```jelly```, ```sand```, ```water```, ```plasticine```. In the file ```nclaw/constants.py``` you can add shapes to ```SHAPE_ENVS``` dictionary. eg. ```'jelly':['bunny', 'armadillo', 'spot', 'blub']```. While generating, this will generate trajectories for all the geometries. To randomize the trajectories, ```config/env/blob/"shape".yaml``` has a parameter called ```override vel```. This can be set to random. Inside ```configs/env/blob/vel/random.yaml```, the seed can be changed to change the initial velocity, altering the trajectory. This can be done manually or within ```eval.py```

    state_root: Path = exp_root / f'state_{seed}' #Add the seed in foldername to create new folders for each trajectory so that the paths look like this /material/shape/armadillo/state_{seed}
    for blob, blob_cfg in sorted(cfg.env.items()):
        blob_cfg.vel['seed'] = seed # Add this line
        

The config files are formatted in a slightly different way, but this can be changed depending on how the dataset is generated. In the below structure, shape_1 and shape_2 refer to two different "runs", generated with different random velocity seeds, 

    water
        - shape_1
            - armadillo
                - state
                    - 0000.pt   #System state at t0
                    - 0001.pt   #System state at t1
                    ...
                - pv
            - blub
            - bunny
            - spot
        - shape_2

    

All the config paths provided are for reference and need to be updated before they can be used. 

#### Train a time-stepper model from the config:

    python run.py --train_config train_configs_nclaw_Water

#### Train a time-stepper model with args:

    python run.py --batch_size 2 --epoch 100 --lr 0.0001 --noise 0.0003 --eval_interval 1500 --rollout_interval 1500 --sampling true --sampling_strategy fps --graph_type radius --connectivity_radius 0.032 --model giorom2d_large --dataset WaterDrop2D --load_checkpoint true --ckpt_name giorom2d_large_WaterDrop2D --dataset_rootdir giorom_datasets/

#### Evaluate a time-stepper model (Untested code):

    python eval.py --eval_config train_configs_Water2D

We have not provided the code to save the rollout output as a pytorch tensor. However, this snippet can be found at ```eval_3d.ipynb```
To render the results with Polyscope or Blender Use the following. This part of the code needs to be modified and appropriate paths need to be provided in the code

    cd Viz
    python createObj.py
    python blender_rendering.py