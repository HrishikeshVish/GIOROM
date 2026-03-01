This directory contains the code for processing raw simulation data and training the offline full-order and time-stepper models.  

> **Disclaimer:** The core contribution of our paper is the **Online Phase** algorithm. While we provide this offline training code for reproducibility, our pipeline is highly flexible. In practice, *any* high-fidelity physics simulator or neural network can be utilized for the offline phase to generate the foundational trajectories.

---

#### ⚙️ Setup & Installation

From the parent directory, install the required dependencies:

```bash
conda create --name giorom_env --file requirements.txt
conda activate giorom_env
```
#### 📂 Datasets

We support existing datasets provided by GNS and new datasets curated from the NCLAW framework.

Create a directory for your datasets:

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

#### Preparing NCLAW datasets

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