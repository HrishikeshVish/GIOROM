from datasets import load_dataset, DownloadConfig
from huggingface_hub import hf_hub_download, snapshot_download
import os
import argparse
from argparse import Namespace
import yaml

base_path = "hrishivish23/MPM-Verse-MaterialSim-Small"
materials = {"Water2D":"WaterDrop2DSmall", "Water3D":"Water3DNCLAWSmall", 
             "Water3D_long":"Water3DNCLAWSmall_longer_duration", "Sand2D":"Sand2DSmall", 
             "Sand3D":"Sand3DNCLAWSmall", "Sand3D_long":"Sand3DNCLAW_Small_longer_duration", 
             "MultiMaterial2D":"MultiMaterial2DSmall", "Plasticine3D":"Plasticine3DNCLAWSmall", 
             "Elasticity3D":"Elasticity3DSmall", "Jelly3D":"Jelly3DNCLAWSmall", "RigidCollision3D":"RigidCollision3DNCLAWSmall", 
             "Melting3D":"Melting3DSampleSeq"}
def load():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--sequence", help="Simulation Sequence (Enter Name of sequence or 'all' for all sequences)", type=str, required=True)
    params = parser.parse_args()
    if(params.sequence is not None):
        sequence = params.sequence
        if sequence == "all":
            # Load the dataset from Hugging Face
            if(os.path.exists("./dataset_mpmverse") is None):
                os.mkdir("./dataset_mpmverse")
            #dataset = load_dataset(base_path, cache_dir="./dataset_mpmverse")
            dataset = snapshot_download(repo_id=base_path, repo_type='dataset', cache_dir="./dataset_mpmverse")
        elif sequence in materials.keys():
            directory = os.path.join(base_path, materials[sequence])
            if('2D' in sequence):
                files = ['train.pt', 'test.pt', 'rollout.pt', 'metadata.json']
            else:
                files = ['train.obj', 'test.pt', 'rollout.pt', 'metadata.json', 'rollout_full.pt']
            cache_dir = "./dataset_mpmverse"
            for file in files:
                dataset = hf_hub_download(repo_id=base_path, repo_type='dataset', filename=os.path.join(materials[sequence], file), cache_dir=cache_dir)
            #dataset = load_dataset(directory, data_dir=materials[sequence], cache_dir="./dataset_mpmverse")
        else:
            raise Exception(f"Invalid Sequence Name, please enter a valid sequence name or 'all' for all sequences. \
                Available sequences are: {list(materials.keys())}")


if __name__ == '__main__':
    load()