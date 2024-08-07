import os
import json
import yaml
import argparse
from argparse import Namespace
from copy import deepcopy
parser = argparse.ArgumentParser()
parser.add_argument("-c", "--data_config", help="Dataset Config File Name", required=True)

params = parser.parse_args()
if(params.train_config is not None):
    if(params.train_config.endswith('.yaml') == False):
        params.train_config += '.yaml'
    config_path = os.path.join(os.getcwd(), 'dataset_configs', params.train_config)
    if(os.path.exists(config_path) == False):
        raise Exception("Invalid Config Name")
    with open(config_path, 'r') as f:
        params = yaml.full_load(f)
        func_params = deepcopy(params)
    params = Namespace(**params)

if(params.dataset_name == 'Interaction'):
    from parseDiffGeomData import generate_data
if(params.dataset_name == '0401'):
    from parseDiffGeomData import generate_data
if(params.dataset_name == 'Plasticine'):
    from parseNCLAWData import generate_data
if(params.dataset_name == 'Water'):
    from parseNCLAWData import generate_data
if(params.dataset_name == 'Sand'):
    from parseNCLAWData import generate_data
if(params.dataset_name == 'owl_p2d'):
    from parseFEMElasticity import generate_data
if(params.dataset_name == 'WaterDrop2D'):
    from parseGNSData import generate_data
    
generate_data(**func_params)