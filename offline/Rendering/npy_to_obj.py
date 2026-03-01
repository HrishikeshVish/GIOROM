import numpy as np
import os
import torch


def write_obj(vertices, output_obj_path='output.obj'):
    with open(output_obj_path, 'w') as obj_file:
        for vertex in vertices:
            obj_file.write(f'v {vertex[0]} {vertex[1]} {vertex[2]}\n')

    print(f'OBJ file saved to {output_obj_path}')


def bulk_npy_to_obj(npy_folder_path: str, output_folder_path: str = ""):
    """Convert all npy files in the given folder to obj and save objs to same directory"""
    if output_folder_path == "":
        output_folder_path = npy_folder_path
    for f in os.listdir(npy_folder_path):
        if f.endswith(".npy"):
            output_filename = ".".join(f.split(".")[:-1]) + ".obj"
            filepath = os.path.join(npy_folder_path, f)
            arr = np.load(filepath)
            write_obj(arr, output_obj_path=os.path.join(output_folder_path, output_filename))


def pt_file_to_objs(pt_filepath: str, output_folder_path: str = ""):
    """Parse pytorch tensor file and output obj files for each frame into the same folder, each named i.obj for frame i"""
    dat = torch.load(pt_filepath)
    if output_folder_path == "":
        folder_path = os.path.join(*(os.path.split(pt_filepath))[:-1])
    else:
        folder_path = output_folder_path
    for i in range(len(dat)):
        write_obj(dat[i], os.path.join(folder_path, f"{i}.obj"))


if __name__ == "__main__":
    p = "./Rendering/0129_results_all/Results_NKF"
    bulk_npy_to_obj(p, os.path.join(p, "input"))
