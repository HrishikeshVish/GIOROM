import polyscope as ps
ps.init()
import torch
import numpy as np
import h5py
import os
#from NetMap import *


sim_id= 7
#path = f'plasticine_res/plasticine_{sim_id}'
path = f'Owl/'
#path = f'water_nclaw_discs_res/water_{sim_id}/'
if(os.path.exists(os.path.join(os.getcwd(), path)) == False):
    os.mkdir(os.path.join(os.getcwd(), path))
#input_mesh = torch.load(f'Plasticine/GIOROM_Plasticine_{sim_id}.pt')
input_mesh = torch.load('Owl/GIOROM_owl_sim_0.pt')
print(input_mesh.shape)


#exit()
def export_point_cloud_to_obj(vertices, filename):
    with open(filename, 'w') as f:
        # Write vertices
        for vertex in vertices:
            f.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
            
            
idxx = 0
x0 = None
def callback():
    global idxx, U, random_indices, encoder
    global summ1, summ2
    global V_prev
    global V0_prev
    global V_prev2, x0
    global V0_prev2
    
    
    idx = idxx #% 200
    idxx = idxx + 1
    #print(input_mesh.shape)
    gt = input_mesh[idx,:]#gt looks like this
    
    #gt = input_mesh[idx, :]
    print(gt.shape)
    
    x = gt #+ disp
    V = np.zeros((x.shape[0], 3))
    V[:,0] = x[:,0] #+ 1
    V[:,1] = x[:,1]
    V[:,2] = x[:,2]
    
    if x0 is None:
       x0 = V.copy()
    vt = V.copy()
    
    
    #print("HERE")
    ps_cloud2 = ps.register_point_cloud("gt", V) #, enabled = False)
    ps.screenshot()
    export_point_cloud_to_obj(V,os.path.join(path, str(idxx) + ".obj"))
    
	    
def test():
    
    
    ps.set_user_callback(callback)
    ps.set_ground_plane_mode('none')
    ps.show()
    ps.clear_user_callback()

test()
#callback()