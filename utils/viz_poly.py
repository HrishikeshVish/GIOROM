import polyscope as ps
ps.init()
import torch
import numpy as np
#input_mesh = torch.load('rollout_dense.pt')
#predicted_mesh = torch.load('big_creature_2.pt')
predicted_mesh = torch.load('fom_pred_nclaw_Sand.pt')
#This code is used to visualize the time sequence of the point cloud of the format [T, N, 3], where T is the time-step

idxx = 0
def callback():
    global idxx, U, random_indices
    idx = idxx % 2000
    idxx = idxx + 1

    x = predicted_mesh[idx]


    V = np.zeros((x.shape[0], 3))
    V[:,0] = x[:,0]
    V[:,1] = x[:,1]
    V[:,2] = x[:,2]

    ps_cloud2 = ps.register_point_cloud("predicted", V)
    filename = f'./ps_water_viz_gt/{str(idx).zfill(4)}.png'
    ps.screenshot(filename)
	    
def test():
    
    
    ps.set_user_callback(callback)
    ps.set_ground_plane_mode('none')
    ps.show()
    ps.clear_user_callback()

test()