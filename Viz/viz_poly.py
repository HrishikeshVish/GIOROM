import polyscope as ps
ps.init()
import torch
import numpy as np
input_mesh = torch.load('rollout_dense.pt')
predicted_mesh = torch.load('big_creature_2.pt')

idxx = 0
def callback():
    global idxx, U, random_indices
    idx = idxx % 200
    idxx = idxx + 1
    gt = input_mesh[idx].numpy()
    x = predicted_mesh[idx].numpy()
    #with h5py.File('/home/hviswan/Documents/Neural Operator/PCA/Sand/ground_truth/h5_f_'+str(idx).zfill(10)+'.h5', 'r') as h5_file: #change the file path
    #        x = h5_file['/x'][:].T
    #        q = h5_file['/q'][:].T

    #V = np.zeros((q.shape[0], 3))
    #V[:,0] = x[:,0] + q[:,0]
    #V[:,1] = x[:,1] + q[:,1]
    V = np.zeros((gt.shape[0], 3))
    V[:,0] = gt[:,0]
    V[:,1] = gt[:,1]
    V[:,2] = gt[:,2]

    #print(q)
    #V = V[random_indices.numpy(),:] # random sampling
    #q = q[random_indices.numpy(),:] # random sampling

        
    ps_cloud = ps.register_point_cloud("ground_truth", V)
    #with h5py.File('/home/hviswan/Documents/Neural Operator/PCA/Sand/predicted/h5_f_'+str(idx).zfill(10)+'.h5', 'r') as h5_file: #change the file path
    #        #x = h5_file['/x'][:].T
    #        q = h5_file['/q'][:].T
    #q = torch.tensor(q)
    #q = q.reshape(-1, 1)

    #xhat = encoder @ q
    #reconstruct = U @ xhat
    #print(xhat.shape)
    #reconstruct = reconstruct.reshape(-1,2).cpu().numpy()


    V = np.zeros((x.shape[0], 3))
    V[:,0] = x[:,0]
    V[:,1] = x[:,1]
    V[:,2] = x[:,2]
    #V[:,0] = x[:,0] + reconstruct[:,0]
    #V[:,1] = x[:,1] + reconstruct[:,1]

    ps_cloud2 = ps.register_point_cloud("predicted", V)
    ps.screenshot()
    #if(idxx % 10 == 0):
    #    ps.screenshot()
	    
def test():
    
    
    ps.set_user_callback(callback)
    ps.set_ground_plane_mode('none')
    ps.show()
    ps.clear_user_callback()

test()