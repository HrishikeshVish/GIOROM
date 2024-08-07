import torch        
import h5py
import os
import numpy as np

points = torch.load('/home/hviswan/Documents/Neural Operator/rollout_ground_truth.pt')['points']

import torch        
import h5py
import os
import numpy as np

 
def PCA():        # don't change this training part !!!!!! 
        snapshots = None
        for i in range(320):
            with h5py.File('/home/hviswan/Documents/Neural Operator/PCA/Sand/ground_truth/h5_f_'+str(i).zfill(10)+'.h5', 'r') as h5_file: #change the file path
               x = h5_file['/x'][:].T
               q = torch.tensor(h5_file['/q'][:].T)
               q = q.reshape(1, q.shape[0], q.shape[1])
               #print(q.shape)
            
            if snapshots is None:
                snapshots = q
            else:
                snapshots = torch.cat([snapshots, q], 0)
        print(snapshots.shape)
        lbllength = 20 #dimension of the latent space, can be some other number
        snapshots = snapshots.view(snapshots.size(0), -1)
        snapshots = torch.transpose(snapshots, 0, 1) # m (vector field values) by n (time-steps), m >= n
        assert(snapshots.size(0)>=snapshots.size(1))

        U, S, V = torch.pca_lowrank(snapshots, lbllength, center=False)
        #print(U.shape)
        
        #you can initialize a network layer using U
        #such as "net.decoder.linear_layer.weight = nn.Parameter(decoder_matrix.clone())"
        return U
        

U = PCA()

print("U shape = ", U.shape)


#the sampling is applied here
U_reshape = U.reshape(-1, 2, 20)
#random_indices = torch.randint(0, U_reshape.shape[0], (50,))
random_indices = torch.from_numpy(np.asarray(points))
print(random_indices)

U_select = U_reshape[random_indices, :, :]
print(U_select.shape)
U_select = U_select.reshape(-1, 20)


encoder = torch.inverse(torch.matmul(U_select.transpose(1, 0), U_select)).matmul(U_select.transpose(1, 0))
print(encoder.shape, U.shape)
idxx = 0

#exit()


#for visualization
import polyscope as ps
ps.init()
    
def callback():
    global idxx, U, random_indices
    idx = idxx % 320
    idxx = idxx + 1
    with h5py.File('/home/hviswan/Documents/Neural Operator/PCA/Sand/ground_truth/h5_f_'+str(idx).zfill(10)+'.h5', 'r') as h5_file: #change the file path
            x = h5_file['/x'][:].T
            q = h5_file['/q'][:].T

    V = np.zeros((q.shape[0], 3))
    V[:,0] = x[:,0] + q[:,0]
    V[:,1] = x[:,1] + q[:,1]
    #print(q)
    V = V[random_indices.numpy(),:] # random sampling
    q = q[random_indices.numpy(),:] # random sampling

        
    ps_cloud = ps.register_point_cloud("ground truth", V)
    with h5py.File('/home/hviswan/Documents/Neural Operator/PCA/Sand/predicted/h5_f_'+str(idx).zfill(10)+'.h5', 'r') as h5_file: #change the file path
            #x = h5_file['/x'][:].T
            q = h5_file['/q'][:].T
    q = torch.tensor(q)
    q = q.reshape(-1, 1)

    xhat = encoder @ q
    reconstruct = U @ xhat
    print(xhat.shape)
    reconstruct = reconstruct.reshape(-1,2).cpu().numpy()


    V = np.zeros((x.shape[0], 3))
    V[:,0] = x[:,0] + reconstruct[:,0]
    V[:,1] = x[:,1] + reconstruct[:,1]

    ps_cloud2 = ps.register_point_cloud("predicted", V)
    #if(idxx % 10 == 0):
    #    ps.screenshot()
	    
def test():
    
    
    ps.set_user_callback(callback)
    ps.set_ground_plane_mode('none')
    ps.show()
    ps.clear_user_callback()

test()
    
