import torch
import random
import numpy as np
import h5py
import json
import pickle
#bunnydata = torch.load('/home/hviswan/Documents/new_dataset/spot.pt')
#owlpath = '/home/csuser/Documents/new_dataset/owl_p2d/sim_seq_'
owlpath = '/home/csuser/Documents/new_dataset/0401/1_p2d/sim_seq_data_'
owlpath2 = '/home/csuser/Documents/new_dataset/0401/2_p2d/sim_seq_data_'
#random.seed(42)

full_rollout = []
full_velocity = []
full_acceleration = []
for j in range(1, 2):
    filepath = owlpath + str(j)
    rollout_traj = []
    for i in range(5, 200, 5):
        file = filepath + '/h5_f_' + str(i).zfill(10) + '.h5'
        with h5py.File(file, 'r') as f:
            
            data = f
            
            position  = np.asarray(data['x']) + np.asarray(data['q'])
            rollout_traj.append(position)
    rollout_traj = np.asarray(rollout_traj)
    full_rollout.append(rollout_traj)


    velocities = []
    for i in range(1, rollout_traj.shape[0]):
        velocities.append((rollout_traj[i]-rollout_traj[i-1]))
    

    velocities = np.asarray(velocities)
    full_velocity.append(velocities)
    accelerations = []
    for i in range(1, len(velocities)):
        accelerations.append((velocities[i] - velocities[i-1]))
    accelerations = np.asarray(accelerations)
    full_acceleration.append(accelerations)

full_rollout = torch.from_numpy(np.asarray(full_rollout))
full_velocity = torch.from_numpy(np.asarray(full_velocity))
full_acceleration = torch.from_numpy(np.asarray(full_acceleration))
print(full_rollout.shape)


x_mean = full_rollout.mean(dim=[0,1, 3])
x_std = full_rollout.std(dim=[0,1, 3])
v_mean = full_velocity.mean(dim=[0,1, 3])
v_std = full_velocity.std(dim=[0,1, 3])
a_mean = full_acceleration.mean(dim=[0,1, 3])
a_std = full_acceleration.std(dim=[0,1, 3])
meta = {"bounds": [[-2.3160473e-01, 0.232097],[-6.5052130e-19, 0.89966255],[-2.0693564e-01, 0.206798]], "sequence_length": 400, "default_connectivity_radius": 0.015, "dim": 3, "dt": 0.025, 
        "vel_mean": [1.1927917091800243e-05, -0.0002563314637168018], "vel_std": [0.0013973410613251076, 0.00131291713199288], "acc_mean": [-1.10709094667326e-08, 8.749365512454699e-08], "acc_std": [6.545267379756913e-05, 7.965494666766224e-05]}
meta['vel_mean'] = list(v_mean.numpy())
meta['vel_std'] = list(v_std.numpy())
meta['acc_mean'] = list(a_mean.numpy())
meta['acc_std'] = list(a_std.numpy())
print(meta)
# with open('/home/hviswan/Documents/new_dataset/bunny.json', 'w') as outfile:
#     json.dump(meta, outfile)

full_rollout_valid = []
full_velocity_valid = []
full_acceleration_valid = []
for j in range(2, 3):
    filepath = owlpath2 + str(j)
    rollout_traj = []
    for i in range(5, 200, 5):
        file = filepath + '/h5_f_' + str(i).zfill(10) + '.h5'
        with h5py.File(file, 'r') as f:
            
            data = f
            
            position  = np.asarray(data['x']) + np.asarray(data['q'])
            rollout_traj.append(position)
    rollout_traj = np.asarray(rollout_traj)
    full_rollout_valid.append(rollout_traj)


    velocities = []
    for i in range(1, rollout_traj.shape[0]):
        velocities.append((rollout_traj[i]-rollout_traj[i-1]))
    

    velocities = np.asarray(velocities)
    full_velocity_valid.append(velocities)
    accelerations = []
    for i in range(1, len(velocities)):
        accelerations.append((velocities[i] - velocities[i-1]))
    accelerations = np.asarray(accelerations)
    full_acceleration_valid.append(accelerations)

full_rollout_valid = torch.from_numpy(np.asarray(full_rollout_valid))
full_velocity_valid = torch.from_numpy(np.asarray(full_velocity_valid))
full_acceleration_valid = torch.from_numpy(np.asarray(full_acceleration_valid))
print(full_rollout_valid.shape)


x_mean = full_rollout_valid.mean(dim=[0,1, 3])
x_std = full_rollout_valid.std(dim=[0,1, 3])
v_mean = full_velocity_valid.mean(dim=[0,1, 3])
v_std = full_velocity_valid.std(dim=[0,1, 3])
a_mean = full_acceleration_valid.mean(dim=[0,1, 3])
a_std = full_acceleration_valid.std(dim=[0,1, 3])
meta = {"bounds": [[-2.3160473e-01, 0.232097],[-6.5052130e-19, 0.89966255],[-2.0693564e-01, 0.206798]], "sequence_length": 400, "default_connectivity_radius": 0.015, "dim": 3, "dt": 0.025, 
        "vel_mean": [1.1927917091800243e-05, -0.0002563314637168018], "vel_std": [0.0013973410613251076, 0.00131291713199288], "acc_mean": [-1.10709094667326e-08, 8.749365512454699e-08], "acc_std": [6.545267379756913e-05, 7.965494666766224e-05]}
meta['vel_mean'] = list(v_mean.numpy())
meta['vel_std'] = list(v_std.numpy())
meta['acc_mean'] = list(a_mean.numpy())
meta['acc_std'] = list(a_std.numpy())
print(meta)


particle_type = []
position = []
n_particles_per_example = []
y = []
mesh_size = 1000
for k in range(1):
    for i in range(500):
        mesh_size = np.random.randint(int(0.02*79908), int(0.05*79908))
        points = sorted(list(random.sample(range(0, 79908), mesh_size)))
        position_tensor = []
        sampled_tensor = full_rollout[k].permute(2, 0, 1)
        sampled_tensor = sampled_tensor[points].cpu()
        sampled_tensor = sampled_tensor.permute(1,0,2)
        init_pos = [sampled_tensor[0].numpy(), sampled_tensor[1].numpy(), sampled_tensor[2].numpy(), sampled_tensor[3].numpy(), sampled_tensor[4].numpy(), sampled_tensor[5].numpy()]
        p_type = (torch.ones(len(points))*5).numpy().astype(np.int32)
        pos_tensor = torch.from_numpy(np.asarray(init_pos))
        pos_tensor = pos_tensor.permute(1,0,2)

        position.append(pos_tensor.numpy())
        n_particles_per_example.append(np.asarray([len(points)]).astype(np.int32))
        y.append(sampled_tensor[6].numpy())
        particle_type.append(p_type)
        for j in range(6, full_rollout[k].shape[0]-1):
            init_pos.pop(0)
            init_pos.append(sampled_tensor[j].numpy())
            pos_tensor = torch.from_numpy(np.asarray(init_pos))
            pos_tensor = pos_tensor.permute(1,0,2)
            position.append(pos_tensor.numpy())
            n_particles_per_example.append(np.asarray([len(points)]).astype(np.int32))
            particle_type.append(p_type)
            y.append(sampled_tensor[j+1].numpy())
print("HERE")
print(len(particle_type))
print(position[0].shape)
print(particle_type[0].shape)
train_dict = {'particle_type':particle_type, 'position':position, 'n_particles_per_example':n_particles_per_example, 'output':y}
with open(f'/home/csuser/Documents/new_dataset/human/train.obj', 'wb') as f:
    pickle.dump(train_dict, f)
print("DONE")
#torch.save(train_dict, '/home/csuser/Documents/new_dataset/owl/train.pt')

particle_type = []
position = []
n_particles_per_example = []
y = []
for k in range(1):
    for i in range(12):
        mesh_size = np.random.randint(int(0.02*79760), int(0.05*79760))
        points = sorted(list(random.sample(range(0, 79760), mesh_size)))
        position_tensor = []
        sampled_tensor = full_rollout_valid[k].permute(2, 0, 1)
        sampled_tensor = sampled_tensor[points].cpu()
        sampled_tensor = sampled_tensor.permute(1,0,2)
        init_pos = [sampled_tensor[0].numpy(), sampled_tensor[1].numpy(), sampled_tensor[2].numpy(), sampled_tensor[3].numpy(), sampled_tensor[4].numpy(), sampled_tensor[5].numpy()]
        p_type = (torch.ones(len(points))*5).numpy().astype(np.int32)
        pos_tensor = torch.from_numpy(np.asarray(init_pos))
        pos_tensor = pos_tensor.permute(1,0,2)

        position.append(pos_tensor.numpy())
        n_particles_per_example.append(np.asarray([len(points)]).astype(np.int32))
        y.append(sampled_tensor[6].numpy())
        particle_type.append(p_type)
        for j in range(6, full_rollout[k].shape[0]-1):
            init_pos.pop(0)
            init_pos.append(sampled_tensor[j].numpy())
            pos_tensor = torch.from_numpy(np.asarray(init_pos))
            pos_tensor = pos_tensor.permute(1,0,2)
            position.append(pos_tensor.numpy())
            n_particles_per_example.append(np.asarray([len(points)]).astype(np.int32))
            particle_type.append(p_type)
            y.append(sampled_tensor[j+1].numpy())

train_dict = {'particle_type':particle_type, 'position':position, 'n_particles_per_example':n_particles_per_example, 'output':y}
torch.save(train_dict, '/home/csuser/Documents/new_dataset/human/test.pt')

particle_type = []
position = []
n_particles_per_example = []
y = []
for i in range(2):
    mesh_size = np.random.randint(int(0.02*79760), int(0.05*79760))
    points = sorted(list(random.sample(range(0, 79760), mesh_size)))
    position_tensor = []
    sampled_tensor = full_rollout_valid[0].permute(2, 0, 1)
    sampled_tensor = sampled_tensor[points].cpu()
    #sampled_tensor = sampled_tensor.permute(1,0,2)
    position.append(sampled_tensor.numpy())
    
    p_type = (torch.ones(len(points))*5).numpy().astype(np.int32)
    n_particles_per_example.append(np.asarray([len(points)]).astype(np.int32))
    y.append(sampled_tensor[6].numpy())
    particle_type.append(p_type)

print(position[0].shape)

train_dict = {'particle_type':particle_type, 'position':position, 'n_particles_per_example':n_particles_per_example, 'output':y}
torch.save(train_dict, '/home/csuser/Documents/new_dataset/human/rollout.pt')

particle_type = []
position = []
n_particles_per_example = []
y = []
for i in range(2):
    mesh_size = np.random.randint(int(0.02*79760), int(0.05*79760))
    points = sorted(list(random.sample(range(0, 79760), mesh_size)))
    position_tensor = []
    sampled_tensor = full_rollout[0].permute(2, 0, 1)
    sampled_tensor = sampled_tensor[points].cpu()
    #sampled_tensor = sampled_tensor.permute(1,0,2)
    position.append(sampled_tensor.numpy())
    
    p_type = (torch.ones(len(points))*5).numpy().astype(np.int32)
    n_particles_per_example.append(np.asarray([len(points)]).astype(np.int32))
    y.append(sampled_tensor[6].numpy())
    particle_type.append(p_type)

print(position[0].shape)

train_dict = {'particle_type':particle_type, 'position':position, 'n_particles_per_example':n_particles_per_example, 'output':y}
torch.save(train_dict, '/home/csuser/Documents/new_dataset/human/rollout_gt.pt')