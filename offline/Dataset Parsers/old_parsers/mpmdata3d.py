import torch
import random
import numpy as np
import json
bunnydata = torch.load('/home/hviswan/Documents/new_dataset/spot.pt')
print(bunnydata['x'].shape)

rollout_traj = []
for i in range(0, 2001, 10):
    rollout_traj.append(bunnydata['x'][i].cpu().numpy())
rollout_traj = torch.from_numpy(np.asarray(rollout_traj))


velocities = []
for i in range(1, rollout_traj.shape[0]):
    velocities.append((rollout_traj[i]-rollout_traj[i-1]).numpy())

accelerations = []
for i in range(1, len(velocities)):
    accelerations.append((velocities[i] - velocities[i-1]))

velocities = torch.from_numpy(np.asarray(velocities))
accelerations = torch.from_numpy(np.asarray(accelerations))
x_mean = rollout_traj.mean(dim=[0,1])
x_std = rollout_traj.std(dim=[0,1])
v_mean = velocities.mean(dim=[0,1])
v_std = velocities.std(dim=[0,1])
a_mean = accelerations.mean(dim=[0,1])
a_std = accelerations.std(dim=[0,1])
meta = {"bounds": [[0.1, 0.9], [0.1, 0.9], [0.1,0.9]], "sequence_length": 200, "default_connectivity_radius": 0.015, "dim": 2, "dt": 0.0025, 
        "vel_mean": [1.1927917091800243e-05, -0.0002563314637168018], "vel_std": [0.0013973410613251076, 0.00131291713199288], "acc_mean": [-1.10709094667326e-08, 8.749365512454699e-08], "acc_std": [6.545267379756913e-05, 7.965494666766224e-05]}
meta['vel_mean'] = list(v_mean.numpy())
meta['vel_std'] = list(v_std.numpy())
meta['acc_mean'] = list(a_mean.numpy())
meta['acc_std'] = list(a_std.numpy())
print(meta)
# with open('/home/hviswan/Documents/new_dataset/bunny.json', 'w') as outfile:
#     json.dump(meta, outfile)
particle_type = []
position = []
n_particles_per_example = []
y = []

for i in range(550):
    mesh_size = np.random.randint(int(0.01*31580), int(0.03*31580))
    points = sorted(list(random.sample(range(0, 31580), mesh_size)))
    position_tensor = []
    sampled_tensor = rollout_traj.permute(1, 0, 2)
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
    for j in range(6, rollout_traj.shape[0]-1):
        init_pos.pop(0)
        init_pos.append(sampled_tensor[j].numpy())
        pos_tensor = torch.from_numpy(np.asarray(init_pos))
        pos_tensor = pos_tensor.permute(1,0,2)
        position.append(pos_tensor.numpy())
        n_particles_per_example.append(np.asarray([len(points)]).astype(np.int32))
        particle_type.append(p_type)
        y.append(sampled_tensor[j+1].numpy())
print("HERE")
print(particle_type[0].shape)
train_dict = {'particle_type':particle_type, 'position':position, 'n_particles_per_example':n_particles_per_example, 'output':y}
torch.save(train_dict, '/home/hviswan/Documents/new_dataset/WaterDropSmall/bunny_train.pt')

particle_type = []
position = []
n_particles_per_example = []
y = []

for i in range(10):
    mesh_size = np.random.randint(int(0.01*31580), int(0.03*31580))
    points = sorted(list(random.sample(range(0, 31580), mesh_size)))
    position_tensor = []
    sampled_tensor = rollout_traj.permute(1, 0, 2)
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
    for j in range(6, rollout_traj.shape[0]-1):
        init_pos.pop(0)
        init_pos.append(sampled_tensor[j].numpy())
        pos_tensor = torch.from_numpy(np.asarray(init_pos))
        pos_tensor = pos_tensor.permute(1,0,2)
        position.append(pos_tensor.numpy())
        n_particles_per_example.append(np.asarray([len(points)]).astype(np.int32))
        particle_type.append(p_type)
        y.append(sampled_tensor[j+1].numpy())

train_dict = {'particle_type':particle_type, 'position':position, 'n_particles_per_example':n_particles_per_example, 'output':y}
torch.save(train_dict, '/home/hviswan/Documents/new_dataset/WaterDropSmall/bunny_test.pt')
particle_type = []
position = []
n_particles_per_example = []
y = []
for i in range(2):
    mesh_size = np.random.randint(int(0.01*31580), int(0.03*31580))
    points = sorted(list(random.sample(range(0, 31580), mesh_size)))
    position_tensor = []
    sampled_tensor = rollout_traj.permute(1, 0, 2)
    sampled_tensor = sampled_tensor[points].cpu()
    #sampled_tensor = sampled_tensor.permute(1,0,2)
    position.append(sampled_tensor.numpy())
    
    p_type = (torch.ones(len(points))*5).numpy().astype(np.int32)
    n_particles_per_example.append(np.asarray([len(points)]).astype(np.int32))
    y.append(sampled_tensor[6].numpy())
    particle_type.append(p_type)

print(position[0].shape)

train_dict = {'particle_type':particle_type, 'position':position, 'n_particles_per_example':n_particles_per_example, 'output':y}
torch.save(train_dict, '/home/hviswan/Documents/new_dataset/WaterDropSmall/bunny_rollout.pt')