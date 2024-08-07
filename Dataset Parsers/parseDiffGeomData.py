import torch
import random
import numpy as np
import h5py
import json
import pickle
import os

class NumpyTypeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.generic):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    
    
def generate_data(dataset_name, dataset_root_dir, save_root_dir,  traj_dir_prefix='sim_seq_', train_geoms=[], eval_geoms=[],
                  start_time_step=5, stop_time_step=605,
                  jump_time_step=5, window= 6, randomize=True, use_random_seed=False,random_seed=42,test_train_split=0.8,
                  rand_percent_lower=0.05, rand_percent_higher=0.1, num_augmented_train_trajs=20, num_augmented_eval_trajs=1,
                  max_rollout_trajectories=5, set_size_limit=False, max_particle_size=1000, save_hires=True):
    
    assert rand_percent_lower<rand_percent_higher
    assert rand_percent_lower<=1.0
    assert rand_percent_higher<=1.0
    assert stop_time_step>=start_time_step
    assert start_time_step>=0
    assert isinstance(random_seed, int)
    if(len(train_geoms) <=0):
        raise Exception("No Train Geometries provided")
    if(len(eval_geoms)<=0):
        raise Exception("No Eval Geometries Provided")
    
    
    if(randomize == True and use_random_seed == True):
        random.seed(random_seed)
    datapath = os.path.join(dataset_root_dir, dataset_name)
    
    num_trajectories = len(os.listdir(datapath))
    num_train_trajs = int(test_train_split*num_trajectories)
    num_test_trajs = num_trajectories - num_train_trajs
    
    if(os.path.exists(datapath) == False):
        raise Exception("Invalid Dataset path")
    
    if(os.path.exists(save_root_dir) == False):
        raise Exception("Invalid Save Path")
    else:
        save_path = os.path.join(save_root_dir, dataset_name)
    if(os.path.exists(save_path) == False):
        os.mkdir(save_path)
        
    full_rollout_train = []
    for train_geom in train_geoms:
        full_rollout = []
        full_velocity = []
        full_acceleration = []
        filepath = os.path.join(datapath, train_geom, traj_dir_prefix)
        rollout_traj = []
        for i in range(start_time_step, stop_time_step, jump_time_step):
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
        meta = {"bounds": np.array([[-2.3160473e-01, 0.232097],[-6.5052130e-19, 0.89966255],[-2.0693564e-01, 0.206798]]), "sequence_length": np.int32(400), "default_connectivity_radius": np.float32(0.015), "dim": np.int32(3), "dt": np.float32(0.025), 
                "vel_mean": [1.1927917091800243e-05, -0.0002563314637168018], "vel_std": [0.0013973410613251076, 0.00131291713199288], "acc_mean": [-1.10709094667326e-08, 8.749365512454699e-08], "acc_std": [6.545267379756913e-05, 7.965494666766224e-05]}
        meta['vel_mean'] = np.array(list(v_mean.numpy()))
        meta['vel_std'] = np.array(list(v_std.numpy()))
        meta['acc_mean'] = np.array(list(a_mean.numpy()))
        meta['acc_std'] = np.array(list(a_std.numpy()))
        print(meta)
        with open(os.path.join(save_path, 'metadata.json'), 'w') as fp:
            json.dump(meta, fp, cls=NumpyTypeEncoder, indent=2)
        full_rollout_train.append(full_rollout)
        
    full_rollout_valid = []
    for valid_geom in eval_geoms:
        full_rollout = []
        full_velocity = []
        full_acceleration = []
        filepath = os.path.join(datapath, valid_geom, traj_dir_prefix)
        rollout_traj = []
        for i in range(start_time_step, stop_time_step, jump_time_step):
            file = filepath + '/h5_f_' + str(i).zfill(10) + '.h5'
            with h5py.File(file, 'r') as f:
            
                data = f
            
                position  = np.asarray(data['x']) + np.asarray(data['q'])
                rollout_traj.append(position)
        rollout_traj = np.asarray(rollout_traj)
        full_rollout.append(rollout_traj)
        full_rollout_valid.append(full_rollout)


    particle_type = []
    position = []
    n_particles_per_example = []
    y = []
    
    for k in range(len(full_rollout_train)):
        print(f'Generating Training samples for Geometry {k}')
        shape = full_rollout_train[k][0].shape[2]
        print("SHAPE = ", shape)
        for i in range(num_augmented_train_trajs):
            mesh_size = np.random.randint(int(rand_percent_lower*shape), int(rand_percent_higher*shape))
            points = sorted(list(random.sample(range(0, shape), mesh_size)))
            
            sampled_tensor = full_rollout_train[k][0].permute(2, 0, 1)
            if(randomize == True):
                sampled_tensor = sampled_tensor[points].cpu()
            sampled_tensor = sampled_tensor.permute(1,0,2)
            init_pos = []
            for w in range(window):
                init_pos.append(sampled_tensor[w].numpy())
                
            p_type = (torch.ones(len(points))*5).numpy().astype(np.int32)
            pos_tensor = torch.from_numpy(np.asarray(init_pos))
            pos_tensor = pos_tensor.permute(1,0,2)

            position.append(pos_tensor.numpy())
            n_particles_per_example.append(np.asarray([len(points)]).astype(np.int32))
            y.append(sampled_tensor[window].numpy())
            particle_type.append(p_type)
            for j in range(window, full_rollout_train[k][0].shape[0]-1):
                init_pos.pop(0)
                init_pos.append(sampled_tensor[j].numpy())
                pos_tensor = torch.from_numpy(np.asarray(init_pos))
                pos_tensor = pos_tensor.permute(1,0,2)
                position.append(pos_tensor.numpy())
                n_particles_per_example.append(np.asarray([len(points)]).astype(np.int32))
                particle_type.append(p_type)
                y.append(sampled_tensor[j+1].numpy())
    print("....Training Data Generated....")
    print(len(particle_type))
    print(position[0].shape)
    print(particle_type[0].shape)
    train_dict = {'particle_type':particle_type, 'position':position, 'n_particles_per_example':n_particles_per_example, 'output':y}
    with open(os.path.join(save_path, 'train.obj'), 'wb') as f:
        pickle.dump(train_dict, f)
    print("...Training Data Saved...")

    particle_type = []
    position = []
    n_particles_per_example = []
    y = []
    print("...Generating Validation Trajectories...")
    for k in range(len(full_rollout_valid)):
        for i in range(num_augmented_eval_trajs):
            print(f'Generating Validation Trajectory {i} for shape {k}, {dataset_name}')
            shape = full_rollout_valid[k][0].shape[2]
            mesh_size = np.random.randint(int(rand_percent_lower*shape), int(rand_percent_higher*shape))
            points = sorted(list(random.sample(range(0, shape), mesh_size)))
            position_tensor = []
            sampled_tensor = full_rollout_valid[k][0].permute(2, 0, 1)
            if(randomize == True):
                sampled_tensor = sampled_tensor[points].cpu()
            sampled_tensor = sampled_tensor.permute(1,0,2)
            init_pos = []
            for w in range(window):
                init_pos.append(sampled_tensor[w].numpy())
                
            p_type = (torch.ones(len(points))*5).numpy().astype(np.int32)
            pos_tensor = torch.from_numpy(np.asarray(init_pos))
            pos_tensor = pos_tensor.permute(1,0,2)

            position.append(pos_tensor.numpy())
            n_particles_per_example.append(np.asarray([len(points)]).astype(np.int32))
            y.append(sampled_tensor[window].numpy())
            particle_type.append(p_type)
            for j in range(window, full_rollout_valid[k][0].shape[0]-1):
                init_pos.pop(0)
                init_pos.append(sampled_tensor[j].numpy())
                pos_tensor = torch.from_numpy(np.asarray(init_pos))
                pos_tensor = pos_tensor.permute(1,0,2)
                position.append(pos_tensor.numpy())
                n_particles_per_example.append(np.asarray([len(points)]).astype(np.int32))
                particle_type.append(p_type)
                y.append(sampled_tensor[j+1].numpy())


    train_dict = {'particle_type':particle_type, 'position':position, 'n_particles_per_example':n_particles_per_example, 'output':y}
    torch.save(train_dict, os.path.join(save_path, 'test.pt'))
    print('...Validation Data Saved...')
    
    particle_type = []
    position = []
    n_particles_per_example = []
    y = []
    print('...Generating Rollout Data...')
    for i in range(len(full_rollout_valid)):
        shape = full_rollout_valid[i][0].shape[2]
        mesh_size = np.random.randint(int(rand_percent_lower*shape), int(rand_percent_higher*shape))
        points = sorted(list(random.sample(range(0, shape), mesh_size)))
        sampled_tensor = full_rollout_valid[i][0].permute(2, 0, 1)
        if(randomize == True):
            sampled_tensor = sampled_tensor[points].cpu()
        position.append(sampled_tensor.numpy())
        
        p_type = (torch.ones(len(points))*5).numpy().astype(np.int32)
        n_particles_per_example.append(np.asarray([len(points)]).astype(np.int32))
        y.append(sampled_tensor[6].numpy())
        particle_type.append(p_type)

    print(f'Number of Rollout Trajectories: {position[0].shape}')
    train_dict = {'particle_type':particle_type, 'position':position, 'n_particles_per_example':n_particles_per_example, 'output':y}
    torch.save(train_dict, os.path.join(save_path, 'rollout.pt'))
    print("...Rollout Trajectories Saved...")
    
    if(save_hires == True):
        particle_type = []
        position = []
        n_particles_per_example = []
        y = []
        for i in range(len(full_rollout_valid)):
            shape = full_rollout_valid[i][0].shape[2]
            mesh_size = np.random.randint(int(0.971*shape), int(0.986*shape))
            points = sorted(list(random.sample(range(0, shape), mesh_size)))
            position_tensor = []
            sampled_tensor = full_rollout_valid[0][0].permute(2, 0, 1)
            sampled_tensor = sampled_tensor[points].cpu()
            #sampled_tensor = sampled_tensor.permute(1,0,2)
            position.append(sampled_tensor.numpy())
            
            p_type = (torch.ones(len(points))*5).numpy().astype(np.int32)
            n_particles_per_example.append(np.asarray([len(points)]).astype(np.int32))
            y.append(sampled_tensor[6].numpy())
            particle_type.append(p_type)
            
        print(f'Number of Rollout Trajectories: {position[0].shape}')

        train_dict = {'particle_type':particle_type, 'position':position, 'n_particles_per_example':n_particles_per_example, 'output':y}
        
        torch.save(train_dict, os.path.join(save_path, 'rollout_gt.pt'))