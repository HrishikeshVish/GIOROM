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
    

#bunnydata = torch.load('/home/hviswan/Documents/new_dataset/spot.pt')
#owlpath = '/home/csuser/Documents/new_dataset/owl_p2d/sim_seq_'
#dataset_name = 'Sand'
shape_material = {'Sand':'bunny', 'Plasticine':'spot', 'Water':'armadillo'}
def generate_data(dataset_name, dataset_root_dir, save_root_dir, start_time_step=0, stop_time_step=2000, num_trajectories=16,
                  start_traj_index=1, num_train_trajs=12, save_diff_discs=False,
                  jump_time_step=5, window= 6, randomize=True, use_random_seed=False,random_seed=42,test_train_split=0.8,
                  rand_percent_lower=0.05, rand_percent_higher=0.1, num_augmented_train_trajs=20, num_augmented_eval_trajs=1,
                  max_rollout_trajectories=5, set_size_limit=False, max_particle_size=1000, save_hires=True, save_midres=True):
    
    assert rand_percent_lower<rand_percent_higher
    assert rand_percent_lower<=1.0
    assert rand_percent_higher<=1.0
    assert stop_time_step>=start_time_step
    assert start_time_step>=0
    assert isinstance(random_seed, int)    
    
    if(randomize == True and use_random_seed == True):
        random.seed(random_seed)
    datapath = os.path.join(dataset_root_dir, dataset_name)
    
    if(os.path.exists(datapath) == False):
        raise Exception("Invalid Dataset path")
    
    if(os.path.exists(save_root_dir) == False):
        raise Exception("Invalid Save Path")
    else:
        save_path = os.path.join(save_root_dir, 'nclaw_'+dataset_name)
    if(os.path.exists(save_path) == False):
        os.mkdir(save_path)
    
    
    shapes = ['armadillo', 'bunny', 'spot', 'blub']
    full_rollout_train = []
    full_velocity_train = []
    full_acceleration_train = []
    num_train_trajectories = num_train_trajs
    assert stop_time_step <=2000
    for i in range(start_traj_index, num_train_trajectories+start_traj_index):
        #
        if(dataset_name == 'Plasticine'):
            shapes_path = datapath + f'plasticine_v{i}/shape/'
        else:
            shapes_path = datapath + f'shape_{i}/'
        for shape in [shape_material[dataset_name]]:
            traj_path = shapes_path+shape+'/state/'
            rollout_traj = []
            print(f'Loading Trajectory {i} for {shape}_{dataset_name}')
            for j in range(0, stop_time_step, jump_time_step):
                file = traj_path + str(j).zfill(4) + '.pt'
                tensor = torch.load(file)['x']
                rollout_traj.append(tensor.cpu().numpy())
            rollout_traj = np.asarray(rollout_traj)
            full_rollout_train.append(rollout_traj)

            velocities = []
            for j in range(1, rollout_traj.shape[0]):
                velocities.append((rollout_traj[j] - rollout_traj[j-1]))
            velocities = np.asarray(velocities)
            full_velocity_train.append(velocities)

            accelerations = []
            for j in range(1, velocities.shape[0]):
                accelerations.append((velocities[j] - velocities[j-1]))
            accelerations = np.asarray(accelerations)
            full_acceleration_train.append(accelerations)

    print(len(full_rollout_train))
    print(full_rollout_train[0].shape)
    #print(torch.from_numpy(full_velocity_train[0]).mean(dim=[0,1]))
    vel_mean = []
    vel_std = []
    a_mean = []
    a_std = []
    for i in range(len(full_rollout_train)):
        vel_mean.append(torch.from_numpy(full_velocity_train[i]).mean(dim=[0,1]).numpy())
        vel_std.append(torch.from_numpy(full_velocity_train[i]).std(dim=[0,1]).numpy())
        a_mean.append(torch.from_numpy(full_acceleration_train[i]).mean(dim=[0,1]).numpy())
        a_std.append(torch.from_numpy(full_acceleration_train[i]).std(dim=[0,1]).numpy())

    v_mean = torch.from_numpy(np.asarray(vel_mean)).mean(dim=[0])
    v_std = torch.from_numpy(np.asarray(vel_std)).std(dim=[0])
    a_mean = torch.from_numpy(np.asarray(a_mean)).mean(dim=[0])
    a_std = torch.from_numpy(np.asarray(a_std)).std(dim=[0])
    #print(vel_mean, vel_std)


    meta = {"bounds": np.array([[0.1, 0.9],[0.1, 0.9],[0.0, 1.0]]), "sequence_length": np.int32(200), "default_connectivity_radius": np.float32(0.105), "dim": np.int32(3), "dt": np.float32(0.025), 
            "vel_mean": [1.1927917091800243e-05, -0.0002563314637168018], "vel_std": [0.0013973410613251076, 0.00131291713199288], "acc_mean": [-1.10709094667326e-08, 8.749365512454699e-08], "acc_std": [6.545267379756913e-05, 7.965494666766224e-05]}
    meta['vel_mean'] = np.array(list(v_mean.numpy()))
    meta['vel_std'] = np.array(list(v_std.numpy()))
    meta['acc_mean'] = np.array(list(a_mean.numpy()))
    meta['acc_std'] = np.array(list(a_std.numpy()))
    print(meta)
    with open(os.path.join(save_path, 'metadata.json'), 'w') as fp:
        json.dump(meta, fp, cls=NumpyTypeEncoder, indent=2)

    particle_type = []
    position = []
    n_particles_per_example = []
    y = []
    for k in range(len(full_rollout_train)):
        shape = full_rollout_train[k].shape[1]
        print(f'Generating samples for trajectory {k}')

        for i in range(num_augmented_train_trajs):
            if(randomize == True):
                mesh_size = np.random.randint(int(rand_percent_lower*shape), int(rand_percent_higher*shape))
                points = sorted(list(random.sample(range(0, shape), mesh_size)))
            else:
                points = list(range(0, shape))

            sampled_tensor = torch.from_numpy(full_rollout_train[k]).permute(1, 0, 2)
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
            for j in range(window, full_rollout_train[k].shape[0]-1):
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
    print(len(position))
    print(position[0].shape)
    print(particle_type[0].shape)
    train_dict = {'particle_type':particle_type, 'position':position, 'n_particles_per_example':n_particles_per_example, 'output':y}
    with open(os.path.join(save_path, 'train.obj'), 'wb') as f:
        pickle.dump(train_dict, f)
    print("...Training Data Saved...")


    shapes = [shape_material[dataset_name]]
    full_rollout_eval = []
    full_velocity_eval = []
    full_acceleration_eval = []
    print("...Generating Validation Trajectories...")
    for i in range(num_train_trajectories, num_trajectories):
        if(dataset_name == 'Plasticine'):
            shapes_path = datapath + f'plasticine_v{i}/shape/'
        else:
            shapes_path = datapath + f'shape_{i}/'
        for shape in shapes:
            print(f'Generating Validation Trajectory {i} for {shape}_{dataset_name}')
            traj_path = shapes_path+shape+'/state/'
            rollout_traj = []
            for j in range(0, stop_time_step, jump_time_step):
                file = traj_path + str(j).zfill(4) + '.pt'
                tensor = torch.load(file)['x']
                rollout_traj.append(tensor.cpu().numpy())
            rollout_traj = np.asarray(rollout_traj)
            full_rollout_eval.append(rollout_traj)

            velocities = []
            for j in range(1, rollout_traj.shape[0]):
                velocities.append((rollout_traj[j] - rollout_traj[j-1]))
            velocities = np.asarray(velocities)
            full_velocity_eval.append(velocities)

            accelerations = []
            for j in range(1, velocities.shape[0]):
                accelerations.append((velocities[j] - velocities[j-1]))
            accelerations = np.asarray(accelerations)
            full_acceleration_eval.append(accelerations)

    particle_type = []
    position = []
    n_particles_per_example = []
    y = []
    for k in range(len(full_rollout_eval)):
        shape = full_rollout_eval[k].shape[1]
        print(f'Validation Dataset shape: {shape}')
        for i in range(num_augmented_eval_trajs):

            if(randomize == True):
                mesh_size = np.random.randint(int(rand_percent_lower*shape), int(rand_percent_higher*shape))
                points = sorted(list(random.sample(range(0, shape), mesh_size)))
            else:
                points = list(range(0, shape))
                
            sampled_tensor = torch.from_numpy(full_rollout_eval[k]).permute(1, 0, 2)
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
            for j in range(window, full_rollout_eval[k].shape[0]-1):
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


    shapes = [shape_material[dataset_name]]
    full_rollout = []
    full_velocity = []
    full_acceleration = []
    print('...Generating Rollout Data...')
    for i in range(1, 16):
        if(dataset_name == 'Plasticine'):
            shapes_path = datapath + f'plasticine_v{i}/shape/'
        else:
            shapes_path = datapath + f'shape_{i}/'
        for shape in shapes:
            print(f'Generating Rollout Trajectory {i} for {shape}_{dataset_name}')
            traj_path = shapes_path+shape+'/state/'
            rollout_traj = []
            for j in range(0, stop_time_step, jump_time_step):
                file = traj_path + str(j).zfill(4) + '.pt'
                tensor = torch.load(file)['x']
                rollout_traj.append(tensor.cpu().numpy())
            rollout_traj = np.asarray(rollout_traj)
            full_rollout.append(rollout_traj)

            velocities = []
            for j in range(1, rollout_traj.shape[0]):
                velocities.append((rollout_traj[j] - rollout_traj[j-1]))
            velocities = np.asarray(velocities)
            full_velocity.append(velocities)

            accelerations = []
            for j in range(1, velocities.shape[0]):
                accelerations.append((velocities[j] - velocities[j-1]))
            accelerations = np.asarray(accelerations)
            full_acceleration.append(accelerations)


    particle_type = []
    position = []
    n_particles_per_example = []
    y = []
    for i in range(len(full_rollout)):
        shape = full_rollout[i].shape[1]
        if(randomize == True):
            mesh_size = np.random.randint(int(rand_percent_lower*shape), int(rand_percent_higher*shape))
            points = sorted(list(random.sample(range(0, shape), mesh_size)))
        else:
            points = list(range(0, shape))
        position_tensor = []
        sampled_tensor = torch.from_numpy(full_rollout[i]).permute(1, 0, 2)
        sampled_tensor = sampled_tensor[points].cpu()
        #sampled_tensor = sampled_tensor.permute(1,0,2)
        position.append(sampled_tensor.numpy())
        
        p_type = (torch.ones(len(points))*5).numpy().astype(np.int32)
        n_particles_per_example.append(np.asarray([len(points)]).astype(np.int32))
        y.append(sampled_tensor[6].numpy())
        particle_type.append(p_type)

    print(f'Number of Rollout Trajectories: {position[0].shape}')

    train_dict = {'particle_type':particle_type, 'position':position, 'n_particles_per_example':n_particles_per_example, 'output':y}
    torch.save(train_dict, os.path.join(save_path, 'rollout.pt'))
    print("...Rollout Trajectories Saved...")
    if(save_midres == True):
        particle_type = []
        position = []
        n_particles_per_example = []
        y = []
        for i in range(len(full_rollout)):
            shape = full_rollout[i].shape[1]
            mesh_size = np.random.randint(int(0.60*shape), int(0.65*shape))
            points = sorted(list(random.sample(range(0, shape), mesh_size)))
            position_tensor = []
            sampled_tensor = torch.from_numpy(full_rollout[i]).permute(1, 0, 2)
            sampled_tensor = sampled_tensor[points].cpu()
            #sampled_tensor = sampled_tensor.permute(1,0,2)
            position.append(sampled_tensor.numpy())
            
            p_type = (torch.ones(len(points))*5).numpy().astype(np.int32)
            n_particles_per_example.append(np.asarray([len(points)]).astype(np.int32))
            y.append(sampled_tensor[6].numpy())
            particle_type.append(p_type)

        print(f'Saving Rollout Trajectories with 60% mesh: {position[0].shape}')

        train_dict = {'particle_type':particle_type, 'position':position, 'n_particles_per_example':n_particles_per_example, 'output':y}
        torch.save(train_dict, os.path.join(save_path, 'rollout_mid.pt'))

    if(save_diff_discs == True):
        particle_type = []
        position = []
        n_particles_per_example = []
        y = []
        traj = random.randint(1, num_trajectories-1)
        for i in range(1):
            shape = full_rollout[traj].shape[1]
            mesh_size = int(0.65*shape)
            for k in range(4):
                points = sorted(list(random.sample(range(0, shape), mesh_size)))
                position_tensor = []
                sampled_tensor = torch.from_numpy(full_rollout[i]).permute(1, 0, 2)
                sampled_tensor = sampled_tensor[points].cpu()
                #sampled_tensor = sampled_tensor.permute(1,0,2)
                position.append(sampled_tensor.numpy())
            
                p_type = (torch.ones(len(points))*5).numpy().astype(np.int32)
                n_particles_per_example.append(np.asarray([len(points)]).astype(np.int32))
                y.append(sampled_tensor[6].numpy())
                particle_type.append(p_type)

        print(f'Saving Rollout Trajectory 6 with 65% mesh and 4 Discretizations: {position[0].shape}')

        train_dict = {'particle_type':particle_type, 'position':position, 'n_particles_per_example':n_particles_per_example, 'output':y}
        torch.save(train_dict, os.path.join(save_path, 'rollout_different_discs.pt'))

    if(save_hires == True):
        particle_type = []
        position = []
        n_particles_per_example = []
        y = []
        for i in range(len(full_rollout)):
            shape = full_rollout[i].shape[1]
            mesh_size = np.random.randint(int(0.98*shape), int(0.99*shape))
            points = sorted(list(random.sample(range(0, shape), mesh_size)))
            position_tensor = []
            sampled_tensor = torch.from_numpy(full_rollout[i]).permute(1, 0, 2)
            sampled_tensor = sampled_tensor[points].cpu()
            #sampled_tensor = sampled_tensor.permute(1,0,2)
            position.append(sampled_tensor.numpy())
            
            p_type = (torch.ones(len(points))*5).numpy().astype(np.int32)
            n_particles_per_example.append(np.asarray([len(points)]).astype(np.int32))
            y.append(sampled_tensor[6].numpy())
            particle_type.append(p_type)

        print(f'Saving Rollout Trajectories with full mesh: {position[0].shape}')

        train_dict = {'particle_type':particle_type, 'position':position, 'n_particles_per_example':n_particles_per_example, 'output':y}
        torch.save(train_dict, os.path.join(save_path, 'rollout_full.pt'))