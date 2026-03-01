import tensorflow.compat.v1 as tf
import torch
import numpy as np
from reading_utils import parse_serialized_simulation_example, split_trajectory
import json
import functools
import tensorflow_datasets as tfds
import os
import tree
import random
from random import randint
def _read_metadata(path):
    with open(os.path.join(path, 'metadata.json'), 'rt') as fp:
        return json.loads(fp.read())

def batch_concat(dataset, batch_size):
  """We implement batching as concatenating on the leading axis."""

  # We create a dataset of datasets of length batch_size.
  windowed_ds = dataset.window(batch_size)

  # The plan is then to reduce every nested dataset by concatenating. We can
  # do this using tf.data.Dataset.reduce. This requires an initial state, and
  # then incrementally reduces by running through the dataset

  # Get initial state. In this case this will be empty tensors of the
  # correct shape.
  initial_state = tree.map_structure(
      lambda spec: tf.zeros(  # pylint: disable=g-long-lambda
          shape=[0] + spec.shape.as_list()[1:], dtype=spec.dtype),
      dataset.element_spec)

  # We run through the nest and concatenate each entry with the previous state.
  def reduce_window(initial_state, ds):
    return ds.reduce(initial_state, lambda x, y: tf.concat([x, y], axis=0))

  return windowed_ds.map(
      lambda *x: tree.map_structure(reduce_window, initial_state, x))

def prepare_rollout_inputs(context, features):
  """Prepares an inputs trajectory for rollout."""
  out_dict = {**context}
  # Position is encoded as [sequence_length, num_particles, dim] but the model
  # expects [num_particles, sequence_length, dim].
  pos = tf.transpose(features['position'], [1, 0, 2])
  # The target position is the final step of the stack of positions.
  target_position = pos[:, -1]
  # Remove the target from the input.
  out_dict['position'] = pos[:, :-1]
  # Compute the number of nodes
  out_dict['n_particles_per_example'] = [tf.shape(pos)[0]]
  if 'step_context' in features:
    out_dict['step_context'] = features['step_context']
  out_dict['is_trajectory'] = tf.constant([True], tf.bool)
  return out_dict, target_position

def prepare_inputs(tensor_dict):
  """Prepares a single stack of inputs by calculating inputs and targets.

  Computes n_particles_per_example, which is a tensor that contains information
  about how to partition the axis - i.e. which nodes belong to which graph.

  Adds a batch axis to `n_particles_per_example` and `step_context` so they can
  later be batched using `batch_concat`. This batch will be the same as if the
  elements had been batched via stacking.

  Note that all other tensors have a variable size particle axis,
  and in this case they will simply be concatenated along that
  axis.



  Args:
    tensor_dict: A dict of tensors containing positions, and step context (
    if available).

  Returns:
    A tuple of input features and target positions.

  """
  # Position is encoded as [sequence_length, num_particles, dim] but the model
  # expects [num_particles, sequence_length, dim].
  pos = tensor_dict['position']
  pos = tf.transpose(pos, perm=[1, 0, 2])

  # The target position is the final step of the stack of positions.
  target_position = pos[:, -1]

  # Remove the target from the input.
  tensor_dict['position'] = pos[:, :-1]

  # Compute the number of particles per example.
  num_particles = tf.shape(pos)[0]
  # Add an extra dimension for stacking via concat.
  tensor_dict['n_particles_per_example'] = num_particles[tf.newaxis]

  if 'step_context' in tensor_dict:
    # Take the input global context. We have a stack of global contexts,
    # and we take the penultimate since the final is the target.
    tensor_dict['step_context'] = tensor_dict['step_context'][-2]
    # Add an extra dimension for stacking via concat.
    tensor_dict['step_context'] = tensor_dict['step_context'][tf.newaxis]
  return tensor_dict, target_position

datapath = '/home/hviswan/Documents/new_dataset/WaterDropSmall'
metadata = _read_metadata(datapath)

# ds = tf.data.TFRecordDataset([os.path.join(datapath, 'waterdrop_train.tfrecord')])
# ds = ds.map(functools.partial(parse_serialized_simulation_example, metadata=metadata))
# rollout_ds = ds.map(prepare_rollout_inputs)
# rollout_ds = tfds.as_numpy(rollout_ds)
# rollout_particle_type = []
# rollout_position = []
# rollout_n_particles_per_example = []
# rollout_y = []

# for j, x in enumerate(iter(rollout_ds)):
#    rollout_particle_type.append(x[0]['particle_type'])
#    rollout_position.append(x[0]['position'])
#    rollout_n_particles_per_example.append(x[0]['n_particles_per_example'])
#    rollout_y.append(x[1])

# print(rollout_particle_type[0].shape)
#print(rollout_particle_type[0])

# rand_particle_types = []
# rand_positions = []
# rand_n_particles = []
# rand_y = []
# counter = 1
# for i in range(2):
#   for j in range(1):
#       if(j == 0):
#           cur_traj = rollout_particle_type[j].shape[0]
#           mesh_size =  np.random.randint(int(0.55*cur_traj), int(0.60*cur_traj))
#           points = sorted(list(random.sample(range(0, cur_traj), mesh_size)))
#       elif(rollout_particle_type[j].shape[0] != cur_traj):
#           counter += 1
#           if(counter >2):
#               break
#           cur_traj = rollout_particle_type[j].shape[0]
#           mesh_size =  np.random.randint(int(0.55*cur_traj), int(0.60*cur_traj))
#           points = sorted(list(random.sample(range(0, cur_traj), mesh_size)))
#       rand_particle_types.append(rollout_particle_type[j][points])
#       print(rollout_position[j][points].shape)
#       rand_positions.append(rollout_position[j][points])
#       rand_n_particles.append(np.asarray([len(points)]).astype(np.int32))
#       rand_y.append(rollout_y[j][points])
    
# train_dict = {'particle_type':rand_particle_types, 'position':rand_positions, 'n_particles_per_example':rand_n_particles, 'output':rand_y}
# torch.save(train_dict, '/home/hviswan/Documents/new_dataset/WaterDropSmall/WaterDropSample_rand_valid.pt')



ds = tf.data.TFRecordDataset([os.path.join(datapath, 'waterdrop_train.tfrecord')])
ds = ds.map(functools.partial(parse_serialized_simulation_example, metadata=metadata))

split_with_window = functools.partial(split_trajectory, window_length=7)
ds = ds.flat_map(split_with_window)
ds = ds.map(prepare_inputs)
#ds = batch_concat(ds, 32)
ds = tfds.as_numpy(ds)

counter = 0
particle_type = []
position = []
n_particles_per_example = []
y = []
num_trajectories = 1

for j, x in enumerate(iter(ds)):
  if(j == 0):
    cur_shape = x[0]['position'].shape[0]
    traj_seen = 1
  if(x[0]['position'].shape[0]!=cur_shape):
     traj_seen +=1
     cur_shape = x[0]['position'].shape[0]
  if(traj_seen > num_trajectories):
     break
  #print(x[0]['position'].shape)
  #print(x[0].keys())
  particle_type.append(x[0]['particle_type'])
  position.append(x[0]['position'])
  n_particles_per_example.append(x[0]['n_particles_per_example'])
  y.append(x[1])
  

#print(particle_type[0].shape)
#print(position[0].shape)
#print(particle_type[0])

print(len(position))
#print(n_particles_per_example)
#exit()
rand_particle_types = []
rand_positions = []
rand_n_particles = []
rand_y = []
for i in range(100):
    for j in range(len(particle_type)):
        if(j == 0):
            cur_traj = particle_type[j].shape[0]
            mesh_size =  np.random.randint(int(0.55*cur_traj), int(0.60*cur_traj))
            points = sorted(list(random.sample(range(0, cur_traj), mesh_size)))
        elif(particle_type[j].shape[0] != cur_traj):
            cur_traj = particle_type[j].shape[0]
            mesh_size =  np.random.randint(int(0.55*cur_traj), int(0.60*cur_traj))
            points = sorted(list(random.sample(range(0, cur_traj), mesh_size)))
        rand_particle_types.append(particle_type[j][points])
        #print(position[j][points].shape)
        rand_positions.append(position[j][points])
        rand_n_particles.append(np.asarray([len(points)]).astype(np.int32))
        rand_y.append(y[j][points])

print(rand_positions[0].shape)
pos_0 = torch.from_numpy(rand_positions[0]).permute(1, 0, 2)
pos_1 = torch.from_numpy(rand_positions[1]).permute(1, 0, 2)

rollout_particle_types = []
rollout_positions = []
rollout_n_particles = []
rollout_y = []
num_trajs = 1
cur_traj = rand_positions[0].shape[0]
print("CUR TRAJ = ", cur_traj)
positions = torch.from_numpy(rand_positions[0]).permute(1, 0, 2)
rollout_pos_array = [positions[0].numpy(), positions[1].numpy(), positions[2].numpy(), positions[3].numpy(), positions[4].numpy(), positions[5].numpy()]

for j in range(len(rand_positions)):
      if(rand_positions[j].shape[0]!=cur_traj):
         print("HERE")
         rollout_pos_array = torch.from_numpy(np.asarray(rollout_pos_array)).permute(1,0,2)
         rollout_positions.append(np.asarray(rollout_pos_array))
         rollout_particle_types.append(rand_particle_types[j-1])
         rollout_y.append(rand_y[j])
         rollout_n_particles.append(np.asarray(rollout_pos_array.shape[0]).astype(np.int32))
         num_trajs+=1
         cur_traj = rand_positions[j].shape[0]
         positions = torch.from_numpy(rand_positions[j]).permute(1,0,2)
         rollout_pos_array = [positions[0].numpy(), positions[1].numpy(), positions[2].numpy(), positions[3].numpy(), positions[4].numpy(), positions[5].numpy()]
      else:
        cur_pos = torch.from_numpy(rand_positions[j]).permute(1,0,2)
        rollout_pos_array.append(cur_pos[5].numpy())
      if(num_trajs>2):
         break
      
train_dict = {'particle_type':rollout_particle_types, 'position':rollout_positions, 'n_particles_per_example':rollout_n_particles, 'output':rollout_y}
torch.save(train_dict, '/home/hviswan/Documents/new_dataset/WaterDropSmall/WaterDropSample_rand_valid.pt')
print(len(rollout_positions))
print(rollout_positions[0].shape)



train_dict = {'particle_type':rand_particle_types, 'position':rand_positions, 'n_particles_per_example':rand_n_particles, 'output':rand_y}
torch.save(train_dict, '/home/hviswan/Documents/new_dataset/WaterDropSmall/WaterDropSample_rand_train.pt')

rand_particle_types = []
rand_positions = []
rand_n_particles = []
rand_y = []
for i in range(16):
    for j in range(len(particle_type)):
        if(j == 0):
            cur_traj = particle_type[j].shape[0]
            mesh_size =  np.random.randint(int(0.55*cur_traj), int(0.60*cur_traj))
            points = sorted(list(random.sample(range(0, cur_traj), mesh_size)))
        elif(particle_type[j].shape[0] != cur_traj):
            cur_traj = particle_type[j].shape[0]
            mesh_size =  np.random.randint(int(0.55*cur_traj), int(0.60*cur_traj))
            points = sorted(list(random.sample(range(0, cur_traj), mesh_size)))
        rand_particle_types.append(particle_type[j][points])
        print(position[j][points].shape)
        rand_positions.append(position[j][points])
        rand_n_particles.append(np.asarray([len(points)]).astype(np.int32))
        rand_y.append(y[j][points])
        
train_dict = {'particle_type':rand_particle_types, 'position':rand_positions, 'n_particles_per_example':rand_n_particles, 'output':rand_y}
torch.save(train_dict, '/home/hviswan/Documents/new_dataset/WaterDropSmall/WaterDropSample_rand_test.pt')


