import torch
import tensorflow.compat.v1 as tf
import numpy as np
from old_parsers.reading_utils import parse_serialized_simulation_example, split_trajectory
import json
import functools
import tensorflow_datasets as tfds
import os
import tree
import pickle
import random
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


def generate_data(dataset_name, dataset_root_dir, save_root_dir, randomize=False, use_random_seed=False,random_seed=42,
                  rand_percent_lower=0.05, rand_percent_higher=0.1, num_test_trajectories=5, num_train_trajectories=100,
                  max_rollout_trajectories=5, set_size_limit=False, max_particle_size=1000):
  
  assert rand_percent_lower<rand_percent_higher
  assert rand_percent_lower<=1.0
  assert rand_percent_higher<=1.0
  assert isinstance(random_seed, int)
  assert isinstance(num_train_trajectories, int)
  
  if(randomize == True and use_random_seed == True):
    random.seed(random_seed)
  datapath = os.path.join(dataset_root_dir, dataset_name)
  if(os.path.exists(datapath) == False):
    raise Exception("Invalid Dataset path")
      
  if(os.path.exists(save_root_dir) == False):
    raise Exception("Invalid Save Path")
  else:
    save_path = os.path.join(save_root_dir, dataset_name)
    if(os.path.exists(save_path) == False):
      os.mkdir(save_path)
    
  metadata = _read_metadata(datapath)
  
  ds = tf.data.TFRecordDataset([os.path.join(datapath, f'train.tfrecord')])
  
  ds = ds.map(functools.partial(parse_serialized_simulation_example, metadata=metadata))
  split_with_window = functools.partial(split_trajectory, window_length=7)
  ds = ds.flat_map(split_with_window)
  ds = ds.map(prepare_inputs)
  ds = tfds.as_numpy(ds)
  
  particle_type = []
  position = []
  n_particles_per_example = []
  y = []
  
  num_seen = 0
  for j, x in enumerate(iter(ds)):
    if(set_size_limit == True and x[0]['position'].shape[0]>max_particle_size):
      continue
    if(j == 0 or num_seen == 0):
      num_seen = 1
      cur_shape = x[0]['position'].shape[0]
      
      mesh_size = np.random.randint(int(rand_percent_lower*cur_shape), int(rand_percent_higher*cur_shape))
      points = sorted(list(random.sample(range(0, cur_shape), mesh_size)))
      
      traj_seen = 1
    if(x[0]['position'].shape[0]!=cur_shape):
      traj_seen +=1
      cur_shape = x[0]['position'].shape[0]
      mesh_size = np.random.randint(int(rand_percent_lower*cur_shape), int(rand_percent_higher*cur_shape))
      points = sorted(list(random.sample(range(0, cur_shape), mesh_size)))
      
    if(traj_seen > num_train_trajectories):
      break
    print(f'Loading Train Trajectory {traj_seen} for {dataset_name}')
    print(x[0]['position'].shape)
    print(x[0]['n_particles_per_example'])
    print(x[0]['particle_type'].shape)
    print(x[1].shape)
    if(randomize):
      x[0]['position'] = x[0]['position'][points]
      x[0]['n_particles_per_example'] = [len(points)]
      x[0]['particle_type'] = x[0]['particle_type'][points]
      y_new = x[1][points]
    else:
      y_new = x[1]

    particle_type.append(x[0]['particle_type'])
    position.append(x[0]['position'])
    n_particles_per_example.append(x[0]['n_particles_per_example'])
    y.append(y_new)
  print("....Training Data Generated....")
  print(f"Number of Trajectories: ", len(position))

  train_dict = {'particle_type':particle_type, 'position':position, 'n_particles_per_example':n_particles_per_example, 'output':y}
  with open(os.path.join(save_path, 'train.obj'), 'wb') as f:
    pickle.dump(train_dict, f)

  print("...Training Data Saved...")
  print("...Generating Validation Trajectories...")
  ds = tf.data.TFRecordDataset([os.path.join(datapath, 'test.tfrecord')])
  ds = ds.map(functools.partial(parse_serialized_simulation_example, metadata=metadata))

  split_with_window = functools.partial(split_trajectory, window_length=7)
  ds = ds.flat_map(split_with_window)
  ds = ds.map(prepare_inputs)
  ds = tfds.as_numpy(ds)


  particle_type = []
  position = []
  n_particles_per_example = []
  y = []
  num_seen = 0
  for j, x in enumerate(iter(ds)):
    if(set_size_limit == True and x[0]['position'].shape[0]>max_particle_size):
      continue
    if(j == 0 or num_seen == 0):
      num_seen = 1
      cur_shape = x[0]['position'].shape[0]
      mesh_size = np.random.randint(int(rand_percent_lower*cur_shape), int(rand_percent_higher*cur_shape))
      points = sorted(list(random.sample(range(0, cur_shape), mesh_size)))
      
      traj_seen = 1
    if(x[0]['position'].shape[0]!=cur_shape):
      traj_seen +=1
      cur_shape = x[0]['position'].shape[0]
      
      mesh_size = np.random.randint(int(rand_percent_lower*cur_shape), int(rand_percent_higher*cur_shape))
      points = sorted(list(random.sample(range(0, cur_shape), mesh_size)))
      
    if(traj_seen > num_test_trajectories):
      break
    
    print(f'Loading Validation Trajectory {traj_seen} for {dataset_name}')
    print(x[0]['position'].shape)
    print(x[0]['n_particles_per_example'])
    print(x[0]['particle_type'].shape)
    print(x[1].shape)
    
    if(randomize):
      x[0]['position'] = x[0]['position'][points]
      x[0]['n_particles_per_example'] = [len(points)]
      x[0]['particle_type'] = x[0]['particle_type'][points]
      y_new = x[1][points]
    else:
      y_new = x[1]


    particle_type.append(x[0]['particle_type'])
    position.append(x[0]['position'])
    n_particles_per_example.append(x[0]['n_particles_per_example'])
    y.append(y_new)

  train_dict = {'particle_type':particle_type, 'position':position, 'n_particles_per_example':n_particles_per_example, 'output':y}
  torch.save(train_dict, os.path.join(save_path, 'test.pt'))
  print('...Validation Data Saved...')

  ds = tf.data.TFRecordDataset([os.path.join(datapath, 'rollout.tfrecord')])
  ds = ds.map(functools.partial(parse_serialized_simulation_example, metadata=metadata))
  rollout_ds = ds.map(prepare_rollout_inputs)
  rollout_ds = tfds.as_numpy(rollout_ds)
  rollout_particle_type = []
  rollout_position = []
  rollout_n_particles_per_example = []
  rollout_y = []
  print('...Generating Rollout Data...')
  num_seen = 0
  for j, x in enumerate(iter(rollout_ds)):
    if(set_size_limit == True and x[0]['position'].shape[0]>max_particle_size):
      continue
    if(num_seen==max_rollout_trajectories):
      break
    print(f'Loading Rollout Trajectory {j} for {dataset_name}')
    num_seen += 1
    cur_shape = x[0]['position'].shape[0]
    mesh_size = np.random.randint(int(rand_percent_lower*cur_shape), int(rand_percent_higher*cur_shape))
    points = sorted(list(random.sample(range(0, cur_shape), mesh_size)))
    if(randomize):
      x[0]['position'] = x[0]['position'][points]
      x[0]['n_particles_per_example'] = [len(points)]
      x[0]['particle_type'] = x[0]['particle_type'][points]
      y_new = x[1][points]
    else:
      y_new = x[1]
    rollout_particle_type.append(x[0]['particle_type'])
    rollout_position.append(x[0]['position'])
    rollout_n_particles_per_example.append(x[0]['n_particles_per_example'])
    rollout_y.append(y_new)

  print(f'Number of Rollout Trajectories: {rollout_position[0].shape}')
  rollout_dict = {'particle_type':rollout_particle_type, 'position':rollout_position, 'n_particles_per_example':rollout_n_particles_per_example, 'output':rollout_y}
  torch.save(rollout_dict, os.path.join(save_path, 'rollout.pt'))
  print("...Rollout Trajectories Saved...")
