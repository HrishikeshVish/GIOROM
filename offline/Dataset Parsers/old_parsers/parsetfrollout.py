import tensorflow.compat.v1 as tf
import torch
import numpy as np
from reading_utils import parse_serialized_simulation_example, split_trajectory
import json
import functools
import tensorflow_datasets as tfds
import os
import tree
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


def parse(datasetName):
  datapath = f'/home/hviswan/Documents/new_dataset/{datasetName}'
  metadata = _read_metadata(datapath)
  """
  ds = tf.data.TFRecordDataset([os.path.join(datapath, f'train.tfrecord')])
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
  num_trajectories = 200
  num_seen = 0
  for j, x in enumerate(iter(ds)):
    if(x[0]['position'].shape[0]>1000):
      continue
    if(j == 0 or num_seen == 0):
      num_seen = 1
      cur_shape = x[0]['position'].shape[0]
      traj_seen = 1
    if(x[0]['position'].shape[0]!=cur_shape):
      traj_seen +=1
      cur_shape = x[0]['position'].shape[0]
    if(traj_seen > num_trajectories):
      break
    print(x[0]['position'].shape)
    print(x[0].keys())
    particle_type.append(x[0]['particle_type'])
    position.append(x[0]['position'])
    n_particles_per_example.append(x[0]['n_particles_per_example'])
    y.append(x[1])
  print(len(position))

  train_dict = {'particle_type':particle_type, 'position':position, 'n_particles_per_example':n_particles_per_example, 'output':y}
  torch.save(train_dict, f'/home/hviswan/Documents/new_dataset/{datasetName}/train.pt')

  ds = tf.data.TFRecordDataset([os.path.join(datapath, 'test.tfrecord')])
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

  for j, x in enumerate(iter(ds)):
    if(x[0]['position'].shape[0]>1000):
      continue
    print(x[0]['position'].shape)
    print(x[0].keys())
    particle_type.append(x[0]['particle_type'])
    position.append(x[0]['position'])
    n_particles_per_example.append(x[0]['n_particles_per_example'])
    y.append(x[1])

  train_dict = {'particle_type':particle_type, 'position':position, 'n_particles_per_example':n_particles_per_example, 'output':y}
  torch.save(train_dict, f'/home/hviswan/Documents/new_dataset/{datasetName}/test.pt')
  """

  ds = tf.data.TFRecordDataset([os.path.join(datapath, 'waterdrop_train.tfrecord')])
  ds = ds.map(functools.partial(parse_serialized_simulation_example, metadata=metadata))
  rollout_ds = ds.map(prepare_rollout_inputs)
  rollout_ds = tfds.as_numpy(rollout_ds)
  rollout_particle_type = []
  rollout_position = []
  rollout_n_particles_per_example = []
  rollout_y = []
  num_trajectories = 100
  counter = 0
  traj_seen = 0
  for j, x in enumerate(iter(rollout_ds)):
    print("NUM TRAJ = ", traj_seen)
    if(j == 0):
      num_seen = 1
      cur_shape = x[0]['position'].shape[0]
      traj_seen = 1
    if(x[0]['position'].shape[0]!=cur_shape):
      traj_seen +=1
      cur_shape = x[0]['position'].shape[0]
    if(traj_seen > num_trajectories):
      break
    rollout_particle_type.append(x[0]['particle_type'])
    rollout_position.append(x[0]['position'])
    rollout_n_particles_per_example.append(x[0]['n_particles_per_example'])
    rollout_y.append(x[1])
  
  rollout_dict = {'particle_type':rollout_particle_type, 'position':rollout_position, 'n_particles_per_example':rollout_n_particles_per_example, 'output':rollout_y}
  torch.save(rollout_dict, f'/home/hviswan/Documents/new_dataset/{datasetName}/long_rollout.pt')



parse('WaterDropSmall')

   #break
#print(counter)
#it = next(iter(ds))
#print(it[0]['position'].shape)
#print(it[1].shape)
