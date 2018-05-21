# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import xrange

import netflix_aggregation as nagg
import input
import metrics
import deep_recommender as dr
import torch
import configparser
import utils
import argparse
from DeepRecommender.reco_encoder.data import input_layer, new_input_layer
from DeepRecommender.reco_encoder.model import model
from DeepRecommender.logger import Logger
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn as nn
from torch.autograd import Variable
import copy
import time
import pickle
from pathlib import Path
from math import sqrt, floor
import numpy as np
import os
from os import listdir, path
from random import shuffle


train_dir = './model_save/netflix'
teachers_dir = './model_save/netflix/'
nb_teachers = 10
lap_scale = 10

def ensemble_preds(nb_teachers):

  # Compute shape of array that will hold probabilities produced by each
  # teacher, for each training point, and each output class
  result_shape = (nb_teachers,
                  len(dr.stud_train_data_layer.data.keys()),
                  dr.stud_train_data_layer._vector_dim)

  # Create array that will hold result
  result = np.memmap('/data/Netflix/memmaps/results.dat', dtype=np.int8,
                     mode='w+',shape=result_shape)

  # Get predictions from each teacher
  for teacher_id in xrange(nb_teachers):
    # Compute path of checkpoint file for teacher model with ID teacher_id
    model_path = teachers_dir + '/model_' + str(nb_teachers) + '_' + str(teacher_id) + '.last'  # NOLINT(long-line)

    # Get predictions on our training data and store in result array
    result[teacher_id] = dr.teacher_preds(model_path, teacher_id)

    # This can take a while when there are a lot of teachers so output status
    print("Computed Teacher " + str(teacher_id) + " softmax predictions")

  return result

def iterate_one_epoch(data_layer, results):
  data = data_layer.data
  keys = list(data.keys())
  shuffle(keys)
  s_ind = 0
  e_ind = data_layer._batch_size
  while e_ind < len(keys):
    local_ind = 0
    inds1 = []
    inds2 = []
    vals = []
    for ind in range(s_ind, e_ind):
      inds2 += [v[0] for v in data[keys[ind]]]
      inds1 += [local_ind]*len([v[0] for v in data[keys[ind]]])
      vals += [v[1] for v in data[keys[ind]]]
      local_ind += 1

    i_torch = torch.LongTensor([inds1, inds2])
    v_torch = torch.FloatTensor(vals)

    mini_batch = torch.sparse.FloatTensor(i_torch, v_torch,
                                          torch.Size([data_layer._batch_size,
                                                      data_layer._vector_dim]))
    mini_batch_labels = torch.from_numpy(results[s_ind:e_ind])
#    mini_batch_labels = torch.tensor(results[s_ind:e_ind])
    s_ind += data_layer._batch_size
    e_ind += data_layer._batch_size
    yield (mini_batch, mini_batch_labels)

def train_student(nb_teachers):
  """
  This function trains a student using predictions made by an ensemble of
  teachers. The student and teacher models are trained using the same
  neural network architecture.
  :param nb_teachers: number of teachers (in the ensemble) to learn from
  :return: True if student training went well
  """
  assert input.create_dir_if_needed(train_dir)

  dr.load_maps()
  dr.load_train_data_layer()
  predictions = ensemble_preds(nb_teachers)
  #print("%s, %s, %s" % (nb_teachers, len(dr.stud_train_data_layer.data.keys()),
  #                      dr.stud_train_data_layer._vector_dim))

  #predictions = np.memmap('/data/Netflix/memmaps/results.dat', dtype=np.int8,
  #                        shape=(nb_teachers,
  #                               len(dr.stud_train_data_layer.data.keys()),
  #                               dr.stud_train_data_layer._vector_dim), mode='r')
  labels = nagg.noisy_max(predictions, lap_scale)

  #labels = np.memmap('/data/Netflix/memmaps/results_.dat', dtype=np.float32,
  #                   shape=(len(dr.stud_train_data_layer.data.keys()),
  #                          dr.stud_train_data_layer._vector_dim), mode='r')

  #IN THE ABOVE: it is recommended to run each one at a time - have predictions
  #save to its memmap file, then load it up in the next run to calculate labels.
  #Then again load the labels from file to carry on with the rest of training.
  #This is due to bugs in memory from trying to go directly from one step to the
  #next

  # Prepare checkpoint filename and path
  model_path = train_dir + '/' 'model_' + str(nb_teachers) + '_student.last'  # NOLINT(long-line)

  rencoder = model.AutoEncoder(layer_sizes=[dr.stud_train_data_layer._vector_dim] +
                                            [int(l) for l in dr.config['hidden_layers'].split(',')],
                               nl_type=dr.config['non_linearity_type'],
                               is_constrained=dr.config['constrained'],
                               dp_drop_prob=dr.config['drop_prob'],
                               last_layer_activations=dr.config['skip_last_layer_nl'])

  gpu_ids = [int(g) for g in dr.config['gpu_ids'].split(',')]
  print('Using GPUs: {}'.format(gpu_ids))
  if len(gpu_ids)>1:
    rencoder = nn.DataParallel(rencoder,
                               device_ids=gpu_ids)

  if dr.use_gpu: rencoder = rencoder.cuda()

  if dr.config['optimizer'] == "adam":
    optimizer = optim.Adam(rencoder.parameters(),
                           lr=dr.config['lr'],
                           weight_decay=dr.config['weight_decay'])
  elif dr.config['optimizer'] == "adagrad":
    optimizer = optim.Adagrad(rencoder.parameters(),
                              lr=dr.config['lr'],
                              weight_decay=dr.config['weight_decay'])
  elif dr.config['optimizer'] == "momentum":
    optimizer = optim.SGD(rencoder.parameters(),
                          lr=dr.config['lr'], momentum=0.9,
                          weight_decay=dr.config['weight_decay'])
    scheduler = MultiStepLR(optimizer, milestones=[24, 36, 48, 66, 72], gamma=0.5)
  elif dr.config['optimizer'] == "rmsprop":
    optimizer = optim.RMSprop(rencoder.parameters(),
                              lr=dr.config['lr'], momentum=0.9,
                              weight_decay=dr.config['weight_decay'])
  else:
    raise  ValueError('Unknown optimizer kind')

  t_loss = 0.0
  t_loss_denom = 0.0
  global_step = 0

  if dr.config['noise_prob'] > 0.0:
    dp = nn.Dropout(p=dr.config['noise_prob'])


  # Start student training

  for epoch in range(dr.config['num_epochs']):
    print('Doing epoch {} of {}'.format(epoch, dr.config['num_epochs']))
    e_start_time = time.time()
    rencoder.train()
    total_epoch_loss = 0.0
    denom = 0.0
    if dr.config['optimizer'] == "momentum":
      scheduler.step()

    num_batches = int(len(labels)/dr.config['batch_size'])
    for i, (mb, new_labels) in enumerate(iterate_one_epoch(dr.stud_train_data_layer, labels)):
      if i % 100 == 0:
        print("batch %s out of %s" % (i,num_batches))
      inputs = Variable(mb.cuda().to_dense() if dr.use_gpu else mb.to_dense())
      consensus = Variable(new_labels.cuda() if dr.use_gpu else new_labels)
      optimizer.zero_grad()
      outputs = rencoder(inputs)
      # define consensus
      loss, num_ratings = model.MSEloss(outputs, consensus)
      loss = loss / num_ratings
      loss.backward()
      optimizer.step()
      global_step += 1
      t_loss += torch.Tensor.item(loss.data)
      t_loss_denom += 1

      total_epoch_loss += torch.Tensor.item(loss.data)
      denom += 1

      #if dr.config['aug_step'] > 0 and i % dr.config['aug_step'] == 0 and i > 0:
      if dr.config['aug_step'] > 0:
        # Magic data augmentation trick happen here
        for t in range(dr.config['aug_step']):
          inputs = Variable(outputs.data)
          if dr.config['noise_prob'] > 0.0:
            inputs = dp(inputs)
          optimizer.zero_grad()
          outputs = rencoder(inputs)
          loss, num_ratings = model.MSEloss(outputs, inputs)
          loss = loss / num_ratings
          loss.backward()
          optimizer.step()
    e_end_time = time.time()
    print('Total epoch {} finished in {} seconds with TRAINING RMSE loss: {}'
          .format(epoch, e_end_time - e_start_time, sqrt(total_epoch_loss/denom)))

  torch.save(rencoder.state_dict(), model_path)
  print("STUDENT TRAINED")

  return True

def main(argv=None): # pylint: disable=unused-argument
  # Run student training according to values specified in flags
  dr.load_maps()
  dr.load_train_data_layer()
  assert train_student(nb_teachers)

if __name__ == '__main__':
  main()
