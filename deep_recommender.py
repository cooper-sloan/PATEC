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
import gc
from os import listdir, path

config = configparser.ConfigParser()
config.read('DeepRecommender.cfg')
config = dict(config['DEFAULT'])
config['batch_size'] = int(config['batch_size'])
config['drop_prob'] = float(config['drop_prob'])
config['lr'] = float(config['lr'])
config['weight_decay'] = float(config['weight_decay'])
config['aug_step'] = int(config['aug_step'])
config['noise_prob'] = float(config['noise_prob'])
config['num_epochs'] = int(config['num_epochs'])
config['summary_frequency'] = int(config['summary_frequency'])
config['constrained'] = bool(config['constrained'])
config['skip_last_layer_nl'] = bool(config['skip_last_layer_nl'])
print(config)

params = dict()
params['batch_size'] = config['batch_size']
params['major'] = 'users'
params['itemIdInd'] = 1
params['userIdInd'] = 0

maps_loaded = False
userIdMap = None
itemIdMap = None
src_data_layer_loaded = False
src_data_layer = None
stud_train_data_layer_loaded = False
stud_train_data_layer = None
stud_eval_data_layer_loaded = False
stud_eval_data_layer = None

def load_maps():
  global maps_loaded, userIdMap, itemIdMap
  if maps_loaded:
    print("Maps already loaded!")
    return None

  if os.path.isfile('maps/userIdMap.dict'):
    with open('maps/userIdMap.dict', 'rb') as f:
      userIdMap = pickle.load(f)
    with open('maps/itemIdMap.dict', 'rb') as f:
      itemIdMap = pickle.load(f)
    maps_loaded = True
  else:
    print("maps not saved")

def load_src_data_layer(with_maps=False, save_maps=False):
  global src_data_layer_loaded, src_data_layer, userIdMap, itemIdMap
  if src_data_layer_loaded:
    print("src_data_layer already loaded!")
    return None

  params['data_dir'] = config['path_to_train_data']

  if with_maps:
    src_data_layer = input_layer.UserItemRecDataProvider(params=params,
                                                         item_id_map = itemIdMap,
                                                         user_id_map = userIdMap)
  else:
    src_data_layer = input_layer.UserItemRecDataProvider(params=params)
    userIdMap = src_data_layer.userIdMap
    itemIdMap = src_data_layer.itemIdMap
    maps_loaded = True

    if save_maps:
      with open('maps/userIdMap.dict', 'wb') as f:
        userIdMap = pickle.dump(userIdMap, f)
      with open('maps/itemIdMap.dict', 'wb') as f:
        itemIdMap = pickle.dump(itemIdMap, f)
      print("Done saving maps")

  src_data_layer_loaded = True


def load_train_data_layer():
  global stud_train_data_layer_loaded, stud_train_data_layer
  if stud_train_data_layer_loaded:
    print("stud_train_data_layer already loaded!")
    return None

  params['data_dir'] = config['path_to_stud_train_data']
  if not (userIdMap and itemIdMap):
    print("maps not loaded; please load maps")
    return None
  else:
    stud_train_data_layer = input_layer.UserItemRecDataProvider(params=params,
                                                                user_id_map=userIdMap,
                                                                item_id_map=itemIdMap)
    stud_train_data_layer_loaded = True

    if src_data_layer:
      stud_train_data_layer.src_data = src_data_layer.data

def load_eval_data_layer():
  global stud_eval_data_layer_loaded, stud_eval_data_layer
  if stud_eval_data_layer_loaded:
    print("stud_eval_data_layer already loaded!")
    return None

  params['data_dir'] = config['path_to_stud_eval_data']
  if not (userIdMap and itemIdMap):
    print("maps not loaded; please load maps")
    return None
  else:
    stud_eval_data_layer = input_layer.UserItemRecDataProvider(params=params,
                                                               user_id_map=userIdMap,
                                                               item_id_map=itemIdMap)
    stud_eval_data_layer_loaded = True
    if src_data_layer:
      stud_eval_data_layer.src_data = src_data_layer.data


logger = Logger(config['logdir'])

use_gpu = torch.cuda.is_available() # global flag
if use_gpu:
    print('GPU is available.')
else:
    print('GPU is not available.')

def do_eval(encoder, evaluation_data_layer):
  encoder.eval()
  denom = 0.0
  total_epoch_loss = 0.0
  for i, (eval, src) in enumerate(evaluation_data_layer.iterate_one_epoch_eval()):
    inputs = Variable(src.cuda().to_dense() if use_gpu else src.to_dense())
    targets = Variable(eval.cuda().to_dense() if use_gpu else eval.to_dense())
    outputs = encoder(inputs)
    loss, num_ratings = model.MSEloss(outputs, targets)
    total_epoch_loss += torch.Tensor.item(loss.data)
    denom += torch.Tensor.item(num_ratings.data)
  return sqrt(total_epoch_loss / denom)

def log_var_and_grad_summaries(logger, layers, global_step, prefix, log_histograms=False):
  """
  Logs variable and grad stats for layer. Transfers data from GPU to CPU automatically
  :param logger: TB logger
  :param layers: param list
  :param global_step: global step for TB
  :param prefix: name prefix
  :param log_histograms: (default: False) whether or not log histograms
  :return:
  """
  for ind, w in enumerate(layers):
    # Variables
    w_var = w.data.cpu().numpy()
    logger.scalar_summary("Variables/FrobNorm/{}_{}".format(prefix, ind), np.linalg.norm(w_var),
                          global_step)
    if log_histograms:
      logger.histo_summary(tag="Variables/{}_{}".format(prefix, ind), values=w.data.cpu().numpy(),
                           step=global_step)

    # Gradients
    w_grad = w.grad.data.cpu().numpy()
    logger.scalar_summary("Gradients/FrobNorm/{}_{}".format(prefix, ind), np.linalg.norm(w_grad),
                          global_step)
    if log_histograms:
      logger.histo_summary(tag="Gradients/{}_{}".format(prefix, ind), values=w.grad.data.cpu().numpy(),
                         step=global_step)

def teacher_preds(model_path, teacher_id):

  # Compute nb samples and deduce nb of batches
  data_length = len(stud_train_data_layer.data.keys())
  vector_length = stud_train_data_layer._vector_dim
  print("shape: (%s, %s)" %(data_length, vector_length))

  # Will hold the tmp result
  preds = np.memmap('/data/Netflix/memmaps/preds_%s.dat.tmp' % teacher_id,
                    dtype=np.float32, mode='w+', shape=(data_length,
                                                        vector_length))

  #num features varies depending on what items were in its training data
  rencoder = model.AutoEncoder(layer_sizes=[vector_length] +
                                            [int(l) for l in config['hidden_layers'].split(',')],
                               nl_type=config['non_linearity_type'],
                               is_constrained=config['constrained'],
                               dp_drop_prob=config['drop_prob'],
                               last_layer_activations=config['skip_last_layer_nl'])

  rencoder.load_state_dict(torch.load(model_path))
  rencoder.eval()
  if use_gpu:
    rencoder.cuda()

  # Parse data by batch
  for i, mb in enumerate(stud_train_data_layer.iterate_one_epoch(do_shuffle=False)):
    inputs = Variable(mb.cuda().to_dense() if use_gpu else mb.to_dense())
    start = i*config['batch_size']
    end = (i+1)*config['batch_size']

    # Prepare feed dictionary
    preds[start:end, :] = rencoder(inputs).cpu().detach().numpy()

  final_preds = np.memmap('/data/Netflix/memmaps/preds_%s.dat' % teacher_id,
                         dtype=np.int8, mode='w+', shape=(data_length,
                                                          vector_length))
  final_preds[:,:] = np.rint(preds.clip(min=1, max=5))
  del preds
  gc.collect()
  os.remove('/data/Netflix/memmaps/preds_%s.dat.tmp' % teacher_id)

  return final_preds
