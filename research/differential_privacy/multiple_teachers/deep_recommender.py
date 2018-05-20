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
params['data_dir'] = config['path_to_train_data']
params['major'] = 'users'
params['itemIdInd'] = 1
params['userIdInd'] = 0
print("Loading Src Data")
with open('userIdMap.dict', 'rb') as f:
  userIdMap = pickle.load(f)
with open('itemIdMap.dict', 'rb') as f:
  itemIdMap = pickle.load(f)

#src_data_layer = input_layer.UserItemRecDataProvider(params=params)
#itemIdMap = src_data_layer.itemIdMap
#userIdMap = src_data_layer.userIdMap

#with open('userIdMap.dict', 'wb') as f:
#  userIdMap = pickle.dump(userIdMap, f)
#with open('itemIdMap.dict', 'wb') as f:
#  itemIdMap = pickle.dump(itemIdMap, f)
#print("Done saving maps")
#exit()


params['data_dir'] = config['path_to_stud_train_data']
stud_train_data_layer = input_layer.UserItemRecDataProvider(params=params,
                                                            user_id_map=userIdMap,
                                                            item_id_map=itemIdMap)
params['data_dir'] = config['path_to_stud_eval_data']
stud_eval_data_layer = input_layer.UserItemRecDataProvider(params=params,
                                                           user_id_map=userIdMap,
                                                           item_id_map=itemIdMap)
#stud_train_data_layer.src_data = src_data_layer.data
#stud_eval_data_layer.src_data = src_data_layer.data

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

def train_teacher(nb_teachers, teacher_id):
  '''
    Very similar to code from DeepRecommender/run.py
  '''
  nf_data_dir = config['path_to_train_data']
  nf_eval_data_dir = config['path_to_eval_data']

  all_files = [path.join(nf_data_dir, f) for f in listdir(nf_data_dir)
                  if path.isfile(path.join(nf_data_dir, f)) and f.endswith('.txt')]
  chunk_size = floor(len(all_files)/nb_teachers)
  start = teacher_id*chunk_size
  chunk = all_files[start:start+chunk_size]

  params = dict()
  params['batch_size'] = config['batch_size']
  params['src_files'] = chunk
  params['major'] = 'users'
  params['itemIdInd'] = 1
  params['userIdInd'] = 0
  print("Loading Training Data")
  data_layer = new_input_layer.UserItemRecDataProviderNew(params=params,
                                                          user_id_map=userIdMap,
                                                          item_id_map=itemIdMap)
  print("Data loaded")
  print("Total items found: {}".format(len(data_layer.data.keys())))
  print("Vector dim: {}".format(data_layer.vector_dim))

  print("Loading eval data")
  eval_params = copy.deepcopy(params)
  del eval_params['src_files']
  # must set eval batch size to 1 to make sure no examples are missed
  eval_params['data_dir'] = nf_eval_data_dir
  eval_data_layer = input_layer.UserItemRecDataProvider(params=eval_params,
                                                       user_id_map=userIdMap,
                                                       item_id_map=itemIdMap)

  eval_data_layer.src_data = src_data_layer.data

  rencoder = model.AutoEncoder(layer_sizes=[data_layer.vector_dim] +
                                            [int(l) for l in config['hidden_layers'].split(',')],
                               nl_type=config['non_linearity_type'],
                               is_constrained=config['constrained'],
                               dp_drop_prob=config['drop_prob'],
                               last_layer_activations=config['skip_last_layer_nl'])
  os.makedirs(config['logdir'], exist_ok=True)
  model_checkpoint = config['logdir'] + "/model_%s_%s" % (nb_teachers,
                                                          teacher_id)
  path_to_model = Path(model_checkpoint)
  if path_to_model.is_file():
    print("Loading model from: {}".format(model_checkpoint))
    rencoder.load_state_dict(torch.load(model_checkpoint))

  print('######################################################')
  print('######################################################')
  print('############# AutoEncoder Model: #####################')
  print(rencoder)
  print('######################################################')
  print('######################################################')

  gpu_ids = [int(g) for g in config['gpu_ids'].split(',')]
  print('Using GPUs: {}'.format(gpu_ids))
  if len(gpu_ids)>1:
    rencoder = nn.DataParallel(rencoder,
                               device_ids=gpu_ids)

  if use_gpu: rencoder = rencoder.cuda()

  if config['optimizer'] == "adam":
    optimizer = optim.Adam(rencoder.parameters(),
                           lr=config['lr'],
                           weight_decay=config['weight_decay'])
  elif config['optimizer'] == "adagrad":
    optimizer = optim.Adagrad(rencoder.parameters(),
                              lr=config['lr'],
                              weight_decay=config['weight_decay'])
  elif config['optimizer'] == "momentum":
    optimizer = optim.SGD(rencoder.parameters(),
                          lr=config['lr'], momentum=0.9,
                          weight_decay=config['weight_decay'])
    scheduler = MultiStepLR(optimizer, milestones=[24, 36, 48, 66, 72], gamma=0.5)
  elif config['optimizer'] == "rmsprop":
    optimizer = optim.RMSprop(rencoder.parameters(),
                              lr=config['lr'], momentum=0.9,
                              weight_decay=config['weight_decay'])
  else:
    raise  ValueError('Unknown optimizer kind')

  t_loss = 0.0
  t_loss_denom = 0.0
  global_step = 0

  if config['noise_prob'] > 0.0:
    dp = nn.Dropout(p=config['noise_prob'])

  for epoch in range(config['num_epochs']):
    print('Doing epoch {} of {}'.format(epoch, config['num_epochs']))
    e_start_time = time.time()
    rencoder.train()
    total_epoch_loss = 0.0
    denom = 0.0
    if config['optimizer'] == "momentum":
      scheduler.step()
    for i, mb in enumerate(data_layer.iterate_one_epoch()):
      inputs = Variable(mb.cuda().to_dense() if use_gpu else mb.to_dense())
      optimizer.zero_grad()
      outputs = rencoder(inputs)
      loss, num_ratings = model.MSEloss(outputs, inputs)
      loss = loss / num_ratings
      loss.backward()
      optimizer.step()
      global_step += 1
      t_loss += torch.Tensor.item(loss.data)
      t_loss_denom += 1

      if i % config['summary_frequency'] == 0:
        print('[%d, %5d] RMSE: %.7f' % (epoch, i, sqrt(t_loss / t_loss_denom)))
        logger.scalar_summary("Training_RMSE", sqrt(t_loss/t_loss_denom), global_step)
        t_loss = 0
        t_loss_denom = 0.0
        log_var_and_grad_summaries(logger, rencoder.encode_w, global_step, "Encode_W")
        log_var_and_grad_summaries(logger, rencoder.encode_b, global_step, "Encode_b")
        if not rencoder.is_constrained:
          log_var_and_grad_summaries(logger, rencoder.decode_w, global_step, "Decode_W")
        log_var_and_grad_summaries(logger, rencoder.decode_b, global_step, "Decode_b")

      total_epoch_loss += torch.Tensor.item(loss.data)
      denom += 1

      #if config['aug_step'] > 0 and i % config['aug_step'] == 0 and i > 0:
      if config['aug_step'] > 0:
        # Magic data augmentation trick happen here
        for t in range(config['aug_step']):
          inputs = Variable(outputs.data)
          if config['noise_prob'] > 0.0:
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
    logger.scalar_summary("Training_RMSE_per_epoch", sqrt(total_epoch_loss/denom), epoch)
    logger.scalar_summary("Epoch_time", e_end_time - e_start_time, epoch)
    if epoch == config['num_epochs'] - 1:
      eval_loss = do_eval(rencoder, eval_data_layer)
      print('Epoch {} EVALUATION LOSS: {}'.format(epoch, eval_loss))
      logger.scalar_summary("EVALUATION_RMSE", eval_loss, epoch)

  print("Saving model to {}".format(model_checkpoint + ".last"))
  torch.save(rencoder.state_dict(), model_checkpoint + ".last")

  return True

def softmax_preds(model_path):
  """
  Compute softmax activations (probabilities) with the model saved in the path
  specified as an argument
  :param images: a np array of images
  :param ckpt_path: a pytorch model checkpoint
  :param logits: if set to True, return logits instead of probabilities
  :return: probabilities (or logits if logits is set to True)
  """
  # Compute nb samples and deduce nb of batches
  data_length = len(stud_train_data_layer.data.keys())
  vector_length = stud_train_data_layer._vector_dim

  # Will hold the result
  preds = np.zeros((data_length, vector_length), dtype=np.float32)
  #num features varies depending on what items were in its training data
  rencoder = model.AutoEncoder(layer_sizes=[vector_length] +
                                            [int(l) for l in config['hidden_layers'].split(',')],
                               nl_type=config['non_linearity_type'],
                               is_constrained=config['constrained'],
                               dp_drop_prob=config['drop_prob'],
                               last_layer_activations=config['skip_last_layer_nl'])

  rencoder.load_state_dict(torch.load(model_path))
  rencoder.eval()

  # Parse data by batch
  for i, mb in enumerate(stud_train_data_layer.iterate_one_epoch(do_shuffle=False)):
    inputs = Variable(mb.cuda().to_dense() if use_gpu else mb.to_dense())
    start = i*config['batch_size']
    end = (i+1)*config['batch_size']

    # Prepare feed dictionary
    preds[start:end, :] = rencoder(inputs)

  return preds
