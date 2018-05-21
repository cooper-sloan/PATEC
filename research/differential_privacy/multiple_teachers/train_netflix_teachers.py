import deep_recommender as dr
import tensorflow as tf
import torch
from DeepRecommender.reco_encoder.data import input_layer, new_input_layer
from DeepRecommender.reco_encoder.model import model
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn as nn
from torch.autograd import Variable
import copy
import time
from pathlib import Path
from math import sqrt, floor
import numpy as np
import os
from os import listdir, path

nb_teachers = 10

def train_teacher(nb_teachers, teacher_id):
  '''
    Very similar to code from DeepRecommender/run.py
  '''
  nf_data_dir = dr.config['path_to_train_data']
  nf_eval_data_dir = dr.config['path_to_eval_data']

  all_files = [path.join(nf_data_dir, f) for f in listdir(nf_data_dir)
                  if path.isfile(path.join(nf_data_dir, f)) and f.endswith('.txt')]
  chunk_size = floor(len(all_files)/nb_teachers)
  start = teacher_id*chunk_size
  chunk = all_files[start:start+chunk_size]

  params['src_files'] = chunk
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
                                            [int(l) for l in dr.config['hidden_layers'].split(',')],
                               nl_type=dr.config['non_linearity_type'],
                               is_constrained=dr.config['constrained'],
                               dp_drop_prob=dr.config['drop_prob'],
                               last_layer_activations=dr.config['skip_last_layer_nl'])
  os.makedirs(dr.config['logdir'], exist_ok=True)
  model_checkpoint = dr.config['logdir'] + "/model_%s_%s" % (nb_teachers,
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

  gpu_ids = [int(g) for g in dr.config['gpu_ids'].split(',')]
  print('Using GPUs: {}'.format(gpu_ids))
  if len(gpu_ids)>1:
    rencoder = nn.DataParallel(rencoder,
                               device_ids=gpu_ids)

  if use_gpu: rencoder = rencoder.cuda()

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

  for epoch in range(dr.config['num_epochs']):
    print('Doing epoch {} of {}'.format(epoch, dr.config['num_epochs']))
    e_start_time = time.time()
    rencoder.train()
    total_epoch_loss = 0.0
    denom = 0.0
    if dr.config['optimizer'] == "momentum":
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

      if i % dr.config['summary_frequency'] == 0:
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
    logger.scalar_summary("Training_RMSE_per_epoch", sqrt(total_epoch_loss/denom), epoch)
    logger.scalar_summary("Epoch_time", e_end_time - e_start_time, epoch)
    if epoch == dr.config['num_epochs'] - 1:
      eval_loss = do_eval(rencoder, eval_data_layer)
      print('Epoch {} EVALUATION LOSS: {}'.format(epoch, eval_loss))
      logger.scalar_summary("EVALUATION_RMSE", eval_loss, epoch)

  print("Saving model to {}".format(model_checkpoint + ".last"))
  torch.save(rencoder.state_dict(), model_checkpoint + ".last")

  return True

def main(argv=None):
  dr.load_maps()
  dr.load_src_data_layer()

  for i in range(nb_teachers):
    print("Training Teacher %s" % i)
    train_teacher(nb_teachers, i)
    print('-'*160)

  print("All Teachers Trained")

if __name__ == '__main__':
  main()
