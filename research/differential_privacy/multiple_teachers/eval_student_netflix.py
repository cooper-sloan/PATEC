from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import xrange
import tensorflow as tf

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

nb_teachers = 10
student_dir = './model-save/netflix/'
teacher_id = 0
train = "student"

def eval_student(nb_teachers):
  model_path = student_dir + '/model_' + str(nb_teachers) + '_student.last'
  rencoder = model.AutoEncoder(layer_sizes=[dr.stud_train_data_layer._vector_dim] +
                                            [int(l) for l in dr.config['hidden_layers'].split(',')],
                               nl_type=dr.config['non_linearity_type'],
                               is_constrained=dr.config['constrained'],
                               dp_drop_prob=dr.config['drop_prob'],
                               last_layer_activations=dr.config['skip_last_layer_nl'])

  rencoder.load_state_dict(torch.load(model_path))
  rencoder.eval()
  if dr.use_gpu:
    rencoder.cuda()

  eval_loss = dr.do_eval(rencoder, dr.stud_eval_data_layer)
  print("EVALUATION LOSS: %s" % eval_loss)

def eval_teacher(nb_teachers, teacher_id):
  model_path = student_dir + '/model_' + str(nb_teachers) + '_' + str(teacher_id) + '.last'
  rencoder = model.AutoEncoder(layer_sizes=[dr.stud_train_data_layer._vector_dim] +
                                            [int(l) for l in dr.config['hidden_layers'].split(',')],
                               nl_type=dr.config['non_linearity_type'],
                               is_constrained=dr.config['constrained'],
                               dp_drop_prob=dr.config['drop_prob'],
                               last_layer_activations=dr.config['skip_last_layer_nl'])

  rencoder.load_state_dict(torch.load(model_path))
  rencoder.eval()
  if dr.use_gpu:
    rencoder.cuda()

  eval_loss = dr.do_eval(rencoder, dr.stud_eval_data_layer)
  print("EVALUATION LOSS: %s" % eval_loss)

def main(argv=None):
  dr.load_maps()
  dr.load_src_data_layer()
  dr.load_train_data_layer()
  dr.load_eval_data_layer()
  if train == "student":
    eval_student(nb_teachers)
  elif train == "teacher":
    eval_teacher(nb_teachers, teacher_id)

if __name__ == '__main__':
  main()
