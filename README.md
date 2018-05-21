# PATEC

## Overview

This project is a fork of https://github.com/tensorflow/models/tree/master/research/differential_privacy/pate.
We extended the code to work with two new datasets, Netflix and Wiki.
The following files are of interest.

- input.py: data loading and processing
- train_teachers.py: trains an ensemble of teachers on a given datatset
- train_student.py: trains the student given the teacher models
- deep_cnn.py: CNN models for training MNIST and WIKI datasets
- data_prep.py, deep_recommender.py: Processes and handles Netflix data for training
- train_netflix_teachers.py: Training for Netflix ensemble
- train_netflix_students.py: Training for Netflix student model
- netflix_aggregation.py: Teacher aggregation for collaborative filtering
- eval_netflix.py: Evaluates the Netlflix model to determine test loss

![PATEC](patek.jpg)
