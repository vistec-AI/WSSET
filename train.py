import os
import sys
import argparse
import numpy as np
import math
from tensorflow import keras
import tensorflow as tf
import tensorflow_addons as tfa

from data_processing import get_data_from_dataset_name
from model import get_model
from eval_metric import knn_evaluate
from loss import emd_loss, WSSET_loss

import csv

import datetime

import random


parser = argparse.ArgumentParser()
parser.add_argument(
        "--dsname",
        default="bbcsport",
    )
parser.add_argument(
        "--batchsize",
        default=32,
        type=int,
    )
parser.add_argument(
        "--trainsize",
        default=0.8,
        type=float,
    )
parser.add_argument(
        "--maxmember",
        default=350,
        type=int,
    )
parser.add_argument(
        "--lr",
        default=1e-5,
        type=float,
    )
parser.add_argument(
        "--margin",
        default=0.1,
        type=float,
    )
parser.add_argument(
        "--treshold",
        default=20,
        type=int,
    )
parser.add_argument(
        "--epochs",
        default=100,
        type=int,
    )
parser.add_argument(
        "--loss",
        default="gaussian",
    )
parser.add_argument(
        "--save-dir",
        default="saved_model/",
    )
parser.add_argument(
    "--time",
    default=datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
)
parser.add_argument(
        "--temperature",
        default=0.5,
        type=float,
    )
parser.add_argument("--istransfer", action="store_true")
parser.add_argument(
    "--loadmodel",
    default=None,
)
args = parser.parse_args()

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

P, test_idx, train_idx, val_idx, Y, train_dataset = get_data_from_dataset_name(args.dsname,args.trainsize,args.loss,args.batchsize)

transformer_num=5
head_num=7
feed_forward_dim=1000
dropout_rate=0.1

FILENAME="data/wmd/"+args.dsname+".csv"
reader = csv.reader(open(FILENAME, "r"), delimiter=";")
emd_list=list(reader)

model = get_model(transformer_num,head_num,feed_forward_dim,dropout_rate,P.shape[1],args.istransfer,args.loadmodel)
adam = keras.optimizers.Adam(learning_rate=args.lr)

log_dir=args.save_dir+args.dsname+"_"+args.loss+"_"+args.time

high_val =0.0
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
tensorboard_callback.set_model(model)

file_writer = tf.summary.create_file_writer(log_dir+ "/train")
file_writer.set_as_default()

#training loop
for epoch in range(args.epochs):
    losses=[]
    for (batch,(instance,labels)) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            logits = model(instance, training=True)
            # print(logits)
            # print(labels)
            if args.loss == "triplet":
                loss_value = tfa.losses.triplet_semihard_loss(labels,logits,args.margin)
                print(loss_value)
            elif args.loss == "approx_emd":
                loss_value = emd_loss(labels,logits,emd_list)
                print(loss_value)
            elif args.loss == "ce":
                loss_value = tf.keras.losses.sparse_categorical_crossentropy(labels, logits)
                print(loss_value)
            else:
                loss_value = WSSET_loss(labels, logits,args.loss,args.treshold,emd_list,args.margin)        
        grads = tape.gradient(loss_value, model.trainable_variables)
        adam.apply_gradients(zip(grads, model.trainable_variables))
        current_step=adam.iterations.numpy()
        losses.append(loss_value.numpy().mean())

    thisloss = np.mean(losses)
    print('Epoch {} finished'.format(epoch))

    #eval validation set
    results = model.predict(P[val_idx])
    acc = knn_evaluate(10,results,Y[val_idx])
    #eval test set
    results = model.predict(P[test_idx])
    acc2 = knn_evaluate(10,results,Y[test_idx])

    with file_writer.as_default():
        tf.summary.scalar('loss value', data=thisloss, step=epoch)
        tf.summary.scalar('val acc', data=acc, step=epoch)
        tf.summary.scalar('test acc', data=acc2, step=epoch)

    if acc2 >high_val:
        high_val=acc
        model.save_weights(log_dir+"/model_"+args.loss+"_"+args.time+".h5")



