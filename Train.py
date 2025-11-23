#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 16 18:53:25 2025

@author: wileyjones
"""

import Model
import GolfDataset as GD
from sklearn.model_selection import train_test_split
from tensorflow.keras import callbacks
import tensorflow as tf
import pandas as pd
import numpy as np
import Metrics as mt
from tensorflow.keras.metrics import SparseCategoricalAccuracy
import os
from datetime import datetime

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILENAME = os.path.join(BASE_DIR, "GolfDB.pkl")
VIDEO_DIR = os.path.join(BASE_DIR, "videos_160")

#logs
log_root = os.path.join(BASE_DIR, "logs")
os.makedirs(log_root, exist_ok=True)
run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = os.path.join(log_root, run_id)

# Hyperparameters
CLIP_LEN = 32
BATCH = 1
NUM_CLASSES = 9

#Callbacks
tensorboard_cb = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=1,   
    write_graph=True,
    write_images=False
)

# train test split 
df = pd.read_pickle(FILENAME)
train, test = train_test_split(df)

train_obj = GD.GolfDataset(
    train, 
    VIDEO_DIR, 
    clip_len=CLIP_LEN, 
    repeat_per_video=1
    )

test_obj = GD.GolfDataset(
    test, 
    VIDEO_DIR, 
    clip_len=CLIP_LEN, 
    train=False,
    repeat_per_video=1)

train_ds = GD.make_tf_ds(
    train_obj, 
    batch_size=BATCH, 
    clip_len=CLIP_LEN
    )

test_ds = GD.make_tf_ds(
    test_obj, 
    batch_size=1, 
    augment=False, 
    shuffle=False, 
    clip_len=CLIP_LEN)

steps_per_epoch = len(train_obj) // BATCH        
validation_steps = len(test_obj) 

# Build Model 
model = Model.build_LSTM(clip_len=CLIP_LEN)

model.summary()
model.compile(
    loss=mt.weighted_sparse_cce,
    metrics=[
            SparseCategoricalAccuracy(name="frame_accuracy"),
            mt.EventPrecision(),
            mt.EventRecall(),
            mt.EventAUC()
        ],
    optimizer='adam',
    run_eagerly=False
    )


history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=100,
    verbose=1,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    callbacks=[tensorboard_cb]
    )


