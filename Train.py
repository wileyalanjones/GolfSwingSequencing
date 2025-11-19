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

# Paths
FILENAME = "/Users/wileyjones/Desktop/CS435/GolfProject/GolfDB.pkl"
VIDEO_DIR = "/Users/wileyjones/Desktop/CS435/GolfProject/videos_160"

# Hyperparameters
CLIP_LEN = 32
BATCH = 2
NUM_CLASSES = 9

# train test split 
df = pd.read_pickle(FILENAME)
train, test = train_test_split(df)

train_obj = GD.GolfDataset(train, VIDEO_DIR, clip_len=CLIP_LEN)
test_obj = GD.GolfDataset(test, VIDEO_DIR, clip_len=CLIP_LEN, train=False)

train_ds = GD.make_tf_ds(train_obj, batch_size=BATCH, clip_len=CLIP_LEN)
test_ds = GD.make_tf_ds(test_obj, batch_size=1, augment=False, clip_len=CLIP_LEN)

# Calculate steps per epoch based on your data frame split
steps_per_epoch_train = len(train) // BATCH
steps_per_epoch_val = len(test) // 1 

# Build Model 
model = Model.build_LSTM(clip_len=CLIP_LEN)

model.summary()

model.compile(
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
    optimizer='adam',
    run_eagerly=False
    )

history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=10,
    verbose=1,
    steps_per_epoch=steps_per_epoch_train,
    validation_steps=steps_per_epoch_val
    )