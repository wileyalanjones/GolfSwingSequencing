from google.colab import drive
drive.mount('/content/drive')

import sys, os

PROJECT_DIR = "/content/drive/MyDrive/Colabs/GolfSwingSequencing"
sys.path.append(PROJECT_DIR)
os.chdir(PROJECT_DIR)

print("cwd:", os.getcwd())
print("files:", os.listdir(".")[:10])

!pip -q install "tensorflow>=2.12"

!pip -q install opencv-python-headless

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

print("Imports OK")

"""#PATH AND LOG SETUP"""

# Paths
BASE_DIR = os.getcwd()
FILENAME = os.path.join(BASE_DIR, "GolfDB.pkl")
VIDEO_DIR = "/content/videos_160"

print(VIDEO_DIR, FILENAME, BASE_DIR)

#logs
log_root = os.path.join(BASE_DIR, "logs")
os.makedirs(log_root, exist_ok=True)
run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = os.path.join(log_root, run_id)

# Models
model_root = os.path.join(BASE_DIR, "models")
os.makedirs(model_root, exist_ok=True)
filename_template = os.path.join(
    model_root,
    'weight_norm_loss_64_model_epoch_{epoch:02d}_val_event_auc_pr{val_event_auc_pr:.4f}.keras')

print("Paths, Models, and Logs OK")

"""# HYPERPARAMETER AND CALLBACKS"""

from tensorflow.keras.callbacks import ReduceLROnPlateau

# Hyperparameters and Constants
CLIP_LEN = 64
BATCH = 10
NUM_CLASSES = 9

#Callbacks
tensorboard_cb = callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=1,
    write_graph=True,
    write_images=False
)

checkpoint_callback = callbacks.ModelCheckpoint(
    filepath=filename_template,
    monitor='val_event_auc_pr',
    save_best_only=True,
    mode='max',
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_event_auc_pr',
    factor=0.2,
    patience=5,
    min_lr=0.001
)

print("Hyperparameters and Callbacks Good")

"""Copy videos to notebook
  - needed to stop CV2 from hanging when trying to access videos from drive directly
"""

# Copy Drive videos to local SSD
!rsync -a "/content/drive/MyDrive/Colabs/GolfSwingSequencing/videos_160/" "/content/videos_160/"

VIDEO_DIR = "/content/videos_160"
print("New VIDEO_DIR:", VIDEO_DIR)
print("Example files:", os.listdir(VIDEO_DIR)[:5])

"""TRAINING"""

# train test split
df = pd.read_pickle(FILENAME)
train, test = train_test_split(df, test_size=.25)

print(train, test)

# Make Dataset Objects
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

steps_per_epoch = len(train_obj) // BATCH
validation_steps = len(test_obj)

print("Training Datasets and Splits Good")

# Build Model
model_256_units_32_clip = Model.build_LSTM(clip_len=CLIP_LEN, units=256)

# Model Compile Boilerplate
model_256_units_32_clip.summary()
model_256_units_32_clip.compile(
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

# Loading Model from saved verison
import Metrics as mt

model_path = "/content/drive/MyDrive/Colabs/GolfSwingSequencing/models/model_epoch_01_val_event_auc_pr0.2501.keras"

loaded_model = tf.keras.models.load_model(
    model_path,
    custom_objects={
        "weighted_sparse_cce": mt.weighted_sparse_cce,
        "EventPrecision": mt.EventPrecision,
        "EventRecall": mt.EventRecall,
        "EventAUC": mt.EventAUC,
    }
)

# Logger Callback
from tensorflow.keras.callbacks import CSVLogger
logger_path = os.path.join(log_root, "CL128_U256_B2.csv")
csv_logger = CSVLogger(logger_path, append=True)

# Training Loop
# Make new dataset each time to get randomized clips and augments for each
# epoch. Inefficient but didn't have time to fix.
for i in range(50):
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

  print("Datasets Good")

  history= model_256_units_32_clip.fit(
      train_ds,
      validation_data=test_ds,
      epochs=1,
      verbose=1,
      steps_per_epoch=steps_per_epoch,
      validation_steps=validation_steps,
      callbacks=[tensorboard_cb, checkpoint_callback, csv_logger, reduce_lr],
      )

  print(f"epoch {i} complete")

# Reset learning rate after model reaches plateau
new_lr = 0.0001
model_256_units_32_clip.optimizer.learning_rate = new_lr

"""#NEW MODEL - CL32 B10 Loss: Focal"""

!pip install focal-loss

model_256_units_32_clip_focal = Model.build_LSTM(clip_len=CLIP_LEN, units=256)

from focal_loss import SparseCategoricalFocalLoss

alpha = [2.0] + [3.0]*8
gamma = 1.0

focal_loss = SparseCategoricalFocalLoss(
    gamma=gamma,
    class_weight=alpha,
    from_logits=False
)

model_256_units_32_clip_focal.compile(
    loss=focal_loss,
    metrics=[
            SparseCategoricalAccuracy(name="frame_accuracy"),
            mt.EventPrecision(),
            mt.EventRecall(),
            mt.EventAUC()
        ],
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    run_eagerly=False
    )

from tensorflow.keras.callbacks import CSVLogger
logger_path = os.path.join(log_root, "CL32_Focal_U256_B10_Reset.csv")
csv_logger = CSVLogger(logger_path, append=True)

for i in range(10):
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

  print("Datasets Good")

  history= model_256_units_32_clip_focal.fit(
      train_ds,
      validation_data=test_ds,
      epochs=1,
      verbose=1,
      steps_per_epoch=steps_per_epoch,
      validation_steps=validation_steps,
      callbacks=[tensorboard_cb, checkpoint_callback, csv_logger, reduce_lr],
      )

  print(f"epoch {i+1} complete\n")

"""#RETRAINING BEST 64 CLIP MODEL"""

# Hyperparameters and Constants
import Metrics as mt

CLIP_LEN = 64

model_path = "/content/drive/MyDrive/Colabs/GolfSwingSequencing/models/model_epoch_01_val_event_auc_pr0.3136.keras"

loaded_model = tf.keras.models.load_model(
    model_path,
    custom_objects={
        "weighted_sparse_cce": mt.weighted_sparse_cce,
        "EventPrecision": mt.EventPrecision,
        "EventRecall": mt.EventRecall,
        "EventAUC": mt.EventAUC,
    }
)

from tensorflow.keras.callbacks import CSVLogger
logger_path = os.path.join(log_root, "M256_B5.csv")
csv_logger = CSVLogger(logger_path, append=True)

new_lr = 0.0001
loaded_model.optimizer.learning_rate = new_lr

for i in range(20):
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

  print("Datasets Good")

  history= loaded_model.fit(
      train_ds,
      validation_data=test_ds,
      epochs=1,
      verbose=1,
      steps_per_epoch=steps_per_epoch,
      validation_steps=validation_steps,
      callbacks=[tensorboard_cb, checkpoint_callback, csv_logger, reduce_lr],
      )

  print(f"epoch {i+1} complete\n")

new_lr = 0.00001
loaded_model.optimizer.learning_rate = new_lr

for i in range(20):
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

  print("Datasets Good")

  history= loaded_model.fit(
      train_ds,
      validation_data=test_ds,
      epochs=1,
      verbose=1,
      steps_per_epoch=steps_per_epoch,
      validation_steps=validation_steps,
      callbacks=[tensorboard_cb, checkpoint_callback, csv_logger, reduce_lr],
      )

  print(f"epoch {i+1} complete\n")

"""#VALIDATION"""

# Best 64 clip model for validation
# Ignore the name of the path. I made a mistake when saving
# This uses the same custom weighted loss sparse crossentropy loss

import Metrics as mt
import GolfDataset as GD

model_path = "/content/drive/MyDrive/Colabs/GolfSwingSequencing/models/weight_norm_loss_64_model_epoch_01_val_event_auc_pr0.3311.keras"

loaded_model = tf.keras.models.load_model(
    model_path,
    custom_objects={
        "weighted_sparse_cce": mt.weighted_sparse_cce,
        "EventPrecision": mt.EventPrecision,
        "EventRecall": mt.EventRecall,
        "EventAUC": mt.EventAUC,
    }
)

val_obj = GD.FullVideoGolfDataset(
    test,
    VIDEO_DIR,
    )

val_ds = GD.make_full_video_tf_ds(val_obj)

PCE = mt.compute_dataset_pce_FIXED(loaded_model, val_ds, clip_len=64)

f = open("PCE_Results.txt", "w")
f.write(f"{PCE}")
f.close()

"""#Test Video"""

import Metrics as mt
import GolfDataset as GD

model_path = "/content/drive/MyDrive/Colabs/GolfSwingSequencing/models/weight_norm_loss_64_model_epoch_01_val_event_auc_pr0.3311.keras"

loaded_model = tf.keras.models.load_model(
    model_path,
    custom_objects={
        "weighted_sparse_cce": mt.weighted_sparse_cce,
        "EventPrecision": mt.EventPrecision,
        "EventRecall": mt.EventRecall,
        "EventAUC": mt.EventAUC,
    }
)

import TestVideo as tv
import importlib

importlib.reload(tv)

test_video = tv.TestVideo(
    "/content/drive/MyDrive/Colabs/GolfSwingSequencing/IMG_3226.MOV",
    target_size=(160, 160)) # Specify target size for cropping

test_video.frames

preds = tv.compute_event_frames(loaded_model, test_video.frames)

print(preds)

type(preds)

f = open("Pred_Frame_Results.txt", "w")
f.write(f"{preds}")
f.close()

import cv2
import matplotlib.pyplot as plt

test_video_path = "/content/drive/MyDrive/Colabs/GolfSwingSequencing/IMG_3226.MOV"

cap = cv2.VideoCapture(test_video_path)

if not cap.isOpened():
    print("Error opening video file")

for i, idx in enumerate(preds):
  cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
  ret, frame = cap.read()

  if not ret:
    break

  frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  plt.imshow(frame)
  plt.axis('off') # Add this line to turn off axes
  plt.show()

cap.release()

"""Confusion Matrix"""

!pip install scikit-learn

val_obj = GD.FullVideoGolfDataset(
    test,
    VIDEO_DIR,
    )

val_ds = GD.make_full_video_tf_ds(val_obj)

import TestVideo as tv
import importlib

importlib.reload(tv)

event_preds = []
true_labels = []

for frames, labels in val_ds:
  event_preds.append(tv.compute_all_frame_predictions(loaded_model, frames))
  true_labels.append(labels)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

event_preds = tf.concat(event_preds, axis=0)
true_labels = tf.concat(true_labels, axis=0)

cm = confusion_matrix(true_labels, event_preds, labels=range(9))
print(cm)

import matplotlib.pyplot as plt

disp = ConfusionMatrixDisplay(
    cm,
    display_labels=range(9),
)

disp.plot(cmap=plt.cm.Blues)
disp.ax_.set_title("Total Frame Prediction Confusion Matrix")
plt.show()