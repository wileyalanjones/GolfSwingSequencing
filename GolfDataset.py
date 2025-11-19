#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 16 17:35:49 2025

@author: wileyjones
"""

import Utils as ut
import tensorflow as tf
import numpy as np
import pandas as pd

AUTOTUNE = tf.data.AUTOTUNE

class GolfDataset:
    def __init__(
            self,
            dataframe,
            vid_dir,
            clip_len=64,
            num_events=8,
            train=True,
            repeat_per_video=2
            ):
       
        self.id_to_events = ut.get_events_dataframe(dataframe, num_events)
        self.vid_dir = vid_dir
        self.clip_len = clip_len
        self.num_events = num_events
        self.repeat_per_video = repeat_per_video
        self.train = train
    
        self.records = []
        for vid in self.id_to_events.keys():
            vid_path = f"{self.vid_dir}/{vid}.mp4"
            num_frames = ut.get_num_frames(vid_path)
            self.records.append({"id": vid, 
                                 "vid_path": vid_path,
                                 "num_frames": num_frames, 
                                 "events": self.id_to_events[vid]})
        
    def gen(self):
        """
        Yields
        ------
        frames : np.array
            frames as np.arrays of values.
        labels : np.array
            labels of events per frame.

        """
        for rec in self.records:
            idxs = ut.get_random_clip_indexes(rec['num_frames'], self.clip_len)
            frames = ut.get_frames_by_idxs(rec["vid_path"], idxs)
            labels = ut.get_clip_labels(idxs, rec['events'])
            if self.train:
                weights = ut.get_clip_weights(labels)
            else:
                weights = np.ones(labels.shape)
            yield frames, labels, weights
                
def make_tf_ds(
        dataset,
        batch_size=2,
        shuffle=True,
        augment=True,
        num_classes=9,
        clip_len=None
        ):
    
    if not clip_len:
        clip_len = dataset.clip_len
    
    output_sig = (
        tf.TensorSpec(shape=(clip_len, 160, 160, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(clip_len,), dtype=tf.int32),
        tf.TensorSpec(shape=(clip_len,), dtype=tf.float32)
        )
    
    ds = tf.data.Dataset.from_generator(dataset.gen, output_signature=output_sig)
    
    if shuffle:
        ds = ds.shuffle(buffer_size=8*batch_size, reshuffle_each_iteration=True)
    
    norm = tf.keras.layers.Rescaling(1./255)
    
    aug = tf.keras.Sequential([ 
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomZoom(.05),
        tf.keras.layers.RandomRotation(.05),
        ]) 
    
    def augment_frames(x, y, w):
        x = norm(x)
        if augment:
            x = aug(x)
        return x, y, w
    
    ds = ds.map(augment_frames, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(AUTOTUNE)
    
    return ds 

