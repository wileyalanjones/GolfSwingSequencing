#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 19:54:32 2025

@author: wileyjones
"""

import pandas as pd
import numpy as np
import pickle 
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import os.path as osp


# Read in Dataset from Pickle File
filename = "/Users/wileyjones/Desktop/CS435/GolfProject/GolfDB.pkl"
df = pd.read_pickle(filename)

# Video Directory
vid_dir = "/Users/wileyjones/Desktop/CS435/GolfProject/videos_160"

# missing values 
columns = df.columns
print(columns)

# in DF
for col in columns:
    print(f"{col} has {df[col].isna().sum()} missing values")\
       
# In Video Links
for num in df['id']:
    if osp.exists(f"{vid_dir}/{num}.mp4"):
        print(f"ID {num} has a video")

# Angle Breakdown
sns.set_theme(style='whitegrid')
ax = sns.countplot(df, x='club', hue='club')
plt.show()

# Make sure every list has the right number of events
events = df['events']
for i, event in enumerate(events):
    if len(event) != 10:
        print(f"Event number {i} has {len(event)} events")
  
# Make sure event frame numbers are int
print(type(events[0][0]))
print(df.dtypes)
