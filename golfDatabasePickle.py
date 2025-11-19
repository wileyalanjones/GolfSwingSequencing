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

# Read in Dataset from Pickle File
filename = "/Users/wileyjones/Desktop/CS435/GolfProject/GolfDB.pkl"
df = pd.read_pickle(filename)

print(df)
