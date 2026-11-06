import pandas as pd
import numpy as np
import pickle 
import cv2

# Read in Dataset from Pickle File
filename = "/Users/wileyjones/Desktop/CS435/GolfProject/GolfDB.pkl"
df = pd.read_pickle(filename)

print(df)
