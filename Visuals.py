#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 20:56:52 2025

@author: wileyjones
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

CL16 = "/Users/wileyjones/Desktop/CS435/GolfSwingSequencing/logs/M256_B1_C16.csv"
CL32 = "/Users/wileyjones/Desktop/CS435/GolfSwingSequencing/logs/CL32_U256_B10.csv"
CL64 = "/Users/wileyjones/Desktop/CS435/GolfSwingSequencing/logs/M256_B5.csv"

df16 = pd.read_csv(CL16)
df32 = pd.read_csv(CL32)
df64 = pd.read_csv(CL64)

## Match Lengths of Datasets
df32_reduced = df32.iloc[-50:]
df64_reduced = df64.iloc[-50:]

# Make Label Category
df16["Model"] = "Clip_Length_16"
df32_reduced["Model"] = "Clip_Length_32"
df64_reduced["Model"] = "Clip_Length_64"

# Add epochs 
df16['epoch'] = range(1, 51)
df32_reduced['epoch'] = range(1,51)
df64_reduced['epoch'] = range(1,51)

## Combine
final_df = pd.concat([df16, df32_reduced, df64_reduced])
final_df

PCE = [0.5932142857142857,
       0.08857143, 
       0.66285714, 
       0.64857143, 
       0.53142857, 
       0.86, 
       0.89428571, 
       0.91714286, 
       0.14285714]

SEQUENCES = {"Total",
             "Address",
             "Toe-up",
             "Mid-backswing",
             "Top",
             "Mid-downswing",
             "Impact",
             "Mid-follow-through",
             "Finish"
    }

PCE_df = pd.DataFrame()
PCE_df["Sequence"] = SEQUENCES
PCE_df["PCE"] = PCE

FRAME_PREDS = [  6,  64,  66,  24,  86, 183, 184, 138]

sns.set_palette("deep")

# Event AUC
plot = sns.lineplot(
    x=final_df['epoch'], 
    y=final_df['val_event_auc_pr'], 
    hue=final_df["Model"],
    legend="brief",
    )
plot.set_ylim(0, .35)
plt.legend(fontsize='small')
plt.title("PR AUC per Model Tested")
plt.show()

# Frame Accuracy
plot = sns.lineplot(
    x=final_df['epoch'], 
    y=final_df['val_frame_accuracy'], 
    hue=final_df["Model"],
    legend="brief")
plt.legend(fontsize='small')
plt.title("Frame Accuracy per Model Tested")
plt.show()

# Event Precision
plot = sns.lineplot(
    x=final_df['epoch'], 
    y=final_df['val_event_precision'], 
    hue=final_df["Model"],
    legend="brief")
plt.legend(fontsize='small')
plt.title("Event Precision per Model Tested")
plt.show()

# Event Recall
plot = sns.lineplot(
    x=final_df['epoch'], 
    y=final_df['val_event_recall'], 
    hue=final_df["Model"],
    legend="brief")
plt.legend(fontsize='small')
plt.title("Event Recall per Model Tested")
plt.show()

#PCE 
ax = sns.barplot(
    PCE_df,
    x="PCE",
    y="Sequence",
    hue="PCE",
    legend=None)
ax.set_ylabel(None)
for bars in ax.containers:
    ax.bar_label(bars, fmt='%.2f')
ax.margins(x=.1)
plt.title("PCE Results")
plt.show()