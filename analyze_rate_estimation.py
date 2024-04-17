# coding:utf-8
import time
import sys
import numpy as np
import gc
import os
import datetime
from config import *
from collections import Counter
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

SAVE_DIR = os.path.join("other_records", "240412_rate")

if __name__ == "__main__":
    rate_df = pd.read_csv(os.path.join(SAVE_DIR, "estimated_rate.csv"), index_col=0)
    print(rate_df)

    N_arr = np.loadtxt(os.path.join(SAVE_DIR, "N_arr_extracted.csv"), delimiter=",")
    n_arr = np.loadtxt(os.path.join(SAVE_DIR, "n_arr_extracted(1).csv"), delimiter=",")
    print(N_arr)
    print(n_arr)
    print(N_arr.shape, n_arr.shape)

    print(np.sum(N_arr, axis=1))
    print(np.sum(n_arr, axis=1) / np.sum(N_arr, axis=1))

    win_rate_arr = np.where(N_arr != 0, n_arr / N_arr, np.nan)
    win_rate_df = pd.DataFrame(win_rate_arr, index=rate_df['AI_id'], columns=rate_df['AI_id'])
    win_rate_df.to_csv(os.path.join(SAVE_DIR, "win_rate.csv"), na_rep='')
    #np.savetxt(os.path.join(SAVE_DIR, "win_rate.csv"), win_rate_arr, delimiter=",")

    analysis_df = rate_df.copy()

    analysis_df["game num"] = np.sum(N_arr, axis=1)
    analysis_df["win rate"] = np.sum(n_arr, axis=1) / np.sum(N_arr, axis=1)
    analysis_df.to_csv(os.path.join(SAVE_DIR, "analysis.csv"))
