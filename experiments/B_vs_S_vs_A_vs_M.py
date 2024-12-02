# File for running (B vs S vs A vs M) experiments
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "src")))

from model.train_n_test import experiments



# Run training and all experiments at once
data_files = ["data/processed/fft_normalized.pkl", 
              "data/processed/fft_cr_normalized.pkl", 
              "data/processed/fft_cr_cqt_normalized.pkl"
             ]

tasks = ["B_vs_S_vs_A_vs_M"]

experiments(data_files, tasks)
