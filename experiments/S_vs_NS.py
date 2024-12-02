# File for running (S vs NS) experiments
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "src")))

from model.train_n_test import experiments



# Run training and all experiments at once
data_files = ["data/processed/fft_normalized.pkl", 
              "data/processed/fft_cr_normalized.pkl", 
              "data/processed/fft_cr_cqt_normalized.pkl"
             ]

tasks = ["S_vs_NS"]

experiments(data_files, tasks)
