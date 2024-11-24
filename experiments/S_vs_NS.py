# File for running (S vs NS) experiments
from model.train_n_test import experiments



# Run training and all experiments at once
data_files = ["./Data/WESAD_final/fft_normalized.pkl", 
              "./Data/WESAD_final/fft_cr_normalized.pkl", 
              "./Data/WESAD_final/fft_cr_cqt_normalized.pkl"
             ]

tasks = ["S_vs_NS"]

experiments(data_files, tasks)
