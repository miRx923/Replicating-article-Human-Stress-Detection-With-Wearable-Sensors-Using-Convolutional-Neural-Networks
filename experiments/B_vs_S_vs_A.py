# File for running (B vs S vs A) experiments
from model.train_n_test import experiments



# Run training and all experiments at once
data_files = ["./Data/WESAD_final/fft_normalized.pkl", 
              "./Data/WESAD_final/fft_cr_normalized.pkl", 
              "./Data/WESAD_final/fft_cr_cqt_normalized.pkl"
             ]

tasks = ["B_vs_S_vs_A"]

experiments(data_files, tasks)
