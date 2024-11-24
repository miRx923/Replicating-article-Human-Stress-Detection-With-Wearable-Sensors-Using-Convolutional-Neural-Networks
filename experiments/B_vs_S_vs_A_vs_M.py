# File for running (B vs S vs A vs M) experiments
from model.train_n_test import experiments



# Run training and all experiments at once
data_files = ["./Data/WESAD_final/fft_normalized.pkl", 
              "./Data/WESAD_final/fft_cr_normalized.pkl", 
              "./Data/WESAD_final/fft_cr_cqt_normalized.pkl"
             ]

tasks = ["B_vs_S_vs_A_vs_M"]

experiments(data_files, tasks)
