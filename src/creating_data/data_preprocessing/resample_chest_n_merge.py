import pandas as pd
from math import gcd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "src")))

from utils import load_data, convert_columns_to_int32, resample_signal_poly, save_data, merge_chest_wrist, delete0567



def process_subject_chest(df_chest, subject_id, original_fs=700, target_fs=64):
    """
    Processes data for individual subjects:
    - Resamples signals to the target_fs
    - Combines signals into a single DataFrame
    """
    # Filter data for the given subject
    df_chest_subj = df_chest[df_chest['sid'] == subject_id].reset_index(drop=True)
    
    # Determine the resampling ratio
    g = gcd(original_fs, target_fs)
    up = target_fs // g
    down = original_fs // g
    
    # Resample all sensor columns
    df_chest_subj_resampled_acc1 = resample_signal_poly(df_chest_subj, 'acc1', up, down)
    df_chest_subj_resampled_acc2 = resample_signal_poly(df_chest_subj, 'acc2', up, down)
    df_chest_subj_resampled_acc3 = resample_signal_poly(df_chest_subj, 'acc3', up, down)
    df_chest_subj_resampled_ecg = resample_signal_poly(df_chest_subj, 'ecg', up, down)
    df_chest_subj_resampled_emg = resample_signal_poly(df_chest_subj, 'emg', up, down)
    df_chest_subj_resampled_eda = resample_signal_poly(df_chest_subj, 'eda', up, down)
    df_chest_subj_resampled_temp = resample_signal_poly(df_chest_subj, 'temp', up, down)
    df_chest_subj_resampled_resp = resample_signal_poly(df_chest_subj, 'resp', up, down)
    
    # Ensure all resampled signals have the same number of samples
    min_length = min(len(df_chest_subj_resampled_acc1), len(df_chest_subj_resampled_acc2),
                    len(df_chest_subj_resampled_acc3), len(df_chest_subj_resampled_ecg),
                    len(df_chest_subj_resampled_emg), len(df_chest_subj_resampled_eda),
                    len(df_chest_subj_resampled_temp), len(df_chest_subj_resampled_resp))
    
    # Truncate to the minimum length
    df_chest_subj_resampled_acc1 = df_chest_subj_resampled_acc1[:min_length]
    df_chest_subj_resampled_acc2 = df_chest_subj_resampled_acc2[:min_length]
    df_chest_subj_resampled_acc3 = df_chest_subj_resampled_acc3[:min_length]
    df_chest_subj_resampled_ecg = df_chest_subj_resampled_ecg[:min_length]
    df_chest_subj_resampled_emg = df_chest_subj_resampled_emg[:min_length]
    df_chest_subj_resampled_eda = df_chest_subj_resampled_eda[:min_length]
    df_chest_subj_resampled_temp = df_chest_subj_resampled_temp[:min_length]
    df_chest_subj_resampled_resp = df_chest_subj_resampled_resp[:min_length]
    
    # Adjust labels to the new length
    labels = df_chest_subj['label'].values[:min_length]
    
    # Create a DataFrame for the subject
    df_subj_chest = pd.DataFrame({
        'sid': [subject_id] * min_length, 'label': labels, 'acc1': df_chest_subj_resampled_acc1, 'acc2': df_chest_subj_resampled_acc2,
        'acc3': df_chest_subj_resampled_acc3, 'ecg': df_chest_subj_resampled_ecg, 'emg': df_chest_subj_resampled_emg,
        'eda': df_chest_subj_resampled_eda, 'temp': df_chest_subj_resampled_temp, 'resp': df_chest_subj_resampled_resp
    })
    
    return df_subj_chest


# Load chest data
df_chest = load_data('data/processed/chest.pkl')

# Convert 'sid' and 'label' columns to integer types
df_chest = convert_columns_to_int32(df_chest, ['sid', 'label'])

# Process data for each subject
all_subjects_chest_data = []
unique_sids = df_chest['sid'].unique()

for subject_id in unique_sids:
    df_subj_chest = process_subject_chest(df_chest, subject_id, original_fs=700, target_fs=64)
    all_subjects_chest_data.append(df_subj_chest)

# Combine data from all subjects
df_all_chest = pd.concat(all_subjects_chest_data, ignore_index=True)
save_data(df_all_chest, 'data/processed/chest_64hz.pkl')
os.remove('data/processed/chest.pkl')

print("\nSaved chest_64hz.pkl")

merge_chest_wrist()
print("Saved to 'all_data.pkl'")

delete0567()
print("Saved to 'all_data_final.pkl'")
