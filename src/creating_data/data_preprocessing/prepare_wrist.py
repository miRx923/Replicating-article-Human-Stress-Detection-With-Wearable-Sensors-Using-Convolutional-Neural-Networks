import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "src")))

from utils import load_data, save_data, resample_signal_interpolation, convert_columns_to_int32



def process_subject(df_acc, df_bvp, df_eda_temp, subject_id):
    """
    Processes data for a single subject:
    - Resamples signals to 64 Hz
    - Combines signals into a single DataFrame
    """
    # Filter data for the given subject
    df_acc_subj = df_acc[df_acc['sid'] == subject_id].reset_index(drop=True)
    df_bvp_subj = df_bvp[df_bvp['sid'] == subject_id].reset_index(drop=True)
    df_eda_temp_subj = df_eda_temp[df_eda_temp['sid'] == subject_id].reset_index(drop=True)
    
    # Original sampling frequencies
    fs_acc = 32
    fs_eda_temp = 4
    
    # Resample to 64 Hz
    print(f"Resampling data for subject {subject_id}...")
    df_acc_subj_resampled = resample_signal_interpolation(df_acc_subj, 'w_acc_x', original_fs=fs_acc, target_fs=64)
    df_acc_subj_resampled_y = resample_signal_interpolation(df_acc_subj, 'w_acc_y', original_fs=fs_acc, target_fs=64)
    df_acc_subj_resampled_z = resample_signal_interpolation(df_acc_subj, 'w_acc_z', original_fs=fs_acc, target_fs=64)
    
    df_bvp_subj_resampled = df_bvp_subj['bvp'].values  # already at 64 Hz
    
    df_eda_subj_resampled = resample_signal_interpolation(df_eda_temp_subj, 'w_eda', original_fs=fs_eda_temp, target_fs=64)
    df_temp_subj_resampled = resample_signal_interpolation(df_eda_temp_subj, 'w_temp', original_fs=fs_eda_temp, target_fs=64)
    
    # Synchronize labels
    # Assume that labels in bvp are already at 64 Hz
    labels = df_bvp_subj['label'].values
    
    # Create a DataFrame for the subject
    df_subj = pd.DataFrame({
        'sid': subject_id, 'label': labels, 'w_acc_x': df_acc_subj_resampled, 'w_acc_y': df_acc_subj_resampled_y,
        'w_acc_z': df_acc_subj_resampled_z, 'bvp': df_bvp_subj_resampled, 'w_eda': df_eda_subj_resampled, 'w_temp': df_temp_subj_resampled
    })
    
    return df_subj


# Load data
df_acc = load_data("subj_merged_acc_w.pkl")
df_bvp = load_data('subj_merged_bvp_w.pkl')
df_eda_temp = load_data('subj_merged_eda_temp_w.pkl')

# Convert 'sid' and 'label' columns to integer types
df_acc = convert_columns_to_int32(df_acc, ['sid', 'label'])
df_bvp = convert_columns_to_int32(df_bvp, ['sid', 'label'])
df_eda_temp = convert_columns_to_int32(df_eda_temp, ['sid', 'label'])

# Process data for each subject
all_subjects_data = []
unique_sids = df_acc['sid'].unique()

for subject_id in unique_sids:
    df_subj = process_subject(df_acc, df_bvp, df_eda_temp, subject_id)
    all_subjects_data.append(df_subj)

# Combine data from all subjects
df_all = pd.concat(all_subjects_data, ignore_index=True)

# Save unfiltered merged_wrist.pkl
save_data(df_all, 'wrist.pkl')

os.remove('data/processed/subj_merged_acc_w.pkl')
os.remove('data/processed/subj_merged_bvp_w.pkl')
os.remove('data/processed/subj_merged_eda_temp_w.pkl')

print("\nwrist.pkl was successfully created.")
