# Various utilities
import math
import numpy as np
import pandas as pd
from scipy.signal import resample_poly
import pickle
import os
from sklearn.preprocessing import StandardScaler



def load_data(file_path):
    """
    Loads data from pickle file.
    """
    df = pd.read_pickle(file_path)
    print(f"Načítané dáta z {file_path} s tvarom {df.shape}")

    return df


def save_data(obj, file_path):
    """
    Uloží objekt do pickle súboru.
    """
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)


def resample_signal_interpolation(df, signal_column, original_fs, target_fs=64):
    """
    Resampluje signál z original_fs na target_fs.
    """
    # Vytvorenie časovej osi
    duration_sec = len(df) / original_fs
    original_time = np.linspace(0, duration_sec, len(df))
    target_length = int(duration_sec * target_fs)
    target_time = np.linspace(0, duration_sec, target_length)
    
    # Resamplovanie pomocou interpolácie
    resampled_signal = np.interp(target_time, original_time, df[signal_column].values)
    return resampled_signal


def convert_columns_to_int32(df, columns):
    """
    Skonvertuje špecifikované stĺpce na int32.
    """
    for col in columns:
        df[col] = df[col].astype('int32')
    return df


def resample_signal_poly(df, signal_column, up, down):
    """
    Resampluje signál pomocou resample_poly.
    """
    resampled_signal = resample_poly(df[signal_column].values, up, down)
    return resampled_signal


def filter_labels(df_all):
    """
    Deletes rows with labels 0, 5, 6, 7.
    """
    df_filtered = df_all[~df_all['label'].isin([0, 5, 6, 7])]

    return df_filtered


def merge_chest_wrist():
    """
    Concatenates chest and wrist dataframes
    """
    df_chest = load_data('../../../data/processed/chest_64hz.pkl')
    df_wrist = load_data('../../../data/processed/wrist.pkl')

    # Odstránenie 'sid' a 'label' z oboch datasetov na zabránenie duplicít pri zlučovaní
    df_chest_merged = df_chest.drop(['sid', 'label'], axis=1)
    df_wrist_merged = df_wrist.drop(['sid', 'label'], axis=1)

    # Spojenie dát horizontálne (stĺpcami)
    df_all = pd.concat([df_chest_merged, df_wrist_merged], axis=1)

    # Pridanie 'sid' a 'label' z chest dát späť do zlúčeného DataFrame
    df_all.insert(0, 'sid', df_chest['sid'].values)
    df_all.insert(1, 'label', df_wrist['label'].values)

    save_data(df_all, "../../../data/processed/all_data.pkl")

    os.remove("../../../data/processed/chest_64hz.pkl")
    os.remove("../../../data/processed/wrist.pkl")


def delete0567():
    df_all = load_data("../../../data/processed/all_data.pkl")
    df_filtered = filter_labels(df_all)
    save_data(df_filtered, "../../../data/processed/all_data_final.pkl")
    os.remove("../../../data/processed/all_data.pkl")


def compute_ci(metric, n_samples):
    """
    Computes the 95% confidence interval (CI) for a given metric (Accuracy or F1-score).
    :param metric: Accuracy or F1-score in percentage
    :param n_samples: Number of test samples
    :return: Confidence Interval (CI) (95%)
    """
    return 1.96 * math.sqrt((metric * (100 - metric)) / n_samples)


def modify_labels(y, task):
    """
    Filters the needed labels based on the task.
    :param y: Original labels
    :param task: Task name
    :return: Modified labels
    """
    if task == "S_vs_NS":
        return np.where((y == 1) | (y == 3), 0, 1)  # Non-Stress: 0, Stress: 1
    elif task == "B_vs_S_vs_A":
        return y - 1 # Baseline: 0, Stress: 1, Amusement: 2
    elif task == "B_vs_S_vs_A_vs_M":
        return y - 1 # Baseline: 0, Stress: 1, Amusement: 2, Meditation: 3


def normalize_data(input_path, output_path):
    """
    Function to normalize the data per signal and per subject.
    :param input_path: Path to the input .pkl file.
    :param output_path: Path to save the normalized data.
    """
    # Load data
    data = pd.read_pickle(input_path)
    X = data['X']
    y = data['y']
    sid = data['sid']

    # Get the shape of X
    num_samples, num_signals, num_features = X.shape

    # Initialize arrays to hold the normalized data
    X_normalized = np.zeros_like(X)

    # Get unique subject IDs
    unique_sids = np.unique(sid)

    # For each subject, perform normalization
    for test_sid in unique_sids:
        # Identify test and train indices
        test_indices = np.where(sid == test_sid)[0]
        train_indices = np.where(sid != test_sid)[0]

        # Perform per-signal normalization using training data
        for i in range(num_signals):
            # Extract training data for the i-th signal
            Xi_train = X[train_indices, i, :]  # Shape: (num_train_samples, num_features)

            # Initialize the scaler and fit on training data
            scaler = StandardScaler()
            scaler.fit(Xi_train)

            # Transform both training and test data
            Xi_train_normalized = scaler.transform(Xi_train)
            Xi_test_normalized = scaler.transform(X[test_indices, i, :])

            # Store the normalized data back into X_normalized
            X_normalized[train_indices, i, :] = Xi_train_normalized
            X_normalized[test_indices, i, :] = Xi_test_normalized

    # Save the normalized data
    normalized_data = {'X': X_normalized, 'y': y, 'sid': sid}
    pd.to_pickle(normalized_data, output_path)

    print(f"Normalized data saved to '{output_path}'")
