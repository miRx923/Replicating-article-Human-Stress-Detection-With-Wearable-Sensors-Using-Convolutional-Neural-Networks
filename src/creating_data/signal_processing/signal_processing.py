import numpy as np
import pandas as pd
from numpy.fft import fft
from scipy.signal import resample
from joblib import Parallel, delayed
from tqdm import tqdm
import os

from utils import normalize_data


# Number of target points in the spectrum
target_num_points = 210
# Shift length in samples (0.25 seconds at 64 Hz)
shift_length = int(0.25 * 64)
# Window length in samples (60 seconds at 64 Hz)
window_length = int(60 * 64)
# Sampling rate
sample_rate = 64

# Subwindow lengths in seconds for each signal
subwindow_lengths_sec = {
    'acc1': 7, 'acc2': 7, 'acc3': 7, 'ecg': 30, 'emg': 0.84, 
    'eda': 30, 'temp': 35, 'resp': 35, 'w_acc_x': 7, 'w_acc_y': 7, 
    'w_acc_z': 7, 'bvp': 30, 'w_eda': 30, 'w_temp': 35
}

# Frequency ranges for each signal
freq_ranges = {
    'acc1': (0, 30), 'acc2': (0, 30), 'acc3': (0, 30), 'ecg': (0, 7),
    'emg': (0, 250), 'eda': (0, 7), 'temp': (0, 6), 'resp': (0, 6),
    'w_acc_x': (0, 30), 'w_acc_y': (0, 30), 'w_acc_z': (0, 30),
    'bvp': (0, 7), 'w_eda': (0, 7), 'w_temp': (0, 6)
}

# List of signals to process
signals_list = list(subwindow_lengths_sec.keys())

# Function for FFT processing
def process_signal(signal, subwindow_length_samples, shift_length_samples, freq_range, num_points=210, cube_root=False):
    num_samples = len(signal)
    # Number of subwindows
    num_subwindows = (num_samples - subwindow_length_samples) // shift_length_samples + 1
    if num_subwindows <= 0:
        return np.zeros(num_points)
    # Create subwindows
    starts = np.arange(0, num_samples - subwindow_length_samples + 1, shift_length_samples)
    subwindows = np.array([signal[start:start + subwindow_length_samples] for start in starts])
    # Compute FFT
    fft_values = fft(subwindows, axis=1)
    fft_magnitude = np.abs(fft_values)[:, :subwindow_length_samples // 2]
    freqs = np.fft.fftfreq(subwindow_length_samples, d=1 / sample_rate)[:subwindow_length_samples // 2]
    # Select frequency range
    mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    fft_magnitude = fft_magnitude[:, mask]
    # Apply cube root transformation if specified
    if cube_root:
        fft_magnitude = np.cbrt(fft_magnitude)
    # Resample to target number of points
    if fft_magnitude.shape[1] != num_points:
        fft_magnitude = resample(fft_magnitude, num_points, axis=1)
    # Compute average spectrum
    avg_spectrum = np.mean(fft_magnitude, axis=0)
    return avg_spectrum

# Function to process a single window
def process_window(start, all_data, cube_root=False):
    window_data = all_data.iloc[start:start + window_length]
    window_result = []
    # Process each signal
    for signal_name in signals_list:
        if signal_name in window_data.columns:
            subwindow_length_samples = int(subwindow_lengths_sec[signal_name] * sample_rate)
            shift_length_samples = int(0.25 * sample_rate)
            freq_range = freq_ranges[signal_name]
            signal_series = window_data[signal_name].dropna().to_numpy()
            if len(signal_series) >= subwindow_length_samples:
                avg_spectrum = process_signal(
                    signal_series, subwindow_length_samples, shift_length_samples,
                    freq_range, num_points=target_num_points, cube_root=cube_root
                )
            else:
                avg_spectrum = np.zeros(target_num_points)
            window_result.append(avg_spectrum)
        else:
            # If signal is missing, fill with zeros
            window_result.append(np.zeros(target_num_points))
    window_matrix = np.array(window_result)
    # Assign label
    labels_in_window = window_data['label'].unique()
    label = labels_in_window[0] if len(labels_in_window) == 1 else window_data['label'].mode()[0]
    # Assign subject ID
    sids_in_window = window_data['sid'].unique()
    sid = sids_in_window[0] if len(sids_in_window) == 1 else window_data['sid'].mode()[0]
    return window_matrix, label, sid

# Function for computing CQT filters
def compute_cqt_filters(num_bins, freq_range, num_points):
    fmin, fmax = freq_range
    frequencies = np.linspace(fmin, fmax, num_points)
    q = 1 / (2 ** (1 / num_bins) - 1)
    filters = []
    fmin = max(fmin, 1e-6)  # Ensure fmin is positive
    for k in range(num_bins):
        fk = fmin * (2 ** (k / num_bins))
        bandwidth = fk / q
        filter_response = np.exp(-0.5 * ((frequencies - fk) / bandwidth) ** 2)
        filters.append(filter_response)
    filters = np.array(filters)
    filters /= filters.sum(axis=1, keepdims=True)
    return filters

# Complete processing
def process_data(input_file, output_fft, output_fft_cr, output_fft_cr_cqt):
    all_data = pd.read_pickle(input_file)

    # FFT
    print("Processing FFT...")
    num_samples = len(all_data)
    window_starts = range(0, num_samples - window_length + 1, shift_length)
    results = Parallel(n_jobs=4)(delayed(process_window)(start, all_data) for start in tqdm(window_starts))
    X, y, sid = zip(*results)
    X, y, sid = np.array(X), np.array(y), np.array(sid)
    pd.to_pickle({'X': X, 'y': y, 'sid': sid}, output_fft)
    print(f"FFT saved to {output_fft}")

    # FFT + Cube Root
    print("Processing FFT + Cube Root...")
    results = Parallel(n_jobs=4)(delayed(process_window)(start, all_data, cube_root=True) for start in tqdm(window_starts))
    X, y, sid = zip(*results)
    X, y, sid = np.array(X), np.array(y), np.array(sid)
    pd.to_pickle({'X': X, 'y': y, 'sid': sid}, output_fft_cr)
    print(f"FFT + Cube Root saved to {output_fft_cr}")

    # FFT + Cube Root + CQT
    print("Processing FFT + Cube Root + CQT...")
    cqt_num_bins = 21
    signals_list = list(freq_ranges.keys())
    cqt_filters = {signal: compute_cqt_filters(cqt_num_bins, freq_ranges[signal], X.shape[2]) for signal in signals_list}
    X_cqt = []
    for i in range(X.shape[0]):
        window_matrix = X[i]
        window_cqt = []
        for j, signal_name in enumerate(signals_list):
            spectrum = window_matrix[j]
            filters = cqt_filters[signal_name]
            cqt_result = np.dot(spectrum, filters.T)
            window_cqt.append(cqt_result)
        X_cqt.append(np.array(window_cqt))
    X_cqt = np.array(X_cqt)
    pd.to_pickle({'X': X_cqt, 'y': y, 'sid': sid}, output_fft_cr_cqt)
    print(f"FFT + Cube Root + CQT saved to {output_fft_cr_cqt}")

    os.remove(input_file)

    # Normalize all three files
    normalize_data(output_fft, '../../../data/processed/fft_normalized.pkl')
    normalize_data(output_fft_cr, '../../../data/processed/fft_cr_normalized.pkl')
    normalize_data(output_fft_cr_cqt, '../../../data/processed/fft_cr_cqt_normalized.pkl')

    os.remove(output_fft)
    os.remove(output_fft_cr)
    os.remove(output_fft_cr_cqt)


# Run signal processing
process_data(
    input_file='../../../data/processed/all_data_final.pkl',
    output_fft='../../../data/processed/fft.pkl',
    output_fft_cr='../../../data/processed/fft_cr.pkl',
    output_fft_cr_cqt='../../../data/processed/fft_cr_cqt.pkl'
)
