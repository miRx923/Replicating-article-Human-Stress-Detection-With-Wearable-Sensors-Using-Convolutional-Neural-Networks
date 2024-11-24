import pandas as pd
import numpy as np
from scipy import stats
import os



DATA_PATH = '../../../data/original/'
chest_columns = ['sid', 'acc1', 'acc2', 'acc3', 'ecg', 'emg', 'eda', 'temp', 'resp', 'label']
all_columns = ['sid', 'c_acc_x', 'c_acc_y', 'c_acc_z', 'ecg', 'emg', 'c_eda', 'c_temp', 'resp',
               'w_acc_x', 'w_acc_y', 'w_acc_z', 'bvp', 'w_eda', 'w_temp', 'label']
ids = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]

sf_BVP = 64
sf_EDA = 4
sf_TEMP = 4
sf_ACC = 32
sf_chest = 700


# Convert data from pickle dictionary format to numpy arrays for wrist
def pkl_to_np_wrist(filename, subject_id):
    unpickled_df = pd.read_pickle(filename)
    
    # Conversion to float16 and numpy arrays
    wrist_acc = np.asarray(unpickled_df["signal"]["wrist"]["ACC"], dtype=np.float16)
    wrist_bvp = np.asarray(unpickled_df["signal"]["wrist"]["BVP"], dtype=np.float16)
    wrist_eda = np.asarray(unpickled_df["signal"]["wrist"]["EDA"], dtype=np.float16)
    wrist_temp = np.asarray(unpickled_df["signal"]["wrist"]["TEMP"], dtype=np.float16)
    lbl = np.asarray(unpickled_df["label"].reshape(-1, 1), dtype=np.int32)  # Assume labels are integers
    
    n_wrist_acc = wrist_acc.shape[0]
    n_wrist_bvp = wrist_bvp.shape[0]
    n_wrist_eda = wrist_eda.shape[0]
    
    # Create sid for each sensor type
    sid_acc = np.full((n_wrist_acc, 1), subject_id, dtype=np.float16)
    sid_bvp = np.full((n_wrist_bvp, 1), subject_id, dtype=np.float16)
    sid_eda_temp = np.full((n_wrist_eda, 1), subject_id, dtype=np.float16)
    
    # Function to resample labels
    def resample_labels(lbl, batch_size, n_samples):
        lbl_m = np.empty((n_samples, 1), dtype=np.float16)
        for i in range(n_samples):
            start = int(round(i * batch_size))
            end = int(round((i + 1) * batch_size))
            if end > len(lbl):
                end = len(lbl)
            batch_labels = lbl[start:end]
            if len(batch_labels) == 0:
                mode_val = lbl[start][0]  # Handle empty batch
            else:
                mode_val = stats.mode(batch_labels)[0][0]
            lbl_m[i] = mode_val
        return lbl_m
    
    # Resample labels for accelerometer
    batch_size_acc = sf_chest / sf_ACC
    lbl_acc = resample_labels(lbl, batch_size_acc, n_wrist_acc)

    # Resample labels for BVP
    batch_size_bvp = sf_chest / sf_BVP
    lbl_bvp = resample_labels(lbl, batch_size_bvp, n_wrist_bvp)
    
    # Resample labels for EDA and TEMP
    batch_size_eda = sf_chest / sf_EDA
    lbl_eda_temp = resample_labels(lbl, batch_size_eda, n_wrist_eda)
    
    # Concatenate data with corresponding labels
    data1 = np.concatenate((sid_acc, wrist_acc, lbl_acc), axis=1)
    data2 = np.concatenate((sid_bvp, wrist_bvp, lbl_bvp), axis=1)
    data3 = np.concatenate((sid_eda_temp, wrist_eda, wrist_temp, lbl_eda_temp), axis=1)

    return data1, data2, data3


def merge_wrist_data():
    md1, md2, md3 = None, None, None
    for i, sid in enumerate(ids):
        file = DATA_PATH + 'S' + str(sid) + '.pkl'
        current_md1, current_md2, current_md3 = pkl_to_np_wrist(file, sid)
        
        if i == 0:
            md1, md2, md3 = current_md1, current_md2, current_md3
        else:
            md1 = np.concatenate((md1, current_md1), axis=0)
            md2 = np.concatenate((md2, current_md2), axis=0)
            md3 = np.concatenate((md3, current_md3), axis=0)
    
    fn_merged1 = '../../../data/processed/subj_merged_acc_w.pkl'
    fn_merged2 = '../../../data/processed/subj_merged_bvp_w.pkl'
    fn_merged3 = '../../../data/processed/subj_merged_eda_temp_w.pkl'
    all_columns1 = ['sid', 'w_acc_x', 'w_acc_y', 'w_acc_z', 'label']
    all_columns2 = ['sid', 'bvp', 'label']
    all_columns3 = ['sid', 'w_eda', 'w_temp', 'label']
    
    df1 = pd.DataFrame(md1, columns=all_columns1)
    df1[['w_acc_x', 'w_acc_y', 'w_acc_z']] = df1[['w_acc_x', 'w_acc_y', 'w_acc_z']].astype('float16')
    df1.to_pickle(fn_merged1)
    
    df2 = pd.DataFrame(md2, columns=all_columns2)
    df2[['bvp']] = df2[['bvp']].astype('float16')
    df2.to_pickle(fn_merged2)
    
    df3 = pd.DataFrame(md3, columns=all_columns3)
    df3[['w_eda', 'w_temp']] = df3[['w_eda', 'w_temp']].astype('float16')
    df3.to_pickle(fn_merged3)


# Convert data from pickle dictionary format to numpy arrays for chest
def pkl_to_np_chest(filename, subject_id):
    unpickled_df = pd.read_pickle(filename)
    
    # Conversion to float16 and numpy arrays
    chest_acc = np.asarray(unpickled_df["signal"]["chest"]["ACC"], dtype=np.float16)
    chest_ecg = np.asarray(unpickled_df["signal"]["chest"]["ECG"], dtype=np.float16)
    chest_emg = np.asarray(unpickled_df["signal"]["chest"]["EMG"], dtype=np.float16)
    chest_eda = np.asarray(unpickled_df["signal"]["chest"]["EDA"], dtype=np.float16)
    chest_temp = np.asarray(unpickled_df["signal"]["chest"]["Temp"], dtype=np.float16)
    chest_resp = np.asarray(unpickled_df["signal"]["chest"]["Resp"], dtype=np.float16)
    lbl = np.asarray(unpickled_df["label"].reshape(-1, 1), dtype=np.int32)  # Assume labels are integers
    sid = np.full((lbl.shape[0], 1), subject_id, dtype=np.float16)
    
    chest_all = np.concatenate((sid, chest_acc, chest_ecg, chest_emg, chest_eda, chest_temp, chest_resp, lbl), axis=1)
    return chest_all


def merge_chest_data():
    merged_data = None
    for i, sid in enumerate(ids):
        file = DATA_PATH + 'S' + str(sid) + '.pkl'
        current_data = pkl_to_np_chest(file, sid)
        
        if i == 0:
            merged_data = current_data
        else:
            merged_data = np.concatenate((merged_data, current_data), axis=0)
    
    fn_merged = '../../../data/processed/merged_chest.pkl'
    df = pd.DataFrame(merged_data, columns=chest_columns)
    
    # Convert numerical columns to float16
    float_cols = ['acc1', 'acc2', 'acc3', 'ecg', 'emg', 'eda', 'temp', 'resp']
    df[float_cols] = df[float_cols].astype('float16')
    
    df.to_pickle(fn_merged)


def filter_chest_data():
    df = pd.read_pickle("../../../data/processed/merged_chest.pkl")
    
    # Filter labels and temperature
    df_fltr = df[df["label"].isin([1, 2, 3, 4])]
    df_fltr = df_fltr[df_fltr["temp"] > 0]
    
    # Convert numerical columns to float16
    float_cols = ['acc1', 'acc2', 'acc3', 'ecg', 'emg', 'eda', 'temp', 'resp']
    df_fltr[float_cols] = df_fltr[float_cols].astype('float16')
    
    df_fltr.to_pickle("../../../data/processed/chest.pkl")
    os.remove("../../../data/processed/merged_chest.pkl")

    print("Saved chest.pkl")


if __name__ == "__main__":
    merge_wrist_data()
    merge_chest_data()
    filter_chest_data()
