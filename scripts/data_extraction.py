import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset

# Data Extraction and Cleaning Pipeline
# 1. Get all relevant files
# 2. Split into train and test files
# 3. Read files (calculating SOC Percentage and SOC capacity, dropping null values, selecting the features and labels)
#       note: returns two lists of numpy arrays (features, labels)
# 4. Break each sequence into steps so that it can be fed into the LSTM
#       note: step_size=None will use the largest valid dataframe size and each
#             dataframe with 0s to fit max_length
# 5. Normalize the predictors (in train_x and test_x)
#        note: min_max_features is obtained from train data and is stored as a npy file based on the given min_max_path
#        this file will then be used to normalize any test data (for example if function called with test_size=1)
#        everytime training data is obtained the file updates new f_min, f_max
# 6. Smooth out noise in data by taking average from a rolling window (in train_x only)
# 7. Print out cleaned data, check for null/inf values, return a TensorDataset

FEATURES = ['Voltage', 'Current', 'Temperature']

LABEL = ['SOC']

FEATURES_LENGTH = 3

LABEL_LENGTH = 1

# class used for storing selected files and function to read them
class ds:
    def __init__(self, files, read_file):
        self.files = files
        self.read_file = read_file

# select all files in provided directory that contain at least one of file_keywords in name
# a file will be selected only if its folder matches one of folder_keywords
# note: if keywords are None, then all relevant files/folders are selected
# note: only csv files are selected
def get_file_paths(directory, suffix, file_keywords=None, folder_keywords=None):
    file_paths = []

    for root, dirs, files in os.walk(directory):
        if folder_keywords is None or any(keyword in root for keyword in folder_keywords):
            for file in files:
                if file_keywords is None or any(keyword in file for keyword in file_keywords):
                    if suffix is None or file.endswith(suffix):
                        file_paths.append(os.path.join(root, file))

    print(f'Getting files : {file_paths}')

    return file_paths

# read all files
def read_files(file_paths, read_file):
    features = []
    labels = []

    for file_path in file_paths:
        feature, label = read_file(file_path)

        if feature is not None and label is not None:
            features.append(feature)
            labels.append(label)

    return features, labels


# read INR file (calculating SOC Percentage and SOC capacity, dropping null values, selecting the features and labels)
def read_inr(file_path):
    # we skip first 30 rows since they include metadata in invalid csv format
    df = pd.read_csv(file_path, skiprows=30)

    # name columns
    df.columns = ['Time Stamp', 'Step', 'Status', 'Prog Time', 'Step Time', 'Cycle',
                    'Cycle Level', 'Procedure', 'Voltage', 'Current', 'Temperature', 'Capacity', 'WhAccu', 'Cnt',
                    'Empty']

    # drop empty column (created because of extra trailing commas in csv files
    df.drop('Empty', axis=1, inplace=True)

    # if there are null values we can drop them/ interpolate values / take average
    # we simply choose to drop them (there are no null values in this dataset anyway)
    df = df.dropna()

    # we calculate SOC capacity using given battery capacity, like before
    df['SOC Capacity'] = (df['Capacity'] - df['Capacity'].min())
    max_capacity = df['SOC Capacity'].max()

    # note: for some reason test 25DegC/551_HWFET contains all 0 capacity values
    # we filter this file (and other possible files with 0 capacity) out to avoid getting null values during division
    if max_capacity == 0:
        return None, None

    # we calculate SOC Percentage from capacity
    df['SOC'] = df['SOC Capacity'] / max_capacity

    return df[FEATURES], df[LABEL]

# read CALB file
def read_calb(file_path):

    df = pd.read_excel(file_path, sheet_name=1)

    # change display settings so we can see all columns
    pd.set_option('display.max_columns', None)

    # check data
    print(df.head(5))

    # select columns of interest
    df = df[['Test Time (s)', 'Test Time (s)', 'Current (A)', 'Voltage (V)', 'Power (W)', 'Aux_Temperature_1 (C)',
             'Aux_Temperature_1 (C)', 'Charge Capacity (Ah)', 'Discharge Capacity (Ah)']]

    # change column names to match current naming
    df.columns = ['Time Stamp', 'Elapsed Time', 'Current', 'Voltage', 'Power', 'Temperature1', 'Temperature2',
                  'Charge Capacity', 'Discharge Capacity']

    # get temperature average
    df['Temperature'] = (df['Temperature1'] + df['Temperature2']) / 2

    df = df.iloc[df['Voltage'].idxmax(): df['Voltage'].idxmin() + 1]
    df['SOC'] = (df['Discharge Capacity'].max() - df['Discharge Capacity']) / df['Discharge Capacity'].max()

    return df[FEATURES], df[LABEL]

# breaks each time-series sequence into steps and format to feed into LSTM
# breaking sequences greatly reduced memory burden on LSTM (weights update more frequently)
# if step_size=None break_sequences just pads values based on max length
def break_sequences(data, step_size=None):
    if step_size is None:
        # get max
        max_length = max(arr.shape[0] for arr in data)

    # note: np arrays have fixed length to avoid constantly creating new arrays we use list
    broken = []

    for arr in data:
        if step_size is None:
            # pad with 0s until max
            pad_length = max_length - len(arr)
            pad_width = ((0, pad_length), (0, 0))
            broken.append(np.pad(arr, pad_width))
        else:
            # break array into length of step_size and append
            for i in range(0, len(arr) - step_size + 1, step_size):
                broken.append(arr[i:i + step_size])

    # return as np array
    return np.array(broken)

# normalizes using feature_minimums (f_min) and feature_maximums (f_max)
def get_normalized(features, min_max_path):
    if min_max_path is None:
        return features

    f_min, f_max = np.load(min_max_path)

    # epsilon use to avoid null values when dividing
    epsilon = 1e-12
    new_features = (features - f_min) / (f_max - f_min + epsilon)

    return new_features

# smoothen training data with rolling window to reduce noise
# only training data is included
# test data should not be included since we want to evaluate model on real-world values
def get_smoothened(features, window_size):
    new_features = features.rolling(window=window_size, min_periods=1).mean()
    return new_features

# get and process data
# function will return Tensor Datasets that can then be used for training/testing/evaluation
def get_data(file_paths, read_file, min_max_path=None, step_size=None, normalize=False, smoothen=False,
             window_size=1, update_min_max=False):
    # return if no files
    if not file_paths:
        return None

    # read files
    data_x, data_y = read_files(file_paths, read_file)

    # break into sequences
    data_x = break_sequences(data_x, step_size=step_size)
    data_y = break_sequences(data_y, step_size=step_size)

    # update f_min and f_max
    if update_min_max and min_max_path:
        # find min and max of features and save to npy file
        # operate along axis=0 (arrays) and axis=1 (rows) while keeping columns unchanged
        f_min = np.min(data_x, axis=(0,1))
        f_max = np.max(data_x, axis=(0,1))
        np.save(min_max_path, np.array([f_min, f_max]))

    # normalize
    if normalize:
        data_x = get_normalized(data_x, min_max_path)

    # smoothen
    if smoothen:
        data_x = get_smoothened(data_x, window_size)

    # print cleaned data shape, null/inf values
    print(f'Features (shape: {data_x.shape}): \n{data_x[0][:20]}')
    print(f'Labels (shape: {data_y.shape}): \n{data_y[0][:20]}')
    print(f'Features: null = {np.isnan(data_x).sum()}, inf = {np.isinf(data_x).sum()}')
    print(f'Labels: null = {np.isnan(data_y).sum()}, inf = {np.isinf(data_y).sum()}')

    data_x = torch.FloatTensor(data_x)
    data_y = torch.FloatTensor(data_y)
    dataset = TensorDataset(data_x, data_y)

    return dataset

'''
from sklearn.model_selection import train_test_split

INR_DATASET_PATH = 'INR_ds'
INR_FOLDER_KEYWORDS = None
INR_FILE_KEYWORDS = ['Mixed', 'HWFET', 'UDDS', 'US06', 'LA92']

inr_ds = ds(read_file=read_inr, dataset_path='INR_ds', folder_keywords=None,
            file_keywords=['Mixed', 'HWFET', 'UDDS', 'US06', 'LA92'], suffix='.csv')

train_files, test_files = train_test_split(inr_ds.files, test_size=0.2)

print(f'train files : {train_files}')
print(f'test files : {test_files}')

MIN_MAX_PATH = 'experiments/copied_hyperparams_epochs100/min_max_features.npy'

# get tensor datasets
inr_train = get_data(file_paths=train_files, read_file=inr_ds.read_file, step_size=300, smoothen=False,
                     normalize=True, min_max_path=MIN_MAX_PATH, update_min_max=True)

inr_test = get_data(test_files, read_file=inr_ds.read_file, step_size=300, smoothen=False,
                    normalize=True, min_max_path=MIN_MAX_PATH, update_min_max=False)
                    
check min and max:
import numpy as np
MIN_MAX_PATH = 'experiments/copied_hyperparams_epochs100/min_max_features.npy'
data = np.load(MIN_MAX_PATH)
print(data)
'''