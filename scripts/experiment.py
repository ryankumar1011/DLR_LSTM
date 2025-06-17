# import libraries/functions
import os
from pathlib import Path
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import data_extraction as de
import lstm
import evaluation as eval
# experiment file structure
# to train model on different hyperparameters simpy change EXPERIMENT_NAME and the desired hyperparameters
EXPERIMENTS_PATH = Path('experiments')
EXPERIMENT_NAME = 'stateless_epochs30'
EXPERIMENT_PATH = EXPERIMENTS_PATH / EXPERIMENT_NAME
MODEL_PATH = EXPERIMENT_PATH / 'trained_model.pth'
EXPERIMENT_PLOTS_PATH = EXPERIMENT_PATH / 'plots'
LOG_PATH = EXPERIMENT_PATH / 'experiment_log.txt'
MIN_MAX_PATH = EXPERIMENT_PATH / 'min_max_features.npy'

# create directories if they are missing
EXPERIMENT_PATH.mkdir(parents=True, exist_ok=True)
EXPERIMENT_PLOTS_PATH.mkdir(parents=True, exist_ok=True)

# data selection
# list files manually or select them with get_files()
inr_files = de.get_file_paths(directory='INR_ds', suffix='.csv', folder_keywords=None,
                              file_keywords=['Mixed', 'HWFET', 'UDDS', 'US06', 'LA92'])
calb_files = de.get_file_paths(directory='CALB_ds', suffix='.xlsx', file_keywords=['DrivingCycle'],
                              folder_keywords=['Driving_cycles'])
# create ds objects
inr_ds = de.ds(files=inr_files, read_file=de.read_inr)
calb_ds = de.ds(files=calb_files, read_file=de.read_calb)

# choose data to include in experiment
datasets = [inr_ds]

# data hyperparameters
TEST_SIZE = 0.2
RANDOM_STATE = 10
STEP_SIZE = 300
NORMALIZE = True
SMOOTHEN = False

# model hyperparameters
INPUT_SIZE = 3
OUTPUT_SIZE = 1
HIDDEN_SIZE = 256
NUM_LAYERS = 3
DROPOUT = 0.0
NUM_EPOCHS = 30
LEARNING_RATE = 0.00001
BATCH_SIZE = 32

# if model already trained, load model
if os.path.exists(MODEL_PATH):
    # first create model with provided architecture
    model = lstm.SocLSTM(input_size=INPUT_SIZE, output_size=OUTPUT_SIZE, num_layers=NUM_LAYERS,
                         hidden_size=HIDDEN_SIZE, dropout=DROPOUT)

    # then load weights, biases, optimizer, etc from saved state_dict
    model.load_state_dict(torch.load(MODEL_PATH))

# if no saved model
else:
    # create empty datasets and lists to hold accumulated data
    train_ds_all = TensorDataset(torch.empty(0, 100, 3))

    test_ds_all = TensorDataset(torch.empty(0, 100, 1))

    train_files_all = []

    test_files_all = []

    for ds in datasets:
        # split files into train and test
        train_files, test_files = train_test_split(ds.files, test_size=TEST_SIZE, random_state=RANDOM_STATE)

        # concatenate file lists
        train_files_all += train_files
        test_files_all += test_files

        # get tensor datasets
        train_ds = de.get_data(file_paths=train_files, read_file=ds.read_file, step_size=STEP_SIZE,
                                smoothen=SMOOTHEN,normalize=NORMALIZE, min_max_path=MIN_MAX_PATH,
                               update_min_max=True)

        # important note: update_min_max=False since data should only be normalized on training features min/max
        test_ds = de.get_data(test_files, read_file=ds.read_file, step_size=STEP_SIZE,
                              smoothen=SMOOTHEN, normalize=NORMALIZE, min_max_path=MIN_MAX_PATH,
                              update_min_max=False)

        # merge tensor datasets
        train_ds_all += train_ds

        test_ds_all += test_ds

    # save data and hyperparameters used for training to txt log file
    with open(LOG_PATH, 'w') as log_file:
        log_file.write(f'Train files : {train_files_all}\n')
        log_file.write(f'Test files : {test_files_all}\n')
        log_file.write('\n')
        log_file.write('Data Hyperparameters:\n')
        log_file.write(f'Train-Test Split : {TEST_SIZE}\n')
        log_file.write(f'Random State : {RANDOM_STATE}\n')
        log_file.write(f'Step Size : {STEP_SIZE}\n')
        log_file.write(f'Normalize : {NORMALIZE}\n')
        log_file.write('\n')
        log_file.write('Model Hyperparameters:\n')
        log_file.write(f'Input Size: {INPUT_SIZE}\n')
        log_file.write(f'Output Size: {OUTPUT_SIZE}\n')
        log_file.write(f'Hidden Size: {HIDDEN_SIZE}\n')
        log_file.write(f'Number of Layers: {NUM_LAYERS}\n')
        log_file.write(f'Dropout: {DROPOUT}\n')
        log_file.write(f'Number of Epochs: {NUM_EPOCHS}\n')
        log_file.write(f'Learning Rate: {LEARNING_RATE}\n')
        log_file.write(f'Batch Size: {BATCH_SIZE}\n')

    # get train and test loaders
    train_loader = DataLoader(train_ds_all, batch_size=BATCH_SIZE, shuffle=True)

    test_loader = DataLoader(test_ds_all, batch_size=BATCH_SIZE, shuffle=False)

    # build model and save it to specified path (test_loader used to evaluate and print MSE test loss)
    model = lstm.build_model(train_loader=train_loader, test_loader=test_loader,
                             input_size=INPUT_SIZE, output_size=OUTPUT_SIZE, num_layers=NUM_LAYERS, hidden_size=HIDDEN_SIZE,
                             dropout = DROPOUT, num_epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE, path=MODEL_PATH)

    # plot all test graphs with test_data
    # pass in file names to save with correct names, saved in directory EXPERIMENT_PLOTS_PATH
    # eval.plot_tests(model, test_ds_all, test_files_all, EXPERIMENT_PLOTS_PATH)

# evaluate model on specific file/files individually
# train_files = ['INR_ds/0degC/589_US06.csv', 'INR_ds/0degC/590_Mixed6.csv', 'INR_ds/n10degC/602_Mixed5.csv', 'INR_ds/0degC/589_Mixed1.csv', 'INR_ds/n20degC/611_Mixed4.csv', 'INR_ds/40degC/562_Mixed6.csv', 'INR_ds/10degC/571_Mixed7.csv', 'INR_ds/n20degC/611_Mixed8.csv', 'INR_ds/n10degC/604_Mixed7.csv', 'INR_ds/n10degC/596_UDDS.csv', 'INR_ds/10degC/567_Mixed1.csv', 'INR_ds/40degC/556_HWFET.csv', 'INR_ds/10degC/571_Mixed4.csv', 'INR_ds/10degC/576_HWFET.csv', 'INR_ds/n20degC/610_UDDS.csv', 'INR_ds/n10degC/604_Mixed8.csv', 'INR_ds/n10degC/596_HWFET.csv', 'INR_ds/n10degC/604_Mixed6.csv', 'INR_ds/0degC/590_Mixed7.csv', 'INR_ds/n20degC/611_Mixed5.csv', 'INR_ds/n20degC/610_LA92.csv', 'INR_ds/40degC/556_UDDS.csv', 'INR_ds/0degC/589_LA92.csv', 'INR_ds/10degC/571_Mixed6.csv', 'INR_ds/25degC/551_US06.csv', 'INR_ds/n20degC/611_Mixed3.csv', 'INR_ds/0degC/589_UDDS.csv', 'INR_ds/n10degC/596_LA92.csv', 'INR_ds/25degC/552_Mixed4.csv', 'INR_ds/n20degC/610_HWFET.csv', 'INR_ds/25degC/552_Mixed7.csv', 'INR_ds/40degC/556_LA92.csv', 'INR_ds/0degC/589_HWFET.csv', 'INR_ds/0degC/590_Mixed8.csv', 'INR_ds/10degC/571_Mixed5.csv', 'INR_ds/n10degC/601_Mixed2.csv', 'INR_ds/n20degC/610_Mixed2.csv', 'INR_ds/25degC/552_Mixed5.csv', 'INR_ds/n20degC/610_US06.csv', 'INR_ds/40degC/557_Mixed3.csv', 'INR_ds/10degC/571_Mixed8.csv', 'INR_ds/n10degC/602_Mixed4.csv', 'INR_ds/0degC/590_Mixed5.csv', 'INR_ds/10degC/576_UDDS.csv', 'INR_ds/25degC/551_Mixed2.csv', 'INR_ds/10degC/567_US06.csv', 'INR_ds/25degC/551_HWFET.csv', 'INR_ds/25degC/551_LA92.csv', 'INR_ds/40degC/556_US06.csv', 'INR_ds/n10degC/604_Mixed3.csv', 'INR_ds/40degC/562_Mixed4.csv', 'INR_ds/n20degC/611_Mixed6.csv', 'INR_ds/n20degC/611_Mixed7.csv', 'INR_ds/n10degC/601_Mixed1.csv', 'INR_ds/0degC/590_Mixed4.csv', 'INR_ds/40degC/562_Mixed5.csv']
# test_files  = ['INR_ds/25degC/551_UDDS.csv', 'INR_ds/40degC/556_Mixed2.csv', 'INR_ds/40degC/556_Mixed1.csv', 'INR_ds/40degC/562_Mixed7.csv', 'INR_ds/25degC/552_Mixed3.csv', 'INR_ds/n20degC/610_Mixed1.csv', 'INR_ds/25degC/552_Mixed6.csv', 'INR_ds/0degC/589_Mixed2.csv', 'INR_ds/10degC/567_Mixed2.csv', 'INR_ds/n10degC/601_US06.csv', 'INR_ds/25degC/551_Mixed1.csv', 'INR_ds/40degC/562_Mixed8.csv', 'INR_ds/25degC/552_Mixed8.csv', 'INR_ds/10degC/582_LA92.csv']

for file_path in calb_ds.files:
    test_ds = de.get_data(file_paths=[file_path], read_file=de.read_calb, step_size=STEP_SIZE,
                           smoothen=SMOOTHEN,
                           normalize=NORMALIZE, min_max_path=MIN_MAX_PATH, update_min_max=False)

    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    eval.plot_predictions(model, test_loader, file_path, EXPERIMENT_PLOTS_PATH)

'''
# get all relevant files
    # can choose to manually list files instead of selecting by keywords
    inr_files = de.get_file_paths(directory=INR_DATASET_PATH, folder_keywords=INR_FOLDER_KEYWORDS,
                file_keywords = INR_FILE_KEYWORDS, suffix='.csv')

    calb_files = de.get_file_paths(directory=CALB_DATASET_PATH, folder_keywords=CALB_FOLDER_KEYWORDS,
                file_keywords=CALB_FILE_KEYWORDS, suffix='.xlsx')

    # split files into train and test
    inr_train_files, inr_test_files = train_test_split(inr_files, test_size=TEST_SIZE)

    calb_train_files, calb_test_files = train_test_split(calb_files, test_size=TEST_SIZE)

    # print selected files
    print(f'INR dataset train files : {inr_train_files}')
    print(f'INR dataset test files : {inr_test_files}')
    print(f'CALB dataset train files : {calb_train_files}')
    print(f'CALB dataset test files : {calb_test_files}')

    # get tensor datasets
    inr_train = de.get_data(file_paths=inr_train_files, read_file=de.read_inr, step_size=STEP_SIZE, smoothen=SMOOTHEN,
                        normalize=NORMALIZE, min_max_path=MIN_MAX_PATH, update_min_max=True)

    calb_train = de.get_data(file_paths=calb_train_files, read_file=de.read_calb, step_size=STEP_SIZE,smoothen=SMOOTHEN,
                          normalize=NORMALIZE, min_max_path=MIN_MAX_PATH, update_min_max=True)

    inr_test = de.get_data(inr_test_files, read_file=de.read_inr, step_size=STEP_SIZE, smoothen=SMOOTHEN,
                        normalize=NORMALIZE, min_max_path=MIN_MAX_PATH, update_min_max=False)

    calb_test = de.get_data(calb_test_files, read_file=de.read_calb, step_size=STEP_SIZE, smoothen=SMOOTHEN,
                        normalize=NORMALIZE, min_max_path=MIN_MAX_PATH, update_min_max=False)

    # merge tensor datasets (if needed)
    # note: can also use ConcatDataset([inr_test, calb_test]) instead
    train_data = inr_train + calb_train

    test_data = inr_test + calb_test


'''

'''
 # get train and test files
    train_files, test_files = train_test_split(files, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    train_data = get_data(read_file=read)

    # load train and test data
    train_loader = get_data(train_files, step_size=STEP_SIZE, normalize=True,
        min_max_path=MIN_MAX_PATH, batch_size=BATCH_SIZE, update_min_max=True, shuffle=True)

    test_loader = get_data(test_files, normalize=True,
        min_max_path=MIN_MAX_PATH, batch_size=BATCH_SIZE)
        
        
PLOT EDA DATA ON CLEANED FEATURES

    # get test data without step_size (here test_loader used to get complete data for
    # each experiment without steps)
    test_loader = get_data(test_files, normalize=True,
    min_max_path=MIN_MAX_PATH, batch_size=BATCH_SIZE, shuffle=False)


# model evaluation and graphing

# we can get files by listing them
# file_paths = ['dataset/n10degC/604_Mixed3.csv']

# or we can get them by their keywords
file_paths = get_file_paths(directory=DATASET_PATH, folder_keywords=['n10degC'], file_keywords = ['Mixed3'])

# we can load files into test_loader made specifically for testing
# (test_size=1, step_size=None (not broken into sequences), batch_size=1 (default))
# batch_size=1 would result in one plot for each file, batch_size=2
# would result in every 2 files being grouped, etc
_, test_loader = get_data(file_paths=file_paths, test_size=1, normalize=NORMALIZE, min_max_path=MIN_MAX_PATH)

# we can plot and save graph
eval_plot(model, test_loader,'25degC for Mixed1', EXPERIMENT_PLOTS_PATH)

# plot loss over epochs


# plot all graphs

for folder_name in os.listdir(DATASET_PATH):
    folder_path = os.path.join(DATASET_PATH, folder_name)
    if (os.path.isdir(folder_path) and (FOLDER_KEYWORDS is None or
        any(name in folder_name for name in FOLDER_KEYWORDS))):

        folder_plot_path = EXPERIMENT_PLOTS_PATH / folder_name
        folder_plot_path.mkdir(parents=True, exist_ok=True)

        for file_name in os.listdir(folder_path):
            if file_name.endswith('.csv') and (FILE_KEYWORDS is None or
                any(name in file_name for name in FILE_KEYWORDS)):
                file_path = os.path.join(folder_path, file_name)
                _, test_loader = get_data(file_paths=[file_path], test_size=1, normalize=NORMALIZE, min_max_path=MIN_MAX_PATH)
                plot_name = file_name[:-4]
                eval_plot(model, test_loader, plot_name, folder_plot_path)
'''

# To Do:
# Comments: explain hidden layers, dropout, model, etc
# Fix data noise
# Remove abnormal values
# Model is not learning long term enough (maybe because of dropout or maybe irregular intervals?)
# Check intervals
# Visualize more Results (RMSE, MAPE)
# Implement random forest

# Next goal:
# Model features (Current, Voltage, Temperature) given power profile

# Research
# How would I tune hyperparameters effectively
# How do I fix interval length (interpolation)
# How do I decide the sequence length (how many samples used to predict)
# How would SOC modelling relate to modelling SOH?
# How would I model estimated SOC given a future power profile
# CC/CV protocol and reason for process
# Look into HPPC tests
# Savitzky golay filter (SGF)