# FT-DropBlock regularization
# Attribution to: Pierre Sedi N.
# https://github.com/SedCore/FTDropBlock
# Original paper: P. Sedi Nzakuna, V. Paciello, V. Gallo, A. Lay-Ekuakille, A. Kuti Lusala,
#   “FT-DropBlock: A Novel Approach for SPatiotemporal Regularization in EEG-based Convolutional Neural Networks,”
#   in 2025 IEEE International Instrumentation and Measurement Technology Conference (I2MTC), 2025
#
# Inspired from: https://github.com/miguelvr/dropblock

import time
import torch
import argparse
from braindecode.preprocessing import create_windows_from_events
from braindecode.preprocessing import exponential_moving_standardize, preprocess, Preprocessor
from skorch.helper import SliceDataset
from braindecode.util import set_random_seeds
from sklearn.metrics import cohen_kappa_score
from utils import SuppressPrint, load_dataset

import numpy as np
import random
import tensorflow as tf

import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "sans-serif"

from sklearn.metrics import accuracy_score
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.utils import to_categorical

cuda = torch.cuda.is_available() # Check if GPU is available. If not, use CPU.
device = 'cuda' if cuda else 'cpu'

# Random seed to make results reproducible
set_random_seeds(seed=42, cuda=cuda) # Set random seed to be able to reproduce results
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)
torch.manual_seed(42)

# Handle parameters from the terminal CLI
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='EEGNetv4', choices=['EEGNetv4', 'EEGTCNet', 'EEGITNet'],
                    help='Select the model: EEGNetv4, EEGTCNet, EEGITNet.')
parser.add_argument('--reg', type=int, default=1, choices=[0, 1],
                    help='Select the regularization method: 0 for Dropout or 1 for FT-DropBlock')
parser.add_argument('--prob', type=float, default=0.5, choices=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                    help='Define the overall drop probability: 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9')
parser.add_argument('--block', type=int, default=15, choices=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                    help='Define an integer block size value for FT-DropBlock.')
args = parser.parse_args()

reg = args.reg; prob = args.prob; block_size = args.block; mod = args.model
droptype = 'FTDropBlock2D' if reg == 1 else 'Dropout'

print('Selected model: ', mod)
print('Regularization method: ', droptype)
print('Drop probability: ', prob)
if reg==1: print('Block size: ', block_size)

# Load the dataset
print("Loading the dataset BCI Competition IV 2a...")
with SuppressPrint(): raw_dataset = load_dataset("BCICIV_2a")

# Check the sampling frequency consistency across all datasets included in the raw data
sfreq = raw_dataset.datasets[0].raw.info['sfreq'] 
assert all([ds.raw.info['sfreq'] == sfreq for ds in raw_dataset.datasets])

# Preprocess the dataset
print("Signal preprocessing...")
preprocessors = [Preprocessor('pick', picks=['eeg'])] # only use eeg(stim channels must be removed)
preprocessors.append(Preprocessor(exponential_moving_standardize, factor_new=1e-3))
with SuppressPrint(): preprocessed_dataset = preprocess(raw_dataset, preprocessors=preprocessors)

# Create windows in the dataset based on events
trial_start_offset_seconds = -0.5 # 0.5s before the beginning of the MI task.
trial_stop_offset_seconds = 0 # Right at the end of the MI task
trial_start_offset_samples = int(trial_start_offset_seconds * sfreq)
trial_stop_offset_samples = int(trial_stop_offset_seconds * sfreq)
print("Creating windows from events...")
with SuppressPrint(): windows_dataset = create_windows_from_events(
            preprocessed_dataset,
            trial_start_offset_samples=trial_start_offset_samples,
            trial_stop_offset_samples=trial_stop_offset_samples,
            window_size_samples=None,
            window_stride_samples=None,
            preload=True,
            mapping=None
        )

# Classification parameters
n_channels = windows_dataset[0][0].shape[0] # 22
n_times = windows_dataset[0][0].shape[1] # 1125
n_classes = 4

print("Number of EEG electrode channels = ",n_channels)
print("Number of time points per window (number of samples per second) = ",n_times)
print("Frequency = ", sfreq)

# Split the dataset per subject, to get Train & Test subdatasets for each subject
subjects_windows_dataset = windows_dataset.split('subject')
n_subjects = len(subjects_windows_dataset.items())

# Model evaluation
accs = []
ks = []
def evaluate_model(model, trainx, trainy, testx, testy, subject, epochs, batch_size, lr, dropType, callbacks=[]):  
    # Load a saved model
    # with SuppressPrint(): model = tf.keras.models.load_model(f'save/{model.name}/S{subject}_{dropType}.keras', safe_mode=False)

    trainy_oh = to_categorical(trainy, num_classes=4)
    testy_oh = to_categorical(testy, num_classes=4)
    opt = Adam(learning_rate=lr)
    model.compile(loss=categorical_crossentropy, optimizer=opt, metrics=['accuracy'])

    # Train a new model and collect the history
    print('Training the model...')
    start_time = time.time()
    history = model.fit(trainx, trainy_oh, validation_data=(testx,testy_oh), batch_size=batch_size, epochs=epochs, callbacks=callbacks, verbose=0)
    end_time = time.time()
    print(f"Training time: {(end_time - start_time):.2f} seconds")

    train_loss = history.history['loss']; val_loss = history.history['val_loss']
    train_accuracy = history.history['accuracy']; val_accuracy = history.history['val_accuracy']
    plt.figure(figsize=(20, 10))

    # Plot Training & Validation Loss
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, color='red', label='Training Loss'); plt.plot(val_loss, color='blue', label='Validation Loss')
    plt.xlabel('Epochs', fontsize=22); plt.ylabel('Loss', fontsize=22)
    plt.title(f'Train & Val Loss: Subject {subject} - {'FT-DropBlock' if dropType=='FTDropBlock2D' else dropType}', fontsize=22)
    plt.xticks(fontsize=22); plt.yticks(fontsize=22); plt.legend(prop={"size":22})

    # Plot Training & Validation Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracy, color='red', label='Training Accuracy'); plt.plot(val_accuracy, color='blue', label='Validation Accuracy')
    plt.xlabel('Epochs', fontsize=22); plt.ylabel('Accuracy', fontsize=22)
    plt.title(f'Train & Val Accuracy: Subject {subject} - {'FT-DropBlock' if dropType=='FTDropBlock2D' else dropType}', fontsize=22)
    plt.xticks(fontsize=22); plt.yticks(fontsize=22); plt.legend(prop={"size":22})

    # Save the plots to an image file
    plt.savefig(f'figures/{model.name}/train_val_S{subject}_{dropType}.png')
    plt.close()

    # Save the model   
    # model.save(f'save/{model.name}/S{subject}_{dropType}.keras')

    print('Testing...')
    y_pred = model.predict(testx, verbose=0).argmax(axis=-1)
    labels = testy_oh.argmax(axis=-1)
    accuracy_of_test = round(accuracy_score(labels, y_pred)* 100, 4)
    kappa = round(cohen_kappa_score(labels, y_pred), 4)
        
    accs.append(accuracy_of_test)
    ks.append(kappa)
    print(f"Acc: {accuracy_of_test} Kappa: {kappa}")

# Evaluate the model for each subject in the dataset
for subject, windows_dataset in subjects_windows_dataset.items():
    print("---------------------------------------------------")
    print(f"Subject {subject} \n")
    train_dataset = windows_dataset.split('session')['0train']
    test_dataset = windows_dataset.split('session')['1test'] 
    train_X = np.array([x for x in SliceDataset(train_dataset, idx=0)]) #print(train_X.shape) = (288,22,1125) batch,eeg_ch,times
    train_y = np.array([y for y in SliceDataset(train_dataset, idx=1)]) #print(train_y.shape) = (288,)
    
    test_X = np.array([x for x in SliceDataset(test_dataset, idx=0)])#
    test_y = np.array([y for y in SliceDataset(test_dataset, idx=1)])

    #### FOR TENSORFLOW KERAS
    train_X_keras = np.expand_dims(train_X, axis=1) # batch,1,eeg_channels,times (N,C,H,W) Channels first
    test_X_keras = np.expand_dims(test_X, axis=1) # batch,1,eeg_channels,times (N,C,H,W) Channels first
    keras_model = None
    batchsize = 64
    callbacks = []
    lr = 0.001
    epochs = 500

    if mod=='EEGNetv4':
        from models.eegnet import EEGNetv4
        keras_model = EEGNetv4(n_classes, Chans=n_channels, Samples=n_times, dropoutRate=prob, F1=8, kernLength=32, D=2, F2=2*8, dropType=droptype, block=block_size)
    elif mod=='EEGTCNet':
        from models.eegtcnet import EEGTCNet
        keras_model = EEGTCNet(n_classes, Chans=n_channels, Samples=n_times, layers=2,
                               kernel_s=4, filt=12, dropout=0.3, activation='elu', F1=8, D=2, kernLength=32, dropout_eeg=prob, dropType=droptype, block=block_size)
        epochs = 750
    elif mod=='EEGITNet':
        from models.eegitnet import EEGITNet
        keras_model = EEGITNet(out_class=n_classes, drop_rate=prob, dropType=droptype, blocksize=block_size)
        train_X_keras = train_X_keras.transpose(0,2,3,1) # to match the channels last format
        test_X_keras = test_X_keras.transpose(0,2,3,1) # to match the channels last format
    #keras_model.summary();exit()
    evaluate_model(keras_model,
                   trainx = train_X_keras,
                   trainy = train_y,
                   testx = test_X_keras,
                   testy = test_y,
                   subject = subject,
                   epochs = epochs,
                   batch_size = batchsize,
                   lr = lr,
                   dropType=droptype,
                   callbacks = callbacks)

    
print(f"Mean accuracy for all dataset: {np.mean(accs):.4f}% ± {np.std(accs):.4f}")
print(f"Mean Kappa for all dataset: {np.mean(ks):.4f} ± {np.std(ks):.4f}")