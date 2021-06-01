
import argparse
import os
import numpy as np
import pickle
import pandas as pd
import sys
from azureml.core import Run
from azureml.core import Workspace, Dataset
from tensorflow import keras as K
from tensorflow.distribute import MirroredStrategy
#from tensorflow.compat.v1 import RunOptions
import tensorflow_addons as tfa

from sklearn.preprocessing import LabelEncoder
import json
import subprocess

# custom
# from DataGenerator import DataGenerator, AugDataGenerator
from LogToAzure import LogToAzure
from models import MODELS
import tensorflow as tf


def process_arguments():
    '''parse the parameters passed to the this script'''
    parser = argparse.ArgumentParser()

    parser.add_argument('--data-path', type=str,
                        dest='data_path',
                        default='../../wav', #local path for testing
                        help='train data folder mounting point')

    parser.add_argument('--test-data-path', type=str,
                        dest='test_data_path',
                        default='../../sswav5sec22k', #local path for testing
                        help='data folder mounting point')

    parser.add_argument('--noise-data-path', type=str,
                        dest='noise_data_path',
                        default='../../sswav5sec', #local path for testing
                        help='noise data folder mounting point')

    parser.add_argument('--sr', type=int,
                        dest='sr',
                        default=16000,
                        help='sample rate of audio files')

    parser.add_argument('--offline', dest='online', action='store_const',
                        const=False, default=True,
                        help='Do not perform online (Azure specific) tasks')

    parser.add_argument('--model-name', type=str,
                         dest='model_name',
                         default='vggish_time_dist_1',
                         help='name of model to build')

    parser.add_argument('--data-subset',
                        type=str,
                        dest='data_subset',
                        default='kaggle_ogg',
                        help='the subset of the data to use [all, kaggle].')

    parser.add_argument('--augment-position',
                        action='store_const',
                        dest='augment_position',
                        const=True, default=False,
                        help='Whether to choose clip position randomly')

    parser.add_argument('--augment-pitch',
                        action='store_const',
                        dest='augment_pitch',
                        const=True, default=False,
                        help='Use pitch augmentation.')

    parser.add_argument('--augment-stretch',
                        action='store_const',
                        dest='augment_stretch',
                        const=True, default=False,
                        help='Use time stretch augmentation')

    parser.add_argument('--multithread',
                        action='store_const',
                        dest='multithread',
                        const=True, default=False,
                        help='Use multithread processing for data loading')

    parser.add_argument('--epochs',
                        type=int,
                        dest='epochs',
                        default=5,
                        help='number of epochs to try.')

    parser.add_argument('--learning-rate',
                        type=float,
                        dest='learning_rate',
                        default=0.0001,
                        help='learning rate.')

    print('Parsing Args...')
    args = parser.parse_args()
    return args

# set seed for reproducibility
np.random.seed(867)

# output is written to log file, separate output from previous log entries.
print('-'*100)

args = process_arguments()
print('ARGS\n',args)

# print(os.listdir(args.data_path))

# create outputs dir
os.makedirs('outputs', exist_ok = True)

# set variables
sr = args.sr # sample rate
dt = 5 # to second

# build data_subset_id with augmentation info
data_subset = args.data_subset
if args.augment_position:
    data_subset += '_aug'
if args.augment_pitch:
    data_subset += '_pitch'
if args.augment_stretch:
    data_subset += '_stretch'

# if the run is online start logging
if args.online:
    run = Run.get_context()
    run.tag('model_name', args.model_name)
    run.tag('learning_rate', args.learning_rate)
    run.tag('data_subset', data_subset)

    # set local runid
    runid = run.id

    # Print enviornment to stdout log
    print('Environment:',run.get_environment().name)

    # start cpu/gpu utilisation logging
    logger_fname = f'outputs/{runid}-log_compute.csv'
    logger_pid = subprocess.Popen(
        ['python', 'log_gpu_cpu_stats.py',
         logger_fname,
         '--loop',  '0.2',  # Interval between measurements, in seconds (optional, default=1)
        ])
    print('Started logging compute utilisation, pid', logger_pid)

else:
    # for an offline run just set the run id to offline
    runid = 'offline'

# get file lists for train, val, test

label_file = 'data_split_single_label_tv.json'

with open(os.path.join(args.data_path, 'input/resources', label_file), 'r') as f:
    data =  json.load(f)

#     # get file lists for train, val, test
train_files = [os.path.join(args.data_path, name)
           for name in data['train']['files']]
val_files = [os.path.join(args.data_path, name)
           for name in data['val']['files']]

print('Num Training Files:', len(train_files))
print('Num Validation Files:', len(val_files))

classes = data['mapping']
n_classes = len(classes)

# print('classes: ', classes)
le =LabelEncoder()
le.fit(classes)

# transform labels
train_labels = le.transform(data['train']['labels']).tolist()
val_labels = le.transform(data['val']['labels']).tolist()

# print(train_labels)
# select files from soundscapes without birds.

print(args.noise_data_path)
ssdf = pd.read_csv(os.path.join(args.noise_data_path, 'train_soundscape_labels.csv'))

ssdf_filtered = ssdf[ssdf['birds'] == 'nocall']
noise_files = [os.path.join(args.noise_data_path, fid + '.wav') for fid in ssdf_filtered['row_id']]


# Batch size for generators
BATCH_SIZE = 1

#####################
## Create Datasets ##
#####################

from dataset import CombinedDatasetFullAudio

cd = CombinedDatasetFullAudio(train_files, train_labels, noise_files,
                     batch_size=BATCH_SIZE, is_train=True)
train_ds = cd.get_dataset()

# is this the right test????  Should be static but should it have noise??
cd = CombinedDatasetFullAudio(val_files, val_labels,
                     batch_size=BATCH_SIZE, is_train=False)
val_ds = cd.get_dataset()

print('Train Dataset Cardinality:', tf.data.experimental.cardinality(train_ds).numpy())
print('Validation Dataset Cardinality:', tf.data.experimental.cardinality(val_ds).numpy())


# DATASET_SIZE = len(files)
# print('DATASET_SIZE', DATASET_SIZE)

# metrics
metrics = [
    tfa.metrics.F1Score(num_classes=397,
                        average='micro'),
    K.metrics.CategoricalAccuracy(),
#    K.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy'),
#    K.metrics.TopKCategoricalAccuracy(k=5, name='top_5_accuracy')
]

# Construct model
model = MODELS[args.model_name]
model = model()

# compile model

#run_opts = RunOptions(report_tensor_allocations_upon_oom = True)

model.compile(optimizer=K.optimizers.Adam(learning_rate=args.learning_rate),
              loss='categorical_crossentropy',
              metrics=metrics)
#              options=run_opts)

## END scope

# write model summary to log
model.summary()
#
# callbacks
r_lr = K.callbacks.ReduceLROnPlateau(patience=2, factor=0.1)
cb = K.callbacks.EarlyStopping(patience=4)
mc = K.callbacks.ModelCheckpoint(filepath=f'outputs/{args.model_name}-{runid}.h5',
                                 save_best_only=True,
                                 save_weights_only=True)
tb = K.callbacks.TensorBoard(log_dir=f'logs/{args.model_name}/',
                          histogram_freq=1,
                          profile_batch=0)

callbacks = [r_lr, cb, mc, tb]

if args.online:
    # add LogToAure custome Callback if online
    callbacks.append(LogToAzure(run))



# fit model
history = model.fit(train_ds,
                    # steps_per_epoch=steps_per_epoch, # only for quick testing
                    # validation_split=0.2,  # does not work with dataset
                    validation_data=val_ds,
                    # validation_steps=validation_steps,
                    epochs=args.epochs,
                    callbacks=callbacks)

# save history
print('Saving model history...')

with open(f'outputs/{args.model_name}-{runid}.history', 'wb') as f:
    pickle.dump(history.history, f)




print('Done!')
print('-'*100)
