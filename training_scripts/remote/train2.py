
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
from DataGenerator import DataGenerator, AugDataGenerator
from LogToAzure import LogToAzure
from models import MODELS


def process_arguments():
    '''parse the parameters passed to the this script'''
    parser = argparse.ArgumentParser()

    parser.add_argument('--data-path', type=str,
                        dest='data_path',
                        default='../../sswav5sec22k', #local path for testing
                        help='data folder mounting point')

    parser.add_argument('--test-data-path', type=str,
                        dest='test_data_path',
                        default='../../sswav5sec22k', #local path for testing
                        help='data folder mounting point')

    parser.add_argument('--sr', type=int,
                        dest='sr',
                        default=22050,
help='sample rate of audio files')

    parser.add_argument('--offline', dest='online', action='store_const',
                        const=False, default=True,
                        help='Do not perform online (Azure specific) tasks')

    parser.add_argument('--model-name', type=str,
                         dest='model_name',
                         default='construct_transfer2',
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
dt = 10 # to second

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
df = pd.read_pickle(os.path.join(args.data_path, 'soundscapes.pkl'))
files = df['row_id'].map(lambda x: os.path.join(args.data_path, x + '.wav')).tolist()
labels = df['label'].tolist()

print('len(files)',len(files))

# print(files)
# Batch size for generators
BATCH_SIZE = 16


#####################
## Create Datasets ##
#####################


from dataset import SoundScapeDataset

sd = SoundScapeDataset(os.path.join(args.data_path),
                       name='train',
                       batch_size=BATCH_SIZE,
                       files_list=files,
                       labels_list=labels)
ds = sd.get_dataset()
if args.online:
    run.tag('Training', 'Dataset')

DATASET_SIZE = len(files)
print('DATASET_SIZE', DATASET_SIZE)

train_size = int(0.9 * DATASET_SIZE / BATCH_SIZE)
# val_size = int(0.1 * DATASET_SIZE)

print('train_size', train_size)

train_ds = ds.take(train_size)
val_ds = ds.skip(train_size)

# print(next(val_ds.as_numpy_iterator()))

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


# calculate steps per epoch



# fit model
# train_ds = train_ds.batch(2)

# audio, label = next(train_ds.as_numpy_iterator())
# print('audio shape:', audio.shape)
# print('label shape:', label.shape)
#
# history = model.fit(train_ds,
#                     steps_per_epoch=steps_per_epoch,
#                     epochs=2)





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


# print('Loading best model for testing...')
# model.load_weights(f'outputs/{args.model_name}-{runid}.h5')
#
# # save predictions
# # do this first so the generator starts from the beginning
# print('generating predictions on test set...')
# test_pred = model.predict(test_ds)
#
# print('saving test predictions...')
# with open(f'outputs/{args.model_name}-{runid}-test_predictions.pkl', 'wb') as f:
#     pickle.dump(test_pred, f)
#
#
# # evaluate test set
# print('evaluating model on test set...')
# model_val = model.evaluate(test_ds)
#
# print('model_val len',len(model_val))
# print('metrics len',len(metrics))

# build test metric dict
# test_metrics = {}
# for i, m in enumerate(metrics):
#     print(f'test_{m.name}: {model_val[i+1]}')
#     test_metrics['test_'+m.name] = model_val[i+1]
#     if args.online:
#         print('logging metrics...')
#         run.log('test_'+m.name, np.float(model_val[i+1]))

# # save test metrics
# print('Saving test metrics...')
# os.makedirs('outputs', exist_ok=True)
# with open(f'outputs/{args.model_name}-{runid}-test_metrics.pkl', 'wb') as f:
#     pickle.dump(test_metrics, f)


print('Done!')
print('-'*100)
