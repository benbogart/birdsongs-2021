#!/bin/bash

# # create resources
# python azure_create_resources.py --create-workspace\
#  --subscription-id 6b443b21-01ef-4f24-91c6-70c888c1cb50 --ws-name birdsongs-2021

# python azure_create_resources.py --create-compute --gpus 1
# python azure_create_resources.py --create-compute --gpus 2
# python azure_create_resources.py --create-compute --gpus 4

# python azure_create_resources.py --create-env
#
# python azure_create_resources.py --upload-data
#
# python azure_create_resources.py --create-dataset --dataset-name train_short_audio\
#    --data-path /input/birdclef-2021
#
# python azure_create_resources.py --create-dataset --dataset-name birdsongs_npy\
#   --data-path /data/npy/


# Tests
# python azure_train.py --model-name cnn1_audin_nmel_1 --test

# python azure_train.py --model-name construct_transfer1 --augment-position
# python azure_train.py --model-name construct_transfer1 --augment-position --sr 22050


# python azure_train.py --model-name construct_transfer1 --augment-position\
#  --data-subset wav22k_short_audio --test

 # python azure_train.py --model-name construct_transfer1 \
 #  --data-subset wav22k_short_audio

  python azure_train.py --model-name construct_transfer2 \
   --data-subset sswav5sec22k --train-script train2.py --test
