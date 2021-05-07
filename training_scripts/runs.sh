#!/bin/bash

# # create resources
# python azure_create_resources.py --create-workspace\
#  --subscription-id 6b443b21-01ef-4f24-91c6-70c888c1cb50 --ws-name birdsongs-2021
#
python azure_create_resources.py --create-compute --gpus 1
# python azure_create_resources.py --create-compute --gpus 2
# python azure_create_resources.py --create-compute --gpus 4
#
# python azure_create_resources.py --create-env
#
# python azure_create_resources.py --upload-data
#
# python azure_create_resources.py --create-dataset --dataset-name birdsongs_10sec\
#   --data-path /data/audio_10sec/
#
# python azure_create_resources.py --create-dataset --dataset-name birdsongs_npy\
#   --data-path /data/npy/
