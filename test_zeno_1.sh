#!/bin/bash

export MXNET_CUDNN_AUTOTUNE_DEFAULT=0

export CUDA_VISIBLE_DEVICES=1

python mxnet_cnn_cifar10_impl.py --nepochs 2 --lr 0.05 --batch_size 100 --nworkers 16 --nbyz 0 --byz_type bitflip --rho 200 --b 12 --zeno_size 4 --aggregation mean
