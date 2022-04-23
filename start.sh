#!/bin/bash

nohup python -u train.py \
  --gpu cuda:0 \
  --epochs 2 \
  --batch 64 \
  --lr 0.0001 \
  > ./train_LawyerPAN.log 2>&1 &


