#!/bin/bash

export NET="hashed"
export GPUID=0
export TRAIN_DIR="./MNIST"

if [ $# -eq 0 ]
then
  echo "Usage: ./train.sh [options]"
  echo " "
  echo "options:"
  echo "-h, --help                show brief help"
  echo "-net                      (hashed|???)"
  echo "-gpu                      gpu id"
  echo "-train_dir                directory for training logs"
  exit 0
fi

while test $# -gt 0; do
  case "$1" in
    -h|--help)
      echo "Usage: ./train.sh [options]"
      echo " "
      echo "options:"
      echo "-h, --help                show brief help"
      echo "-net                      (hashed|???)"
      echo "-gpu                      gpu id"
      echo "-train_dir                directory for training logs"
      exit 0
      ;;
    -net)
      export NET="$2"
      shift
      shift
      ;;
    -gpu)
      export GPUID="$2"
      shift
      shift
      ;;
    -train_dir)
      export TRAIN_DIR="$2"
      shift
      shift
      ;;
    *)
      break
      ;;
  esac
done

case "$NET" in 
  "hashed")
    export PRETRAINED_MODEL_PATH="../pretrain/MNIST/MNIST.pkl" #data : variable, 
    ;;
  "hashed-LSM")
    export PRETRAINED_MODEL_PATH="./data/ResNet/ResNet-50-weights.pkl"
    ;;
  *)
    echo "net architecture not supported."
    exit 0
    ;;
esac


python3 train.py \
  --dataset=MNIST \
  --image_set=train \
  --max_steps=10000 \
  --summary_step=10 \
  --checkpoint_step=50 \
  --train_dir="$TRAIN_DIR/train" \
  --net=$NET \
  --pretrained_model_path=$PRETRAINED_MODEL_PATH \
  --gpu=$GPUID
