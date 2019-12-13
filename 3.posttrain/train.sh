#!/bin/bash

export NET="hashed"
export GPUID=0
export TRAIN_DIR="./MNIST"
export DATA_SET="MNIST"
export HASHED=True

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
    -data_set)
      export DATA_SET="$2"
      shift
      shift
      ;;
    -train_dir)
      export TRAIN_DIR="$2"
      shift
      shift
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
    -hashed)
      export HASHED="$2"
      shift
      shift
      ;;
    *)
      break
      ;;
  esac
done

case "$NET" in 
  "hashed_fc_MNIST")
    export PRETRAINED_MODEL_PATH="../2.train/MNIST/train/MNIST_FC.pkl" #data : variable, 
    export PRETRAINED_INFO_PATH="../2.train/MNIST/train/MNIST_FC_INFO.pkl" #data : variable, 
    ;;
  "hashed_conv_MNIST")
    export PRETRAINED_MODEL_PATH="../2.train/MNIST/train/MNIST_CONV.pkl" #data : variable, 
    export PRETRAINED_INFO_PATH="../2.train/MNIST/train/MNIST_CONV_INFO.pkl" #data : variable, 
    ;;
  "hashed_conv_IMGNET")
    export PRETRAINED_MODEL_PATH="../2.train/MNIST/train/IMGNET_CONV.pkl" #data : variable, 
    export PRETRAINED_INFO_PATH="../2.train/MNIST/train/IMGNET_CONV_INFO.pkl" #data : variable, 
    ;;
  *)
    echo "net architecture not supported."
    exit 0
    ;;
esac


python3 train.py \
  --dataset=MNIST \
  --train_dir="$TRAIN_DIR/train" \
  --net=$NET \
  --pretrained_model_path=$PRETRAINED_MODEL_PATH \
  --pretrained_info_path=$PRETRAINED_INFO_PATH \
  --gpu=$GPUID\
  --hashed=$HASHED

