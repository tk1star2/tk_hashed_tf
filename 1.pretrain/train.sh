#rm ./MNIST/*
find ./MNIST/ -type f -not -name "*.pkl" -delete
#python3 train_fc_MNIST.py 
python3 train_conv_MNIST.py 
