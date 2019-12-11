import os
import sys
import joblib
import tensorflow as tf
import numpy as np

sys.path.append('../1.pretrain')
#import train_fc_MNIST as pretrain
import train_conv_MNIST as pretrain
sys.path.append('../requirement')
from T2_XXhash import XXhash

MNIST_PATH='../0.data/MNIST'
VALIDATION_SIZE = 5000

#------------1. DATA--------
train_images = pretrain.MNIST_download(MNIST_PATH, 'train', 'images');
train_labels = pretrain.MNIST_download(MNIST_PATH, 'train', 'labels');
test_images =  pretrain.MNIST_download(MNIST_PATH, 'test', 'images');
test_labels =  pretrain.MNIST_download(MNIST_PATH, 'test', 'labels');

validation_images = train_images[:VALIDATION_SIZE];
validation_labels = train_labels[:VALIDATION_SIZE];
train_images = train_images[VALIDATION_SIZE:];
train_labels = train_labels[VALIDATION_SIZE:];

print('train-labels is ', train_labels);

#data_sets = DataSet([],[],fake_data=True, one_hot=True, dtype=tf.float32);
def _variable_on_device(name, shape, initializer, trainable=True):
  """Helper to create a Variable.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable 
  Returns:
    Variable Tensor
  """
  # TODO(bichen): fix the hard-coded data type below
  dtype = tf.float32
  if not callable(initializer):
    var = tf.get_variable(name, initializer=initializer, trainable=trainable)
  else:
    var = tf.get_variable(
        name, shape, initializer=initializer, dtype=dtype, trainable=trainable)
  return var

def _variable_with_weight_decay(name, shape, wd, initializer, trainable=True):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

  Returns:
    Variable Tensor
  """
  var = _variable_on_device(name, shape, initializer, trainable)
  if wd is not None and trainable:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var

class DataSets(object):
	pass
data_sets = DataSets();
data_sets.test =  pretrain.DataSet(test_images, test_labels, dtype=tf.float32, one_hot=True);
data_sets.train =  pretrain.DataSet(train_images, train_labels, dtype=tf.float32, one_hot=True);
data_sets.validation =  pretrain.DataSet(validation_images, validation_labels, dtype=tf.float32, one_hot=True);
a, b = data_sets.train.next_batch(100)
print('***datasets train is ', a.shape[0]);#100 Batchsize
print('***datasets train is ', a.shape[1]);#784 input
print('***datasets train is ', b.shape[0]);#100 Batchsize
print('***datasets train is ', b.shape[1]);#10 output
inputs_np, _ = data_sets.train.next_batch(1)
inputs = tf.convert_to_tensor(inputs_np, tf.float32)
#------------------------------------------------------------------------


# pre-define model
#assert tf.gfile.Exists("../1.pretrain/MNIST/MNIST_fc.pkl"), \
#          'Cannot find pretrained model at the given path:' \
#          '  {}'.format("../1.pretrain/MNIST/MNIST_fc.pkl")
#caffemodel_weight = joblib.load("../1.pretrain/MNIST/MNIST_fc.pkl")
assert tf.gfile.Exists("../1.pretrain/MNIST/MNIST_conv.pkl"), \
          'Cannot find pretrained model at the given path:' \
          '  {}'.format("../1.pretrain/MNIST/MNIST_conv.pkl")
caffemodel_weight = joblib.load("../1.pretrain/MNIST/MNIST_conv.pkl")

use_pretrained_param = False

cw = caffemodel_weight
layer_name = "conv1"
flatten = False
xavier = False
stddev = 0.001
hiddens = 1000
WEIGHT_DECAY = 0.0005
size = 3;
filters = 32;
print("tk: what is layer name is {}".format(layer_name))
if (len(cw[layer_name])!=1) :
	BIAS_USE = True;
else :
	BIAS_USE = False;

if layer_name in cw:
	use_pretrained_param = True
	kernel_val = cw[layer_name][0]
	#bias_val = cw[layer_name][1]
	#print("how about this {}".format(kernel_val))
	print("how about this {}".format(kernel_val.shape[0])) #784
	#print("how about this {}".format(kernel_val.shape[1])) #1000

if layer_name in cw:
	print("tk: what is layer name is {}".format(layer_name))
	kernel_val = cw[layer_name][0];
	if (BIAS_USE) :
		bias_val = cw[layer_name][1];
	# check the shape
	print("size{}, size{}, size{}, size{} ".format(size,size,inputs.get_shape().as_list()[-1], filters));
	print("kernel_val.shape is {}".format(kernel_val.shape))
	print("inputs.get_shape() is {}".format(inputs.get_shape()))
	if (kernel_val.shape == (size, size, inputs.get_shape().as_list()[-1], filters)) \
		and (BIAS_USE==False or bias_val.shape == (filters, )):
		use_pretrained_param = True
	else:
		print ('Shape of the pretrained parameter of {} does not match, '
			'use randomly initialized parameter'.format(layer_name))
else:
	print ('Cannot find {} in the pretrained model. Use randomly initialized '
		'parameters'.format(layer_name))

with tf.variable_scope(layer_name) as scope:
	channels = inputs.get_shape()[3]

	kmeans = XXhash(Conv_FC="conv", cWeights=kernel_val, nCluster=128)
	#kmeans = XXhash(Conv_FC="conv", cWeights=kernel_val, nCluster=9800, blocked=True, blocked_param=64)
	#print("tk: kmeans weight size is ", kmeans.weight().shape)
	print("Kmeans label is", kmeans.label())
	print("Kmeans cluster_centers is", kmeans.centro())
	print("Kmeans weight is", kmeans.weight())

	#TO_DO :  check dimension
	if use_pretrained_param:
		print("1.TKTKTKTK:::::::", kernel_val)
		print("2.TKTKTKTK:::::::", kmeans.weight())
		kernel_init = tf.constant(kmeans.weight(), dtype=tf.float32)
		if (BIAS_USE) :
			bias_init = tf.constant(bias_val, dtype=tf.float32)
	elif xavier:
		kernel_init = tf.contrib.layers.xavier_initializer()
		if (BIAS_USE) :
			bias_init = tf.constant_initializer(0.0)
	else:
		print("ERROR!!!!!!!!!!!!!!!!!!!!!TK");
		print("ERROR!!!!!!!!!!!!!!!!!!!!!TK");
		print("ERROR!!!!!!!!!!!!!!!!!!!!!TK");
		print("ERROR!!!!!!!!!!!!!!!!!!!!!TK");
		kernel_init = tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32)
		if (BIAS_USE) :
			bias_init = tf.constant_initializer(0.0)
	# re-order the caffe kernel with shape [out, in, h, w] -> tf kernel with
	# shape [h, w, in, out]

	kernel = _variable_with_weight_decay(
		'weights', shape=[size, size, int(channels), filters],
		wd=WEIGHT_DECAY, initializer=kernel_init, trainable=True)

	if (BIAS_USE) :
		biases = _variable_on_device('biases', [filters], bias_init, trainable=True)

	model_params = []
	if (BIAS_USE) :
		model_params += [kernel, biases]
	else :
		model_params += [kernel]
				
