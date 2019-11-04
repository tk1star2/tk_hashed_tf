import os
import sys
import joblib
import tensorflow as tf
import numpy as np

sys.path.append('..')
from pretrain import train as pretrain
#from kmeans import kmeans_cluster
from XXhash import XXhash

MNIST_PATH='../data/MNIST'
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
assert tf.gfile.Exists("../pretrain/MNIST/MNIST.pkl"), \
          'Cannot find pretrained model at the given path:' \
          '  {}'.format("../pretrain/MNIST/MNIST.pkl")
caffemodel_weight = joblib.load("../pretrain/MNIST/MNIST.pkl")

use_pretrained_param = False

cw = caffemodel_weight
layer_name = "dense1"
flatten = False
xavier = False
stddev = 0.001
hiddens = 1000
WEIGHT_DECAY = 0.0005
print("tk: what is layer name is {}".format(layer_name))
if layer_name in cw:
	print("tk: what is layer name is {}".format(layer_name))
	use_pretrained_param = True
	kernel_val = cw[layer_name][0]
	bias_val = cw[layer_name][1]
	print("how about this {}".format(kernel_val))
	print("how about this {}".format(kernel_val.shape[0])) #784
	print("how about this {}".format(kernel_val.shape[1])) #1000


with tf.variable_scope(layer_name) as scope:
	input_shape = inputs.get_shape().as_list();
	if flatten:
		dim = input_shape[1]*input_shape[2]*input_shape[3]
		inputs = tf.reshape(inputs, [-1, dim])
		if use_pretrained_param:
			try:
				# check the size before layout transform
				assert kernel_val.shape == (hiddens, dim), \
					'kernel shape error at {}'.format(layer_name)
				kernel_val = np.reshape(
					np.transpose(
						np.reshape(
							kernel_val, # O x (C*H*W)
							(hiddens, input_shape[3], input_shape[1], input_shape[2])
						), # O x C x H x W
						(2, 3, 1, 0)
					), # H x W x C x O
					(dim, -1)
				) # (H*W*C) x O
				# check the size after layout transform
				assert kernel_val.shape == (dim, hiddens), \
					'kernel shape error at {}'.format(layer_name)
			except:
				# Do not use pretrained parameter if shape doesn't match
				use_pretrained_param = False
				print ('1.Shape of the pretrained parameter of {} does not match, '
					'use randomly initialized parameter'.format(layer_name))
	else:
		dim = input_shape[1]
		print("shape is {}".format(input_shape[1])) #784
		print("param is {}".format(use_pretrained_param)) #True
		if use_pretrained_param:
			try:
				#kernel_val = np.transpose(kernel_val, (1,0))
				print("kernelval is {}".format(kernel_val.shape)) 
				print("dim, hidden is {},{}".format(dim,hiddens)) 
				assert kernel_val.shape == (dim, hiddens), \
					'kernel shape error at {}'.format(layer_name)
			except:
				use_pretrained_param = False
				print ('2.Shape of the pretrained parameter of {} does not match, '
					'use randomly initialized parameter'.format(layer_name))

	if use_pretrained_param:
		#tk
		#kmeans = kmeans_cluster(kernel_val, max_iter=2)
		kmeans = XXhash(cWeights=kernel_val, nCluster=9000)
		print("Kmeans label is", kmeans.label())
		print("Kmeans cluster_centers is", kmeans.centro())
		print("Kmeans weight is", kmeans.weight())
			
		kernel_init = tf.constant(kmeans.weight(), dtype=tf.float32)
		bias_init = tf.constant(bias_val, dtype=tf.float32)
	elif xavier:
		kernel_init = tf.contrib.layers.xavier_initializer()
		bias_init = tf.constant_initializer(0.0)
	else:
		kernel_init = tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32)
		bias_init = tf.constant_initializer(0.0)
	  
	weights = _variable_with_weight_decay(
		'weights', shape=[dim, hiddens], wd=WEIGHT_DECAY,
		initializer=kernel_init)
	biases = _variable_on_device('biases', [hiddens], bias_init)
	#tk
	model_params = []
	model_params += [weights, biases]

