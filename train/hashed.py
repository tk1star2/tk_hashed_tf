# Author: Bichen Wu (bichen@berkeley.edu) 08/25/2016

"""SqueezeDet model."""
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

from utils import util
from easydict import EasyDict as edict
from nn_skeleton import ModelSkeleton
'''
import tensorflow as tf
import os
import joblib
import sys
import numpy as np
sys.path.append('../test/')
#from kmeans import kmeans_cluster
from XXhash import XXhash

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
    #all here
    #print("-----------------------------------------!!not callable------------------")
    var = tf.get_variable(name, initializer=initializer, trainable=trainable)
  else:
    #print("-----------------------------------------!!callable------------------")
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype, trainable=trainable)
  return var

def _variable_with_weight_decay(name, shape, initializer, wd, trainable=True):
  # wd : weight decay
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

class hashed():
	def __init__(self, mc, gpu_id=0, hashed=True):
		#with tf.device('/cpu:0'):
		with tf.device('/gpu:{}'.format(gpu_id)):
			self.mc = mc
			# a scalar tensor in range (0, 1]. Usually set to 0.5 in training phase and
			# 1.0 in evaluation phase
			self.keep_prob = 0.5 if mc.IS_TRAINING else 1.0

			#---------------------------------------------------------------------------
			# image batch input
			#self.ph_image_input = tf.placeholder(tf.float32, [mc.BATCH_SIZE, mc.IMAGE_HEIGHT, mc.IMAGE_WIDTH, 3],name='image_input')
			self.ph_image_input = tf.placeholder(tf.float32, [mc.BATCH_SIZE, mc.IMAGE_HEIGHT, mc.IMAGE_WIDTH, 1],name='image_input')
		
			# Tensor used to represent labels
			self.ph_labels = tf.placeholder(tf.float32, [mc.BATCH_SIZE, mc.CLASSES], name='labels')

			# image is 1 in MNIST
			#self.FIFOQueue = tf.FIFOQueue(capacity=mc.QUEUE_CAPACITY,dtypes=[tf.float32, tf.float32],shapes=[[mc.IMAGE_HEIGHT, mc.IMAGE_WIDTH, 3],[mc.CLASSES]])
			self.FIFOQueue = tf.FIFOQueue(capacity=mc.QUEUE_CAPACITY,dtypes=[tf.float32, tf.float32],shapes=[[mc.IMAGE_HEIGHT, mc.IMAGE_WIDTH, 1],[mc.CLASSES]])

			self.enqueue_op = self.FIFOQueue.enqueue_many([self.ph_image_input, self.ph_labels])

			self.image_input, self.labels = tf.train.batch(self.FIFOQueue.dequeue(), batch_size=mc.BATCH_SIZE, capacity=mc.QUEUE_CAPACITY) 
			#---------------------------------------------------------------------------

			# model parameters
			self.model_params = []

			# model size counter
			self.model_size_counter = [] # array of tuple of layer name, parameter size

			# flop counter
			self.flop_counter = [] # array of tuple of layer name, flop number
			
			# activation counter
			self.activation_counter = [] # array of tuple of layer name, output activations
			self.activation_counter.append(('input', mc.IMAGE_WIDTH*mc.IMAGE_HEIGHT*1))
			#---------------------------------------------------------------------------
			self.hash_index = {};
			self.hash_num = {};
	

			#make Tensor
			self._add_forward_graph()
			print("debug1.................................................add_forward__graph : end")
			self._add_loss_graph()
			print("debug2.................................................add_loss_graph : end")
			if hashed:
				self._add_hash_train_graph()
			else:
				self._add_train_graph()
			print("debug3...............................................add_train_graph : end")

	# ***************************************step1****************************
	#-------------------STEP1. cost function for loss---------------------------------
		#save_dict = {'W1' : W1_arr, 'B1' : B1_arr, 'W2' : W2_arr, 'B2' : B2_arr};

		#case1
		#dense1 = self._fc_layer('dense1', self.image_input, hiddens=1000, flatten=True)
		#self.preds = self._fc_layer('dense2', dense1, hiddens=10, flatten=False, relu=False)

		#case2
		#dense1 = self._hashed_fc_layer('dense1', self.image_input, hiddens=1000, flatten=True, centroid_num=98000)
		#self.preds = self._hashed_fc_layer('dense2', dense1, hiddens=10, flatten=False, relu=False, centroid_num=1250)

		#case3
		#dense2 = self._fc_layer('dense2', dense1, hiddens=10, flatten=False)
		#self.preds = tf.nn.dropout(dense2, self.keep_prob, name='drop3')
	#---------------------------------------------------------------------------------
	def _add_forward_graph(self):
		"""NN architecture."""

		mc = self.mc

		print("mc.LOAD_PRETRAINED_MODEL is {}".format(mc.LOAD_PRETRAINED_MODEL))
		
		# pre-define model
		if mc.LOAD_PRETRAINED_MODEL:
			assert tf.gfile.Exists(mc.PRETRAINED_MODEL_PATH), 'Cannot find pretrained model at the given path:' '  {}'.format(mc.PRETRAINED_MODEL_PATH)
		self.caffemodel_weight = joblib.load(mc.PRETRAINED_MODEL_PATH)

		dense1 = self._hashed_fc_layer('dense1', self.image_input, hiddens=1000, flatten=True, centroid_num=98000, blocked=True, blocked_param=64)

		self.preds = self._hashed_fc_layer('dense2', dense1, hiddens=10, flatten=False, relu=False, centroid_num=12500, blocked=True, blocked_param=5)

		#preds :(100, 10)
		#print("tk :preds is this {}".format(self.preds))


	# ***************************************step2****************************
	#-------------------STEP2. cost function for loss---------------------------------
		'''
		with tf.variable_scope('class_regression') as scope:  #11111
			# cross-entropy: q * -log(p) + (1-q) * -log(1-p)

			# add a small value into log to prevent blowing up : preds,labels-(100,10)
			self.class_loss = tf.reduce_sum((self.labels*(-tf.log(self.preds + mc.EPSILON)) + (1-self.labels)*(-tf.log(1-self.preds+mc.EPSILON))) * mc.LOSS_COEF_CLASS)

			tf.add_to_collection('losses', self.class_loss)

		# add above losses as well as weight decay losses to form the total loss
		# tk:accumulate
		self.loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
		'''
	#---------------------------------------------------------------------------------
	def _add_loss_graph(self):
		"""Define the loss operation."""
		mc = self.mc

		with tf.variable_scope('class_regression') as scope:  #11111
			self.class_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.preds, labels=self.labels))
			tf.add_to_collection('losses', self.class_loss)
					

		self.loss = tf.add_n(tf.get_collection('losses'), name='total_loss');
		#self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.preds, labels=self.labels))

	# ***************************************step3****************************
	#-------------------STEP3. train mapping for loss---------------------------------
		#_add_loss_summaries(self.loss)

		#1.trainable variables---------------
		#tf.Variable, dense1/weights:0 : (784,1000)
		#tf.Variable, dense1/biases:0  : (1000, )
		#tf.Variable, dense2/weights:0 : (1000,10)
		#tf.Variable, dense2/biases:0  : (10, )

		#2.grads_vars-------------------------
		#tf.Tensor, control_dependency : (784,1000) grad
		#tf.Variable, dense1/weights:0 : (784,1000) vars

		#tf.Tensor, control_dependency : (1000, ) grad
		#tf.Variable, dense1/biases:0  : (1000, ) vars

		#tf.Tensor, control_dependency : (1000,10) grad
		#tf.Variable, dense2/weights:0 : (1000,10) vars

		#tf.Tensor, control_dependency : (10, ) grad
		#tf.Variable, dense2/biases:0  : (10, ) vars

		# apply grad_vars
		#apply_gradient_op = opt.apply_gradients(grads_vars, global_step=self.global_step)
	#---------------------------------------------------------------------------------
	

	def _add_hash_train_graph(self):
		#Define the training operation.
		mc = self.mc

		self.global_step = tf.Variable(0, name='global_step', trainable=False)
		# decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
		lr = tf.train.exponential_decay(mc.LEARNING_RATE,self.global_step,mc.DECAY_STEPS,mc.LR_DECAY_FACTOR, staircase=True)

		tf.summary.scalar('learning_rate', lr)

		#Momentum + learning decay + weight decay
		opt = tf.train.MomentumOptimizer(learning_rate=lr, momentum=mc.MOMENTUM)
		self.grads_vars = opt.compute_gradients(self.loss, tf.trainable_variables())


		with tf.variable_scope('clip_gradient') as scope:
			print("tk:grads_vars is ", self.grads_vars)
			for i, (grad, var) in enumerate(self.grads_vars):
				print("{} trainable variable is !!!grad: {}".format(i, grad))
				print("{} trainable variable is !!!var: {}".format(i, var))
				self.grads_vars[i] = (tf.clip_by_norm(grad, mc.MAX_GRAD_NORM), var)
		
			#SETTING
			self.grads_placeholder = [(tf.placeholder("float", shape=var.get_shape()),var) for (grad, var) in self.grads_vars]

		#with tf.control_dependencies([grads_vars]):
		#	self.train_op1 = tf.no_op(name='train1')

		#----------------------------------------------------------

		# modify grad_vars
		apply_gradient_op = opt.apply_gradients(self.grads_placeholder, global_step=self.global_step)


		for var in tf.trainable_variables():
			tf.summary.histogram(var.op.name, var)
		for grad, var in self.grads_vars:
			if grad is not None:
				tf.summary.histogram(var.op.name + '/gradients', grad)

		with tf.control_dependencies([apply_gradient_op]):
			self.train_op2 = tf.no_op(name='train2')
	
	#original
	def _add_train_graph(self):
		"""Define the training operation."""
		mc = self.mc


		
		self.global_step = tf.Variable(0, name='global_step', trainable=False)
		# decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
		lr = tf.train.exponential_decay(mc.LEARNING_RATE,self.global_step,mc.DECAY_STEPS,mc.LR_DECAY_FACTOR, staircase=True)

		tf.summary.scalar('learning_rate', lr)


		
		opt = tf.train.MomentumOptimizer(learning_rate=lr, momentum=mc.MOMENTUM)
		grads_vars = opt.compute_gradients(self.loss, tf.trainable_variables())
		#print("-------------------------------------tk:classloss:{}---------------------------------------".format(grads_vars))
	
		print("grad_vars size is ", grads_vars[0][0])
			
		with tf.variable_scope('clip_gradient') as scope:
			for i, (grad, var) in enumerate(grads_vars):
				print("{} trainable variable is !!!var: {}".format(i, grad))
				print("{} trainable variable is !!!var: {}".format(i, var))
				grads_vars[i] = (tf.clip_by_norm(grad, mc.MAX_GRAD_NORM), var)
				#print("{} trainable variable is !!!var: {}".format(i, grads_vars[i]))
				#grads_vars[i] = (lambda grad: 0 if grad is None	else tf.clip_by_norm(grad, mc.MAX_GRAD_NORM), var)
		

		# apply grad_vars
		apply_gradient_op = opt.apply_gradients(grads_vars, global_step=self.global_step)
		
		
		for var in tf.trainable_variables():
			tf.summary.histogram(var.op.name, var)
		for grad, var in grads_vars:
			if grad is not None:
				tf.summary.histogram(var.op.name + '/gradients', grad)
		
		#apply_gradient_op = tf.train.AdamOptimizer(0.001).minimize(self.loss);

		with tf.control_dependencies([apply_gradient_op]):
			self.train_op = tf.no_op(name='train')



	#-------------------------------layer_start----------------------------------
	#---------------------------------------------------------------------------
		# HASH-FCLayer
		"""Fully connected layer operation constructor.
		Args:
			layer_name: layer name.
			inputs: input tensor
			hiddens: number of (hidden) neurons in this layer.
			flatten: if true, reshape the input 4D tensor of shape 
				(batch, height, weight, channel) into a 2D tensor with shape 
				(batch, -1). This is used when the input to the fully connected layer
				is output of a convolutional layer.
			relu: whether to use relu or not.
			xavier: whether to use xavier weight initializer or not.
			stddev: standard deviation used for random weight initializer.
		Returns:
			A fully connected layer operation.
		"""
	#---------------------------------------------------------------------------
	def _hashed_fc_layer(
      self, layer_name, inputs, hiddens, flatten=False, relu=True,
      xavier=False, stddev=0.001, centroid_num=30, hashed = True, blocked=False, blocked_param=64):
		mc = self.mc

		use_pretrained_param = False
		if mc.LOAD_PRETRAINED_MODEL:
			cw = self.caffemodel_weight
			if layer_name in cw:
				use_pretrained_param = True
				kernel_val = np.transpose(cw[layer_name][0])
				bias_val = cw[layer_name][1]

		with tf.variable_scope(layer_name) as scope:
			input_shape = inputs.get_shape().as_list()
			if flatten:#flatten
				#              28			28			1
				dim = input_shape[1]*input_shape[2]*input_shape[3]
				inputs = tf.reshape(inputs, [-1, dim])
				if use_pretrained_param:
					try:
						# check the size before layout transf hiddens-1000, dim 784
						assert kernel_val.shape == (hiddens, dim), 'kernel shape error at {}'.format(layer_name)
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
						print ('Shape of the pretrained parameter of {} does not match, ' 'use randomly initialized parameter'.format(layer_name))
			else:#no flatten
				dim = input_shape[1]
				if use_pretrained_param:
					try:
						kernel_val = np.transpose(kernel_val, (1,0))
						# check the size before layout transf dim 784, hiddens-1000
						assert kernel_val.shape == (dim, hiddens), 'kernel shape error at {}'.format(layer_name)
					except:
						use_pretrained_param = False
						print ('Shape of the pretrained parameter of {} does not match, ' 'use randomly initialized parameter'.format(layer_name))

			kmeans = XXhash(cWeights=kernel_val, nCluster=centroid_num, blocked=blocked, blocked_param=blocked_param)
			self.hash_index[layer_name] = kmeans.label();
			self.hash_num[layer_name] = kmeans.num_centro();
			#print("tk: kmeans weight size is ", kmeans.weight().shape)

			if use_pretrained_param:
				print("1.TKTKTKTK:::::::", kernel_val)
				print("2.TKTKTKTK:::::::", kmeans.weight())
				kernel_init = tf.constant(kmeans.weight(), dtype=tf.float32)
				bias_init = tf.constant(bias_val, dtype=tf.float32)
			elif xavier:
				kernel_init = tf.contrib.layers.xavier_initializer()
				bias_init = tf.constant_initializer(0.0)
			else:
				print("ERROR!!!!!!!!!!!!!!!!!!!!!TK");
				print("ERROR!!!!!!!!!!!!!!!!!!!!!TK");
				print("ERROR!!!!!!!!!!!!!!!!!!!!!TK");
				print("ERROR!!!!!!!!!!!!!!!!!!!!!TK");
				kernel_init = tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32)
				bias_init = tf.constant_initializer(0.0)
	  
			#print("tk: dim is {}, hiddens is {}".format(dim, hiddens))

			#tk : MAKE TENSOR!!!!
			weights = _variable_with_weight_decay('weights', shape=[dim, hiddens], initializer=kernel_init, wd=mc.WEIGHT_DECAY)
			#weights = _variable_on_device('weights', shape=[dim, hiddens], initializer=kernel_init)
			biases = _variable_on_device('biases', [hiddens], bias_init)
			self.model_params += [weights, biases]
  
			outputs = tf.nn.bias_add(tf.matmul(inputs, weights), biases)

			if relu:
				outputs = tf.nn.relu(outputs, 'relu')
			#!!!!!!finished!!!!!!!!!!!!!!!! kernel is fixed

			# count layer stats-----------------------------------------------
			self.model_size_counter.append((layer_name, (dim+1)*hiddens))

			num_flops = 2 * dim * hiddens + hiddens
			if relu:
				num_flops += 2*hiddens
			self.flop_counter.append((layer_name, num_flops))

			self.activation_counter.append((layer_name, hiddens))
			#-----------------------------------------------------------------

			return outputs

	#---------------------------------------------------------------------------
		# FC layer
		"""Fully connected layer operation constructor.
		Args:
			layer_name: layer name.
			inputs: input tensor
			hiddens: number of (hidden) neurons in this layer.
			flatten: if true, reshape the input 4D tensor of shape 
				(batch, height, weight, channel) into a 2D tensor with shape 
				(batch, -1). This is used when the input to the fully connected layer
				is output of a convolutional layer.
			relu: whether to use relu or not.
			xavier: whether to use xavier weight initializer or not.
			stddev: standard deviation used for random weight initializer.
		Returns:
			A fully connected layer operation.
		"""
	#---------------------------------------------------------------------------
	def _fc_layer(
      self, layer_name, inputs, hiddens, flatten=False, relu=True,
      xavier=False, stddev=0.001):
		mc = self.mc

		use_pretrained_param = False
		if mc.LOAD_PRETRAINED_MODEL:
			cw = self.caffemodel_weight
			if layer_name in cw:
				use_pretrained_param = True
				#kernel_val = cw[layer_name][0]
				kernel_val = np.transpose(cw[layer_name][0])
				bias_val = cw[layer_name][1]
				#print("kernel_val {} is {}".format(kernel_val.shape, kernel_val))
				#print("bias_val {} is {}".format(bias_val.shape, bias_val))


		with tf.variable_scope(layer_name) as scope:
			input_shape = inputs.get_shape().as_list()
			if flatten:#flatten
				print("----------------------2!!!flatten------------------{},{},{}",input_shape[1], input_shape[2], input_shape[3])
				#              28			28			1
				dim = input_shape[1]*input_shape[2]*input_shape[3]
				inputs = tf.reshape(inputs, [-1, dim])

				print("----------------------2!!!flatten------------------{}",kernel_val.shape)
				if use_pretrained_param:
					try:
						# check the size before layout transf hiddens-1000, dim 784
						assert kernel_val.shape == (hiddens, dim), 'kernel shape error at {}'.format(layer_name)
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
						assert kernel_val.shape == (dim, hiddens),'kernel shape error at {}'.format(layer_name)
					except:
						# Do not use pretrained parameter if shape doesn't match
						use_pretrained_param = False
						print ('1.Shape of the pretrained parameter of {} does not match, '
							'use randomly initialized parameter'.format(layer_name))
			else:#no flatten
				dim = input_shape[1]
				if use_pretrained_param:
					try:
						kernel_val = np.transpose(kernel_val, (1,0))
						assert kernel_val.shape == (dim, hiddens), 'kernel shape error at {}'.format(layer_name)
					except:
						use_pretrained_param = False
						print ('2.Shape of the pretrained parameter of {} does not match, '
							'use randomly initialized parameter'.format(layer_name))

			#print("kernel_val {} is {}".format(kernel_val.shape, kernel_val))
			#print("bias_val {} is {}".format(bias_val.shape, bias_val))
			#---------------------------------------------------------------------------
			if use_pretrained_param:
				kernel_init = tf.constant(kernel_val, dtype=tf.float32)
				bias_init = tf.constant(bias_val, dtype=tf.float32)
			elif xavier:
				kernel_init = tf.contrib.layers.xavier_initializer()
				bias_init = tf.constant_initializer(0.0)
			else:
				kernel_init = tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32)
				bias_init = tf.constant_initializer(0.0)
	  
			#weights = _variable_with_weight_decay('weights', shape=[dim, hiddens], initializer=kernel_init, wd=mc.WEIGHT_DECAY)
			weights = _variable_on_device('weights', shape=[dim, hiddens], initializer=kernel_init)
			biases = _variable_on_device('biases', shape=[hiddens], initializer=bias_init)
			self.model_params += [weights, biases]
  
			#here!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!11
			outputs = tf.nn.bias_add(tf.matmul(inputs, weights), biases)
				
			if relu:
				outputs = tf.nn.relu(outputs, 'relu')

			# count layer stats-----------------------------------------------
			self.model_size_counter.append((layer_name, (dim+1)*hiddens))

			num_flops = 2 * dim * hiddens + hiddens
		
			if relu:
				num_flops += 2*hiddens
			self.flop_counter.append((layer_name, num_flops))

			self.activation_counter.append((layer_name, hiddens))
			#-----------------------------------------------------------------

			return outputs


	#---------------------------------------------------------------------------
	# hashed CONV layer
	"""Convolutional layer operation constructor.
		Args:
			layer_name: layer name.
			inputs: input tensor
			filters: number of output filters.
			size: kernel size.
			stride: stride
			padding: 'SAME' or 'VALID'. See tensorflow doc for detailed description.
			freeze: if true, then do not train the parameters in this layer.
			xavier: whether to use xavier weight initializer or not.
			relu: whether to use relu or not.
			stddev: standard deviation used for random weight initializer.
		Returns:
			A convolutional layer operation.
	"""

	#---------------------------------------------------------------------------
	def _hashed_conv_layer(
		self, layer_name, inputs, filters, size, stride, padding='SAME', hashed=True,
		freeze=False, xavier=False, relu=True, stddev=0.001, centroid_num=32):

		mc = self.mc
		use_pretrained_param = False
		if mc.LOAD_PRETRAINED_MODEL:
			cw = self.caffemodel_weight
			if layer_name in cw:
				kernel_val = np.transpose(cw[layer_name][0], [2,3,1,0])
				bias_val = cw[layer_name][1]
				# check the shape
				if (kernel_val.shape == (size, size, inputs.get_shape().as_list()[-1], filters)) \
					and (bias_val.shape == (filters, )):
					use_pretrained_param = True
				else:
					print ('Shape of the pretrained parameter of {} does not match, '
						'use randomly initialized parameter'.format(layer_name))
			else:
				print ('Cannot find {} in the pretrained model. Use randomly initialized '
					'parameters'.format(layer_name))

		if mc.DEBUG_MODE:
			print('Input tensor shape to {}: {}'.format(layer_name, inputs.get_shape()))

		with tf.variable_scope(layer_name) as scope:
			channels = inputs.get_shape()[3]

			kmeans = XXhash(cWeights=kernel_val, nCluster=centroid_num)
			self.hash_index[layer_name] = kmeans.label();
			self.hash_num[layer_name] = kmeans.num_centro();
			#print("tk: kmeans weight size is ", kmeans.weight().shape)

			#TO_DO :  check dimension
			if use_pretrained_param:
				print("1.TKTKTKTK:::::::", kernel_val)
				print("2.TKTKTKTK:::::::", kmeans.weight())
				kernel_init = tf.constant(kmeans.weight(), dtype=tf.float32)
				bias_init = tf.constant(bias_val, dtype=tf.float32)
			elif xavier:
				kernel_init = tf.contrib.layers.xavier_initializer()
				bias_init = tf.constant_initializer(0.0)
			else:
				print("ERROR!!!!!!!!!!!!!!!!!!!!!TK");
				print("ERROR!!!!!!!!!!!!!!!!!!!!!TK");
				print("ERROR!!!!!!!!!!!!!!!!!!!!!TK");
				print("ERROR!!!!!!!!!!!!!!!!!!!!!TK");
				kernel_init = tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32)
				bias_init = tf.constant_initializer(0.0)

			# re-order the caffe kernel with shape [out, in, h, w] -> tf kernel with
			# shape [h, w, in, out]

			kernel = _variable_with_weight_decay(
				'kernels', shape=[size, size, int(channels), filters],
				wd=mc.WEIGHT_DECAY, initializer=kernel_init, trainable=(not freeze))

			biases = _variable_on_device('biases', [filters], bias_init, trainable=(not freeze))
			self.model_params += [kernel, biases]

			#-------------------real tensorflow convolution 2d-------------------------------
			print("inputs?? {}".format(inputs))
			print("kernel?? {}".format(kernel))
			conv = tf.nn.conv2d(inputs, kernel, [1, stride, stride, 1], padding=padding, name='convolution')
			print("conv?? {}".format(conv))
			#--------------------------------------------------------------------------------

			conv_bias = tf.nn.bias_add(conv, biases, name='bias_add')
 
			if relu:
				out = tf.nn.relu(conv_bias, 'relu')
			else:
				out = conv_bias

			self.model_size_counter.append((layer_name, (1+size*size*int(channels))*filters))
			out_shape = out.get_shape().as_list()
			num_flops = (1+2*int(channels)*size*size)*filters*out_shape[1]*out_shape[2]

			if relu:
				num_flops += 2*filters*out_shape[1]*out_shape[2]
			self.flop_counter.append((layer_name, num_flops))
	
			self.activation_counter.append((layer_name, out_shape[1]*out_shape[2]*out_shape[3]))

			return out
	#---------------------------------------------------------------------------
	# CONV layer
	"""Convolutional layer operation constructor.
		Args:
			layer_name: layer name.
			inputs: input tensor
			filters: number of output filters.
			size: kernel size.
			stride: stride
			padding: 'SAME' or 'VALID'. See tensorflow doc for detailed description.
			freeze: if true, then do not train the parameters in this layer.
			xavier: whether to use xavier weight initializer or not.
			relu: whether to use relu or not.
			stddev: standard deviation used for random weight initializer.
		Returns:
			A convolutional layer operation.
	"""

	#---------------------------------------------------------------------------
	def _conv_layer(
		self, layer_name, inputs, filters, size, stride, padding='SAME',
		freeze=False, xavier=False, relu=True, stddev=0.001):

		mc = self.mc
		use_pretrained_param = False
		if mc.LOAD_PRETRAINED_MODEL:
			cw = self.caffemodel_weight
			if layer_name in cw:
				kernel_val = np.transpose(cw[layer_name][0], [2,3,1,0])
				bias_val = cw[layer_name][1]
				# check the shape
				if (kernel_val.shape == (size, size, inputs.get_shape().as_list()[-1], filters)) \
					and (bias_val.shape == (filters, )):
					use_pretrained_param = True
				else:
					print ('Shape of the pretrained parameter of {} does not match, '
						'use randomly initialized parameter'.format(layer_name))
			else:
				print ('Cannot find {} in the pretrained model. Use randomly initialized '
					'parameters'.format(layer_name))

		if mc.DEBUG_MODE:
			print('Input tensor shape to {}: {}'.format(layer_name, inputs.get_shape()))

		with tf.variable_scope(layer_name) as scope:
			channels = inputs.get_shape()[3]

			# re-order the caffe kernel with shape [out, in, h, w] -> tf kernel with
			# shape [h, w, in, out]
			if use_pretrained_param:
				if mc.DEBUG_MODE:
					print ('Using pretrained model for {}'.format(layer_name))
				kernel_init = tf.constant(kernel_val , dtype=tf.float32)
				bias_init = tf.constant(bias_val, dtype=tf.float32)
			elif xavier:
				kernel_init = tf.contrib.layers.xavier_initializer_conv2d()
				bias_init = tf.constant_initializer(0.0)
			else:
				kernel_init = tf.truncated_normal_initializer(
					stddev=stddev, dtype=tf.float32)
				bias_init = tf.constant_initializer(0.0)

			kernel = _variable_with_weight_decay(
				'kernels', shape=[size, size, int(channels), filters],
				wd=mc.WEIGHT_DECAY, initializer=kernel_init, trainable=(not freeze))

			biases = _variable_on_device('biases', [filters], bias_init, trainable=(not freeze))
			self.model_params += [kernel, biases]

			#-------------------real tensorflow convolution 2d-------------------------------
			print("inputs?? {}".format(inputs))
			print("kernel?? {}".format(kernel))
			conv = tf.nn.conv2d(inputs, kernel, [1, stride, stride, 1], padding=padding, name='convolution')
			print("conv?? {}".format(conv))
			#--------------------------------------------------------------------------------

			conv_bias = tf.nn.bias_add(conv, biases, name='bias_add')
 
			if relu:
				out = tf.nn.relu(conv_bias, 'relu')
			else:
				out = conv_bias

			self.model_size_counter.append((layer_name, (1+size*size*int(channels))*filters))
			out_shape = out.get_shape().as_list()
			num_flops = (1+2*int(channels)*size*size)*filters*out_shape[1]*out_shape[2]

			if relu:
				num_flops += 2*filters*out_shape[1]*out_shape[2]
			self.flop_counter.append((layer_name, num_flops))
	
			self.activation_counter.append((layer_name, out_shape[1]*out_shape[2]*out_shape[3]))

			return out
