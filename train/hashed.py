# Author: Bichen Wu (bichen@berkeley.edu) 08/25/2016

"""SqueezeDet model."""
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

from utils import util
from easydict import EasyDict as edict
import numpy as np
from nn_skeleton import ModelSkeleton
'''
import tensorflow as tf
import os
import joblib
import sys
#from kmeans import kmeans_cluster
sys.path.append('../test/')
from kmeans_hash import kmeans_hash_cluster

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

class hashed():
	def __init__(self, mc, gpu_id=0):
		#with tf.device('/cpu:0'):
		with tf.device('/gpu:{}'.format(gpu_id)):
			self.mc = mc
			# a scalar tensor in range (0, 1]. Usually set to 0.5 in training phase and
			# 1.0 in evaluation phase
			self.keep_prob = 0.5 if mc.IS_TRAINING else 1.0

			# image batch input
			self.ph_image_input = tf.placeholder(tf.float32, [mc.BATCH_SIZE, mc.IMAGE_HEIGHT, mc.IMAGE_WIDTH, 3],name='image_input')
		
			# Tensor used to represent labels
			self.ph_labels = tf.placeholder(tf.float32, [mc.BATCH_SIZE, mc.CLASSES], name='labels')

			self.FIFOQueue = tf.FIFOQueue(capacity=mc.QUEUE_CAPACITY,dtypes=[tf.float32, tf.float32],shapes=[[mc.IMAGE_HEIGHT, mc.IMAGE_WIDTH, 3],[mc.CLASSES]])

			self.enqueue_op = self.FIFOQueue.enqueue_many([self.ph_image_input, self.ph_labels])

			self.image_input, self.labels = tf.train.batch(self.FIFOQueue.dequeue(), batch_size=mc.BATCH_SIZE, capacity=mc.QUEUE_CAPACITY) 


			# model parameters
			self.model_params = []

			# model size counter
			self.model_size_counter = [] # array of tuple of layer name, parameter size

			# flop counter
			self.flop_counter = [] # array of tuple of layer name, flop number
			
			# activation counter
			self.activation_counter = [] # array of tuple of layer name, output activations
			self.activation_counter.append(('input', mc.IMAGE_WIDTH*mc.IMAGE_HEIGHT*3))

		self._add_forward_graph()
		print("debug1.................................................add_forward__graph : end")
		self._add_loss_graph()
		print("debug2.................................................add_loas_graph : end")
		self._add_train_graph()
		print("debug3...............................................add_train_graph : end")

	def _add_forward_graph(self):
		"""NN architecture."""

		mc = self.mc
		#-----------------------------------------tk---------------------

		print("mc.LOAD_PRETRAINED_MODEL is {}".format(mc.LOAD_PRETRAINED_MODEL))
		
		# pre-define model
		if mc.LOAD_PRETRAINED_MODEL:
			assert tf.gfile.Exists(mc.PRETRAINED_MODEL_PATH), 'Cannot find pretrained model at the given path:' '  {}'.format(mc.PRETRAINED_MODEL_PATH)
		self.caffemodel_weight = joblib.load(mc.PRETRAINED_MODEL_PATH)

		dense1 = self._hashed_layer('dense1', self.image_input, hiddens=1000, flatten=True)
		dense2 = self._hashed_layer('dense2', dense1, hiddens=10, flatten=False)
		self.preds = tf.nn.dropout(dense2, self.keep_prob, name='drop3')
		#----------------------------------------------------------------

	#---------------------------------------------------------------------------
	def _add_loss_graph(self):
		"""Define the loss operation."""
		mc = self.mc

		with tf.variable_scope('class_regression') as scope:  #11111
			# cross-entropy: q * -log(p) + (1-q) * -log(1-p)

			# add a small value into log to prevent blowing up
			self.class_loss = tf.truediv(
				tf.reduce_sum((self.labels*(-tf.log(self.pred_class_probs+mc.EPSILON)) + (1-self.labels)*(-tf.log(1-self.pred_class_probs+mc.EPSILON))) * mc.LOSS_COEF_CLASS), self.num_objects, name='class_loss')
		tf.add_to_collection('losses', self.class_loss)
		#tk : box_input, labels

		# add above losses as well as weight decay losses to form the total loss
		self.loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

	#---------------------------------------------------------------------------
	def _add_train_graph(self):
		"""Define the training operation."""
		mc = self.mc

		self.global_step = tf.Variable(0, name='global_step', trainable=False)
		lr = tf.train.exponential_decay(mc.LEARNING_RATE,
                                    self.global_step,
                                    mc.DECAY_STEPS,
                                    mc.LR_DECAY_FACTOR,
                                    staircase=True)

		tf.summary.scalar('learning_rate', lr)

		#_add_loss_summaries(self.loss)

		opt = tf.train.MomentumOptimizer(learning_rate=lr, momentum=mc.MOMENTUM)
		grads_vars = opt.compute_gradients(self.loss, tf.trainable_variables())

		with tf.variable_scope('clip_gradient') as scope:
			for i, (grad, var) in enumerate(grads_vars):
				grads_vars[i] = (tf.clip_by_norm(grad, mc.MAX_GRAD_NORM), var)
				#grads_vars[i] = (lambda grad: 0 if grad is None	else tf.clip_by_norm(grad, mc.MAX_GRAD_NORM), var)

		# apply grad_vars
		apply_gradient_op = opt.apply_gradients(grads_vars, global_step=self.global_step)

		for var in tf.trainable_variables():
			tf.summary.histogram(var.op.name, var)

		for grad, var in grads_vars:
			if grad is not None:
				tf.summary.histogram(var.op.name + '/gradients', grad)

		with tf.control_dependencies([apply_gradient_op]):
			self.train_op = tf.no_op(name='train')

	#---------------------------------------------------------------------------
	def _hashed_layer(
      self, layer_name, inputs, hiddens, centroid_num=30, flatten=False, relu=True,
      xavier=False, stddev=0.001, hashed = True):
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
		mc = self.mc

		use_pretrained_param = False
		if mc.LOAD_PRETRAINED_MODEL:
			cw = self.caffemodel_weight
			if layer_name in cw:
				use_pretrained_param = True
				kernel_val = cw[layer_name][0]
				bias_val = cw[layer_name][1]

		with tf.variable_scope(layer_name) as scope:
			input_shape = inputs.get_shape().as_list()
			if flatten:#flatten
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
						print ('Shape of the pretrained parameter of {} does not match, '
                		   'use randomly initialized parameter'.format(layer_name))
				else:#no flatten
					dim = input_shape[1]
					if use_pretrained_param:
						try:
							kernel_val = np.transpose(kernel_val, (1,0))
							assert kernel_val.shape == (dim, hiddens), 'kernel shape error at {}'.format(layer_name)
						except:
							use_pretrained_param = False
							print ('Shape of the pretrained parameter of {} does not match, '
							'use randomly initialized parameter'.format(layer_name))

				if use_pretrained_param:
					kmeans = kmeans_hash_cluster(kernel_val, max_iter = 1000)	

					label_init = tf.constant(kmeans.label(), dtype=tf.float32)
					centroid_init = tf.constant(kmeans.centro(), dtype=tf.float32)
					bias_init = tf.constant(bias_val, dtype=tf.float32)
					#real_weights = centroids[weights]
				else:
					label_init = tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32)#dtype=tf.int32
					centroid_init = tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32)
					bias_init = tf.constant_initializer(0.0)
	  
				centroids = _variable_with_weight_decay('centroids', shape=[centroid_num], wd=mc.WEIGHT_DECAY, initializer=centroid_init)
				labels = _variable_on_device('labels', shape=[dim, hiddens], initializer=label_init)
				biases = _variable_on_device('biases', [hiddens], bias_init)
				self.model_params += [centroids, labels, biases]
  
				#----------------------------------
				outputs = tf.nn.bias_add(tf.matmul(inputs, centroids[labels]), biases)
				if relu:
					outputs = tf.nn.relu(outputs, 'relu')
				#!!!!!!finished!!!!!!!!!!!!!!!! kernel is fixed

				# count layer stats
				self.model_size_counter.append((layer_name, (dim+1)*hiddens))

				num_flops = 2 * dim * hiddens + hiddens
				if relu:
					num_flops += 2*hiddens
				self.flop_counter.append((layer_name, num_flops))

				self.activation_counter.append((layer_name, hiddens))

				return outputs

	#---------------------------------------------------------------------------
	def _fc_layer(
      self, layer_name, inputs, hiddens, flatten=False, relu=True,
      xavier=False, stddev=0.001):
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
		mc = self.mc

		use_pretrained_param = False
		if mc.LOAD_PRETRAINED_MODEL:
			cw = self.caffemodel_weight
			if layer_name in cw:
				use_pretrained_param = True
				kernel_val = cw[layer_name][0]
				bias_val = cw[layer_name][1]

		with tf.variable_scope(layer_name) as scope:
			input_shape = inputs.get_shape().as_list()
			if flatten:#flatten
				dim = input_shape[1]*input_shape[2]*input_shape[3]
				inputs = tf.reshape(inputs, [-1, dim])
				if use_pretrained_param:
					try:
						# check the size before layout transform
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
						print ('Shape of the pretrained parameter of {} does not match, '
							'use randomly initialized parameter'.format(layer_name))
			else:#no flatten
				dim = input_shape[1]
				if use_pretrained_param:
					try:
						kernel_val = np.transpose(kernel_val, (1,0))
						assert kernel_val.shape == (dim, hiddens), 'kernel shape error at {}'.format(layer_name)
					except:
						use_pretrained_param = False
						print ('Shape of the pretrained parameter of {} does not match, '
							'use randomly initialized parameter'.format(layer_name))

				if use_pretrained_param:
					kernel_init = tf.constant(kernel_val, dtype=tf.float32)
					bias_init = tf.constant(bias_val, dtype=tf.float32)
				elif xavier:
					kernel_init = tf.contrib.layers.xavier_initializer()
					bias_init = tf.constant_initializer(0.0)
				else:
					kernel_init = tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32)
					bias_init = tf.constant_initializer(0.0)
	  
				weights = _variable_with_weight_decay(
					'weights', shape=[dim, hiddens], wd=mc.WEIGHT_DECAY,
					initializer=kernel_init)
				biases = _variable_on_device('biases', [hiddens], bias_init)
				self.model_params += [weights, biases]
  
				outputs = tf.nn.bias_add(tf.matmul(inputs, weights), biases)
				
				if relu:
					outputs = tf.nn.relu(outputs, 'relu')

				# count layer stats
				self.model_size_counter.append((layer_name, (dim+1)*hiddens))

				num_flops = 2 * dim * hiddens + hiddens
				
				if relu:
					num_flops += 2*hiddens
				self.flop_counter.append((layer_name, num_flops))

				self.activation_counter.append((layer_name, hiddens))

				return outputs

