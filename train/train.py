# Author: Bichen Wu (bichen@berkeley.edu) 08/25/2016

"""Train"""

import os
import sys

import tensorflow as tf
import numpy as np
import threading
import time
from datetime import datetime

#sys.path.append('..')
import pretrain 
from config import base_model_config as config
from hashed import hashed
from six.moves import xrange

MNIST_PATH='../data/MNIST'
VALIDATION_SIZE=5000

'''
import cv2
import os.path

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from utils.util import sparse_to_dense, bgr_to_rgb, bbox_transform
'''
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('dataset', 'MNIST', """Currently only support MNIST? dataset.""")
tf.app.flags.DEFINE_string('image_set', 'train',
                           """ Can be train, trainval, val, or test""")
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Maximum number of batches to run.""")
tf.app.flags.DEFINE_integer('summary_step', 10,
                            """Number of steps to save summary.""")
tf.app.flags.DEFINE_integer('checkpoint_step', 50,
                            """Number of steps to save summary.""")
tf.app.flags.DEFINE_string('train_dir', './MNIST/train',
                            """Directory where to write event logs """
                            """and checkpoint.""")
tf.app.flags.DEFINE_string('net', 'hashedNet',
                           """Neural net architecture. """)
tf.app.flags.DEFINE_string('pretrained_model_path', '../pretrain/MNIST/MNIST.pkl',
                           """Path to the pretrained model.""")
tf.app.flags.DEFINE_string('gpu', '0', """gpu id.""")


#---------------------------------------------------------------------------------
def predataset():
	train_images = pretrain.MNIST_download(MNIST_PATH, 'train', 'images');
	train_labels = pretrain.MNIST_download(MNIST_PATH, 'train', 'labels');
	test_images =  pretrain.MNIST_download(MNIST_PATH, 'test', 'images');
	test_labels =  pretrain.MNIST_download(MNIST_PATH, 'test', 'labels');

	validation_images = train_images[:VALIDATION_SIZE];
	validation_labels = train_labels[:VALIDATION_SIZE];
	train_images = train_images[VALIDATION_SIZE:];
	train_labels = train_labels[VALIDATION_SIZE:];

	#print('train-labels is ', train_labels);

	class DataSets(object):
		pass
	data_sets = DataSets();
	data_sets.test =  pretrain.DataSet(test_images, test_labels, dtype=tf.float32, one_hot=True);
	data_sets.train =  pretrain.DataSet(train_images, train_labels, dtype=tf.float32, one_hot=True);
	data_sets.validation =  pretrain.DataSet(validation_images, validation_labels, dtype=tf.float32, one_hot=True);

	return data_sets
	#print('datasets train is ', data_sets.train);

def train():
	"""Train SqueezeDet model"""
	assert FLAGS.dataset == 'MNIST', 'Currently only support MNIST dataset'
	os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

	with tf.Graph().as_default():
		assert FLAGS.net == 'hashed' or FLAGS.net == 'hashed2', 'Selected neural net architecture not supported: {}'.format(FLAGS.net)
		if FLAGS.net == 'hashed':
			mc = config() # 1.config.py
			mc.IS_TRAINING = True
			mc.PRETRAINED_MODEL_PATH = FLAGS.pretrained_model_path
			model = hashed(mc) # 2.hashed.py
		elif FLAGS.net == 'hashed2':
			mc = config() # 1.config.py
			mc.IS_TRAINING = True
			mc.PRETRAINED_MODEL_PATH = FLAGS.pretrained_model_path
			model = hashed(mc) # 2.hashed.py

		data_sets = predataset()

		#-----------------------------------------------------------------
		def _load_data(load_to_placeholder=True):
			"""Read a batch of image and bounding box annotations.
			Args:
				shuffle: whether or not to shuffle the dataset
			Returns:
				image_per_batch: images. Shape: batch_size x width x height x [b, g, r]
				label_per_batch: labels. Shape: batch_size x object_num
				delta_per_batch: bounding box deltas. Shape: batch_size x object_num x [dx ,dy, dw, dh]
				aidx_per_batch: index of anchors that are responsible for prediction. Shape: batch_size x object_num
				bbox_per_batch: scaled bounding boxes. Shape: batch_size x object_num x [cx, cy, w, h]
			"""
      		# read batch input
			image_per_batch, label_per_batch =	data_sets.train.next_batch(mc.BATCH_SIZE)

			if load_to_placeholder:
				image_input = model.ph_image_input 	#place_holder
				labels = model.ph_labels			#place_holder
			else:
				image_input = model.image_input		#FIFOdequeue a
				labels = model.labels				#FIFOdequeue b

			feed_dict = {
				image_input: image_per_batch, 
				labels: label_per_batch
			}

			return feed_dict, image_per_batch, label_per_batch

		#-----------------------------------------------------------------
		def _enqueue(sess, coord):
			try:
				while not coord.should_stop():
					feed_dict, _, _ = _load_data()
					sess.run(model.enqueue_op, feed_dict=feed_dict)
			except Exception as e:
				coord.request_stop(e)
		#-----------------------------------------------------------------

		#lets go
		sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

		saver = tf.train.Saver(tf.global_variables())
		summary_op = tf.summary.merge_all()

		#restore check_point to session!
		ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
		if ckpt and ckpt.model_checkpoint_path:
			saver.restore(sess, ckpt.model_checkpoint_path)

		summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)

		sess.run(tf.global_variables_initializer())

		coord = tf.train.Coordinator()

		#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		
		if mc.NUM_THREAD > 0: #4
			enq_threads = []
			for _ in range(mc.NUM_THREAD):
				enq_thread = threading.Thread(target=_enqueue, args=[sess, coord])
				enq_thread.start()
				enq_threads.append(enq_thread)
		
		#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

		threads = tf.train.start_queue_runners(coord=coord, sess=sess)
		run_options = tf.RunOptions(timeout_in_ms=60000)

		#***************************************************************************
   	 	# goo: 																	   #
		for step in xrange(FLAGS.max_steps):
			if coord.should_stop():
				sess.run(model.FIFOQueue.close(cancel_pending_enqueues=True))
				coord.request_stop()
				coord.join(threads)
				break

			start_time = time.time()

			#*********************************************************
			if step % FLAGS.summary_step == 0: #10
				feed_dict, image_per_batch, label_per_batch =_load_data(load_to_placeholder=False)
				#tk:op_list = [model.train_op, model.loss, summary_op, model.class_loss]
				op_list = [model.train_op, model.loss, summary_op]

				# sess.run
				#tk:_, loss_value, summary_str, class_loss = sess.run(op_list, feed_dict=feed_dict)
				_, loss_value, summary_str = sess.run(op_list, feed_dict=feed_dict)

				summary_writer.add_summary(summary_str, step)
				summary_writer.flush()

				#tk:print ('class_loss: {}'.format(class_loss))
			else:
				if mc.NUM_THREAD > 0: #4
					# sess.run
					#tk:_, loss_value, class_loss = sess.run([model.train_op, model.loss, model.class_loss], options=run_options)
					_, loss_value = sess.run([model.train_op, model.loss], options=run_options)
				else:
					# sess.run
					feed_dict, _, _, _ = _load_data(load_to_placeholder=False)
					#tk:_, loss_value, class_loss = sess.run([model.train_op, model.loss, model.class_loss], feed_dict=feed_dict)
					_, loss_value = sess.run([model.train_op, model.loss], feed_dict=feed_dict)
			#*********************************************************

			duration = time.time() - start_time

			#tk:assert not np.isnan(loss_value), 'Model diverged. Total loss: {}, class_loss: {}'.format(loss_value, class_loss)
			assert not np.isnan(loss_value), 'Model diverged. Total loss: {}'.format(loss_value)

			if step % 10 == 0:
				num_images_per_step = mc.BATCH_SIZE
				images_per_sec = num_images_per_step / duration
				sec_per_batch = float(duration)
				format_str = ('%s: step %d, loss = %.2f (%.1f images/sec; %.3f ' 'sec/batch)')
				print (format_str % (datetime.now(), step, loss_value,
       	                      images_per_sec, sec_per_batch))
				sys.stdout.flush()

			# Save the model checkpoint periodically. # 50
			if step % FLAGS.checkpoint_step == 0 or (step + 1) == FLAGS.max_steps:
				checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
				saver.save(sess, checkpoint_path, global_step=step)
		
		print("optimization Finished")

		#																		   #
		#***************************************************************************

#---------------------------------------------------------------------------------
def main(argv=None):  # pylint: disable=unused-argument
  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)
  train()
  evaluate()


if __name__ == '__main__':
  tf.app.run()
