# Author: Bichen Wu (bichen@berkeley.edu) 08/25/2016

"""Train"""

from ..pretrain import train as pretrain
import hashed
import config
import os

import tensorflow as tf
import threading
import time
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
from datetime import datetime
import os.path
import sys

import numpy as np
from six.moves import xrange

from config import *
from dataset import pascal_voc, kitti
from utils.util import sparse_to_dense, bgr_to_rgb, bbox_transform
from nets import *
'''
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('dataset', 'MNIST',
                           """Currently only support KITTI dataset.""")
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
tf.app.flags.DEFINE_string('pretrained_model_path', '',
                           """Path to the pretrained model.""")
tf.app.flags.DEFINE_string('gpu', '0', """gpu id.""")

#---------------------------------------------------------------------------------
def train():
  """Train SqueezeDet model"""
  assert FLAGS.dataset == 'MNIST', \
      'Currently only support MNIST dataset'

  os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

  with tf.Graph().as_default():

    assert FLAGS.net == 'hashed' or FLAGS.net == 'hashed2' \
        'Selected neural net architecture not supported: {}'.format(FLAGS.net)
    if FLAGS.net == 'hashed':
      mc = config() # 1.config.py
      mc.IS_TRAINING = True
      mc.PRETRAINED_MODEL_PATH = FLAGS.pretrained_model_path
      model = hashed(mc) # 2.hashed.py
    elif FLAGS.net == 'hashed2':
      #mc = kitti_res50_config()
      #mc.IS_TRAINING = True
      #mc.PRETRAINED_MODEL_PATH = FLAGS.pretrained_model_path
      #model = ResNet50ConvDet(mc)
	#----------------------
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
	class DataSets(object):
		pass
	data_sets = DataSets();
	data_sets.test =  pretrain.DataSet(test_images, test_labels, dtype=tf.float32, one_hot=True);
	data_sets.train =  pretrain.DataSet(train_images, train_labels, dtype=tf.float32, one_hot=True);
	data_sets.validation =  pretrain.DataSet(validation_images, validation_labels, dtype=tf.float32, one_hot=True);

	print('datasets train is ', data_sets.train);
	#----------------------
    def _load_data(load_to_placeholder=True):
      # read batch input
      image_per_batch, label_per_batch, box_delta_per_batch, aidx_per_batch, \
          bbox_per_batch = imdb.read_batch()
	  # tk :imdb is about kitti dataset
      label_indices, bbox_indices, box_delta_values, mask_indices, box_values, \
          = [], [], [], [], []
      aidx_set = set()
      num_discarded_labels = 0
      num_labels = 0

      for i in range(len(label_per_batch)): # batch_size
        for j in range(len(label_per_batch[i])): # number of annotations
          num_labels += 1
          if (i, aidx_per_batch[i][j]) not in aidx_set:
            aidx_set.add((i, aidx_per_batch[i][j]))
            label_indices.append(
                [i, aidx_per_batch[i][j], label_per_batch[i][j]])
            mask_indices.append([i, aidx_per_batch[i][j]])
            bbox_indices.extend(
                [[i, aidx_per_batch[i][j], k] for k in range(4)])
            box_delta_values.extend(box_delta_per_batch[i][j])
            box_values.extend(bbox_per_batch[i][j])
          else:
            num_discarded_labels += 1

      if load_to_placeholder:
        image_input = model.ph_image_input
        input_mask = model.ph_input_mask
        box_delta_input = model.ph_box_delta_input
        box_input = model.ph_box_input
        labels = model.ph_labels
      else:
        image_input = model.image_input
        input_mask = model.input_mask
        box_delta_input = model.box_delta_input
        box_input = model.box_input
        labels = model.labels

      feed_dict = {
          image_input: image_per_batch,
          input_mask: np.reshape(
              sparse_to_dense(
                  mask_indices, [mc.BATCH_SIZE, mc.ANCHORS],
                  [1.0]*len(mask_indices)),
              [mc.BATCH_SIZE, mc.ANCHORS, 1]),
          box_delta_input: sparse_to_dense(
              bbox_indices, [mc.BATCH_SIZE, mc.ANCHORS, 4],
              box_delta_values),
          box_input: sparse_to_dense(
              bbox_indices, [mc.BATCH_SIZE, mc.ANCHORS, 4],
              box_values),
          labels: sparse_to_dense(
              label_indices,
              [mc.BATCH_SIZE, mc.ANCHORS, mc.CLASSES],
              [1.0]*len(label_indices)),
      }

      return feed_dict, image_per_batch, label_per_batch, bbox_per_batch

	#-----------------------------------------------------------------
    def _enqueue(sess, coord):
      try:
        while not coord.should_stop():
          feed_dict, _, _, _ = _load_data()
          sess.run(model.enqueue_op, feed_dict=feed_dict)
          if mc.DEBUG_MODE:
            print ("added to the queue")
        if mc.DEBUG_MODE:
          print ("Finished enqueue")
      except Exception, e:
        coord.request_stop(e)
	#-----------------------------------------------------------------
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

    saver = tf.train.Saver(tf.global_variables())
    summary_op = tf.summary.merge_all()

    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)

    summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)

    init = tf.global_variables_initializer()
    sess.run(init)

    coord = tf.train.Coordinator()

    if mc.NUM_THREAD > 0:
      enq_threads = []
      for _ in range(mc.NUM_THREAD):
        enq_thread = threading.Thread(target=_enqueue, args=[sess, coord])
        enq_thread.start()
        enq_threads.append(enq_thread)

    threads = tf.train.start_queue_runners(coord=coord, sess=sess)
    run_options = tf.RunOptions(timeout_in_ms=60000)

	#***************************************************************************
    # try: 
    for step in xrange(FLAGS.max_steps):
      if coord.should_stop():
        sess.run(model.FIFOQueue.close(cancel_pending_enqueues=True))
        coord.request_stop()
        coord.join(threads)
        break

      start_time = time.time()

	  #*********************************************************
      if step % FLAGS.summary_step == 0:
        feed_dict, image_per_batch, label_per_batch, bbox_per_batch = \
            _load_data(load_to_placeholder=False)
        op_list = [
            model.train_op, model.loss, summary_op, model.det_boxes,
            model.det_probs, model.det_class, model.conf_loss,
            model.bbox_loss, model.class_loss
        ]
        _, loss_value, summary_str, det_boxes, det_probs, det_class, \
            conf_loss, bbox_loss, class_loss = sess.run(
                op_list, feed_dict=feed_dict)

        _viz_prediction_result(
            model, image_per_batch, bbox_per_batch, label_per_batch, det_boxes,
            det_class, det_probs)
        image_per_batch = bgr_to_rgb(image_per_batch)
        viz_summary = sess.run(
            model.viz_op, feed_dict={model.image_to_show: image_per_batch})

        summary_writer.add_summary(summary_str, step)
        summary_writer.add_summary(viz_summary, step)
        summary_writer.flush()

        print ('conf_loss: {}, bbox_loss: {}, class_loss: {}'.
            format(conf_loss, bbox_loss, class_loss))
      else:
        if mc.NUM_THREAD > 0:
          _, loss_value, conf_loss, bbox_loss, class_loss = sess.run(
              [model.train_op, model.loss, model.conf_loss, model.bbox_loss,
               model.class_loss], options=run_options)
        else:
          feed_dict, _, _, _ = _load_data(load_to_placeholder=False)
          _, loss_value, conf_loss, bbox_loss, class_loss = sess.run(
              [model.train_op, model.loss, model.conf_loss, model.bbox_loss,
               model.class_loss], feed_dict=feed_dict)
	  #*********************************************************

      duration = time.time() - start_time

      assert not np.isnan(loss_value), \
          'Model diverged. Total loss: {}, conf_loss: {}, bbox_loss: {}, ' \
          'class_loss: {}'.format(loss_value, conf_loss, bbox_loss, class_loss)

      if step % 10 == 0:
        num_images_per_step = mc.BATCH_SIZE
        images_per_sec = num_images_per_step / duration
        sec_per_batch = float(duration)
        format_str = ('%s: step %d, loss = %.2f (%.1f images/sec; %.3f '
                      'sec/batch)')
        print (format_str % (datetime.now(), step, loss_value,
                             images_per_sec, sec_per_batch))
        sys.stdout.flush()

      # Save the model checkpoint periodically.
      if step % FLAGS.checkpoint_step == 0 or (step + 1) == FLAGS.max_steps:
        checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)
    # except Exception, e:
    #   coord.request_stop(e)
    # finally:
    #   coord.request_stop()
    #   coord.join(threads)
	#***************************************************************************

#---------------------------------------------------------------------------------
def main(argv=None):  # pylint: disable=unused-argument
  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)
  train()


if __name__ == '__main__':
  tf.app.run()
