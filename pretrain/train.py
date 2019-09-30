import tensorflow as tf
import os
import sys
import gzip
import numpy

MNIST_PATH = '../data/MNIST'
SAVE_PATH='../pretrain/MNIST'
VALIDATION_SIZE = 5000
class DataSet(object):
  def __init__(self, images, labels, fake_data=False, one_hot=False,
               dtype=tf.float32):
    """Construct a DataSet.
    one_hot arg is used only if fake_data is true.  `dtype` can be either
    `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
    `[0, 1]`.
    """
    dtype = tf.as_dtype(dtype).base_dtype
    if dtype not in (tf.uint8, tf.float32):
      raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                      dtype)
    if fake_data:
      self._num_examples = 10000
      self.one_hot = one_hot
    else:
      assert images.shape[0] == labels.shape[0], (
          'images.shape: %s labels.shape: %s' % (images.shape,
                                                 labels.shape))
      self._num_examples = images.shape[0]
      # Convert shape from [num examples, rows, columns, depth]
      # to [num examples, rows*columns] (assuming depth == 1)
      assert images.shape[3] == 1
      images = images.reshape(images.shape[0],
                              images.shape[1] * images.shape[2])
      if dtype == tf.float32:
        # Convert from [0, 255] -> [0.0, 1.0].
        images = images.astype(numpy.float32)
        images = numpy.multiply(images, 1.0 / 255.0)
    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0
  @property
  def images(self):
    return self._images
  @property
  def labels(self):
    return self._labels
  @property
  def num_examples(self):
    return self._num_examples
  @property
  def epochs_completed(self):
    return self._epochs_completed
  def next_batch(self, batch_size, fake_data=False):
    """Return the next `batch_size` examples from this data set."""
    if fake_data:
      fake_image = [1] * 784
      if self.one_hot:
        fake_label = [1] + [0] * 9
      else:
        fake_label = 0
      return [fake_image for _ in xrange(batch_size)], [
          fake_label for _ in xrange(batch_size)]
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = numpy.arange(self._num_examples)
      numpy.random.shuffle(perm)
      self._images = self._images[perm]
      self._labels = self._labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._images[start:end], self._labels[start:end]
def _read32(bytestream):
  dt = numpy.dtype(numpy.uint32).newbyteorder('>')
  return numpy.frombuffer(bytestream.read(4), dtype=dt)[0]
def extract_images(filename):
  """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    magic = _read32(bytestream)
    if magic != 2051:
      raise ValueError(
          'Invalid magic number %d in MNIST image file: %s' %
          (magic, filename))
    num_images = _read32(bytestream)
    rows = _read32(bytestream)
    cols = _read32(bytestream)
    buf = bytestream.read(rows * cols * num_images)
    data = numpy.frombuffer(buf, dtype=numpy.uint8)
    data = data.reshape(num_images, rows, cols, 1)
    return data
def dense_to_one_hot(labels_dense, num_classes=10):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = numpy.arange(num_labels) * num_classes
  labels_one_hot = numpy.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot
def extract_labels(filename, one_hot=False):
  """Extract the labels into a 1D uint8 numpy array [index]."""
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    magic = _read32(bytestream)
    if magic != 2049:
      raise ValueError(
          'Invalid magic number %d in MNIST label file: %s' %
          (magic, filename))
    num_items = _read32(bytestream)
    buf = bytestream.read(num_items)
    labels = numpy.frombuffer(buf, dtype=numpy.uint8)
    if one_hot:
      return dense_to_one_hot(labels)
    return labels
def MNIST_download(filepath, train_test, image_label):
	path = os.path.join(filepath, train_test + '-'+  image_label + '.gz');
	if os.path.exists(path):
		print('Successfully Imported', path);
		if image_label=='images':
			return extract_images(path);
		else :
			return extract_labels(path, one_hot=True);

train_images = MNIST_download(MNIST_PATH, 'train', 'images');
train_labels = MNIST_download(MNIST_PATH, 'train', 'labels');
test_images = MNIST_download(MNIST_PATH, 'test', 'images');
test_labels = MNIST_download(MNIST_PATH, 'test', 'labels');

validation_images = train_images[:VALIDATION_SIZE];
validation_labels = train_labels[:VALIDATION_SIZE];
train_images = train_images[VALIDATION_SIZE:];
train_labels = train_labels[VALIDATION_SIZE:];

print('train-labels is ', train_labels);

#data_sets = DataSet([],[],fake_data=True, one_hot=True, dtype=tf.float32);
class DataSets(object):
	pass
data_sets = DataSets();
data_sets.test = DataSet(test_images, test_labels, dtype=tf.float32, one_hot=True);
data_sets.train = DataSet(train_images, train_labels, dtype=tf.float32, one_hot=True);
data_sets.validation = DataSet(validation_images, validation_labels, dtype=tf.float32, one_hot=True);

print('datasets train is ', data_sets.train);

'''------------------------------------------------------------------'''

BATCH_SIZE = 100;
TOTAL_BATCH = int();
EPOCH = 10;#1000
EPOCH_DISPLAY = 10;
LEARNING_RATE = 0.001;

X = tf.placeholder(tf.float32, [None,784]);
Y = tf.placeholder(tf.float32, [None,10]);

#initialize problem!!
#W1 = tf.Variable(tf.zeros([784,1000]));
#B1 = tf.Variable(tf.zeros([1000]));
#W2 = tf.Variable(tf.zeros([1000,10]));
#B2 = tf.Variable(tf.zeros([10]));
W1 = tf.Variable(tf.random_normal([784,1000],stddev=0.01));
B1 = tf.Variable(tf.zeros([1000]));
W2 = tf.Variable(tf.random_normal([1000,10],stddev=0.01));
B2 = tf.Variable(tf.zeros([10]));

#model
act1 = tf.nn.relu(tf.matmul(X,W1)+B1);
act2 = tf.matmul(act1,W2)+B2;
hypothesis = act2;
#hypothesis=tf.nn.softmax(act2);

#error
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=hypothesis, labels=Y));
#cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis), reduction_indices=1));
optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost);
#optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cost); 

#initialize
saver = tf.train.Saver(tf.all_variables());
init = tf.global_variables_initializer();
sess = tf.Session();
sess.run(init);

for epoch in range(EPOCH):
	avg_cost = 0.;

	total_batch = int(data_sets.train.num_examples/BATCH_SIZE);

	for i in range(total_batch):
		batch_xs, batch_ys = data_sets.train.next_batch(BATCH_SIZE);
		#print("****size of batch xs is ", batch_xs.shape, "batch ys is ", batch_ys.shape);
		sess.run(optimizer, feed_dict={X:batch_xs, Y:batch_ys});
		avg_cost += sess.run(cost, feed_dict={X:batch_xs, Y:batch_ys});

	if epoch % EPOCH_DISPLAY ==0 :
		print ("Epoch:", "%04d" % (epoch+1), "cost=", "{:.9f}".format(avg_cost));

print("optimization Finished")

is_correct = tf.equal(tf.argmax(hypothesis,1), tf.argmax(Y,1));
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32));
print('Accuracy : ', sess.run(accuracy,feed_dict={X:data_sets.test.images,Y:data_sets.test.labels}));

'''------------------------------------------------------------------'''
'''
#import keras
#from keras.models import Sequential
#from keras import backend as K
from tensorflow.python import keras
from keras.models import Sequential

model = Sequential();
model.add(Dense(1000,activation='relu'));
model.add(Dense(10,activation='softmax'));

model.compile(loss=keras.losses.categorical_crossentropy, optimizer = keras.optimizers.Adadelta(), metrics=['accuracy']);
model.fit(datasets.train.images,datasets.train.labels, batch_size=BATCH_SIE, epochs=EPOCH, verbose=1, validation_data=(datasets.validation.images.datasets.validation.labels));
score = model.evaluate(datasets.test.images,datasets.test.labels, verbose=0);
print('Test loss:', score[0]);
print('test accuracy:',score[1]);

'''
'''------------------------------------------------------------------'''

W1_arr = W1.eval(sess);
B1_arr = B1.eval(sess);
W2_arr = W2.eval(sess);
B2_arr = B2.eval(sess);

#with open(os.path.join(SAVE_PATH, 'MNIST.ckpt')) as f:
saver.save(sess,os.path.join(SAVE_PATH,"./MNIST.ckpt"));

#tup_ob = {'a' : 3, 'b': 5};
#list_ob = ['string', 1023, 103.4];
save_tuple = {'W1' : W1_arr, 'B1' : B1_arr, 'W2' : W2_arr, 'B2' : B2_arr};

#from sklearn.externals import joblib 
import pickle
#joblib.dump([W1,B1,W2,B2],os.path.join(SAVE_PATH, 'MNIST.pkl'));
#pickle.dump(W1,open(os.path.join(SAVE_PATH, 'MNIST.pkl')));
#pickle.dump(B1,open(os.path.join(SAVE_PATH, 'MNIST.pkl')));
#pickle.dump(W2,open(os.path.join(SAVE_PATH, 'MNIST.pkl')));
#pickle.dump(B2,open(os.path.join(SAVE_PATH, 'MNIST.pkl')));

#os.getcwd()
with open(os.path.join(SAVE_PATH, 'MNIST.pkl'), 'wb') as f:
	pickle.dump(save_tuple, f);


'''------------------------------------------------------------------'''


