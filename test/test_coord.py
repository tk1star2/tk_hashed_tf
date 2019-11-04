import tensorflow as tf

import threading
import time

sess = tf.InteractiveSession()

#size = 100 queue
gen_random_normal = tf.random_normal(shape=())

queue = tf.FIFOQueue(capacity=100, dtypes=[tf.float32], shapes=())
enque = queue.enqueue(gen_random_normal)

#10 queue
def add(coord, i):
	try:
		while not coord.should_stop():
			sess.run(enque)
	except Exception as e:
		coord.request_stop(e)

coord = tf.train.Coordinator()

#threads = [threading.Thread(target=add, args=()) for i in range(10)]
threads = [threading.Thread(target=add, args=(coord,i)) for i in range(10)]

coord.join(threads)
for t in threads:
	t.start()


print(sess.run(queue.size()))
time.sleep(1)

print(sess.run(queue.size()))
time.sleep(1)
