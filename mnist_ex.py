import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def weight_variable(shape):
	initial = tf.truncated_normal(shape,stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1,shape=shape)
	return tf.Variable(initial)

def conv2d(x,W):
	return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

def main(argv=None):
	mnist = input_data.read_data_sets("example/MNIST_data/",one_hot=True)

	print "Training data size:",mnist.train.num_examples
	
	print "Validating data size:", mnist.validation.num_examples

	print "Test data size:",mnist.test.num_examples

	print "Example training data:",mnist.train.images[0]

	print "Example training data lebel:",mnist.train.labels[0]

	sess = tf.InteractiveSession()

	x  = tf.placeholder("float",shape=[None,784])
	y_ = tf.placeholder("float",shape=[None,10])

	x_image = tf.reshape(x,[-1,28,28,1])

	W = tf.Variable(tf.zeros([784,10]))
	b = tf.Variable(tf.zeros([10]))

	W_conv1 = weight_variable([5,5,1,32])
	b_conv1 = bias_variable([32])

	print "x_image_shape=%s,W_conv1_shape=%s,b_conv1_shape=%s" %( x_image.get_shape(),W_conv1.get_shape(),b_conv1.get_shape())

	h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1) + b_conv1)
	print "h_conv1 shape=%s" %( h_conv1.get_shape())
	h_pool1 = max_pool_2x2(h_conv1)

	W_conv2 = weight_variable([5,5,32,64])
	b_conv2 = bias_variable([64])

	h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2) + b_conv2)
	h_pool2 = max_pool_2x2(h_conv2)

	
	print h_pool2.get_shape()

	W_fc1 = weight_variable([7*7*64,1024])
	b_fc1 = bias_variable([1024])

	
	h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])

	print h_pool2_flat.get_shape()
	print W_fc1.get_shape()

	h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1) + b_fc1)

	keep_prob = tf.placeholder("float")
	h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

	W_fc2 = weight_variable([1024,10])
	b_fc2 = bias_variable([10])

	y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2) + b_fc2)


	y = tf.nn.softmax(tf.matmul(x,W) + b)

	cross_entropy = -tf.reduce_mean(y_ * tf.log(y))

	correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))

#	train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
	train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)


	sess.run(tf.initialize_all_variables())

	for i in range(90000):
		batch = mnist.train.next_batch(500)
		if i % 5000 == 0:
			train_accuracy = accuracy.eval(feed_dict={x:batch[0],y_:batch[1],keep_prob:1.0})
			print "After step(s) i=%d, training accuracy %g" %(i,train_accuracy)
		train_step.run(feed_dict={x:batch[0],y_:batch[1],keep_prob:0.5}) 


	print accuracy.eval(feed_dict={x:mnist.test.images,y_:mnist.test.labels,keep_prob:1.0})


if __name__ == '__main__':
	tf.app.run()


