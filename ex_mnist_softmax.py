from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('../data/MNIST_data',one_hot=True)

input_size  = 784
output_size = 10
batch_size  = 100

X  = tf.placeholder(tf.float32,[None,input_size])
y_ = tf.placeholder(tf.float32,[None,10])

W = tf.Variable(tf.random_normal([input_size,output_size],stddev=0.1),name='weight')
b = tf.Variable(tf.zeros([output_size]),name='biases')

y = tf.matmul(X,W) + b

#unstable corss entropy
#cost = tf.reduce_mean(
#		tf.sum(y_*tf.log(tf.nn.softmax(y)),reduction_indices=[1]) )
#	)

cost = tf.reduce_mean(
		tf.nn.softmax_cross_entropy_with_logits(logits=y,labels=y_)
	)

train_step = tf.train.AdamOptimizer(0.5).minimize(cost)

init = tf.global_variables_initializer()



with tf.Session() as sess:
	sess.run(init)

	for i in range(1000):
		batch_xs,batch_ys = mnist.train.next_batch(batch_size)

		sess.run(train_step,feed_dict={
			X:batch_xs,y_:batch_ys
			}
			)	

	correct_pred = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))
	print (sess.run(accuracy,feed_dict={
			X:mnist.test.images,
			y_:mnist.test.labels
		}
		)
		)

