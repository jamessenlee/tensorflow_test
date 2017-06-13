from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

check_point_file = "./check/checkpoint.ckpt"

mnist = input_data.read_data_sets('../../data/MNIST_data',one_hot=True)

input_size  = 28
conv_layer1_size = 5
conv_layer1_channel = 32

conv_layer2_size = 5
conv_layer2_channel = 64

#dcl_layer_input_size is calc by cov and maxpool
dcl_layer1_input_size = 7
dcl_layer1_size = 1024

output_size = 10


epoches = 1000
batch_size  = 50

learning_rate = 1e-4

def weight_variable(shape,name,dtype=tf.float32,initval=None):
	if (initval is None):
		initval = tf.truncated_normal(shape,stddev=0.1)

	w =  tf.Variable(initval,name=name,dtype=dtype)
	tf.summary.histogram(name,w)

	return w
	
def biases_variable(shape,name,dtype=tf.float32,initval=None):
	if (initval is None):
		initval = tf.constant(0.1,shape=shape)

	b =  tf.Variable(initval)
	tf.summary.histogram(name,b)

	return b

	
def conv2d(x,W):
	return tf.nn.conv2d(x,W,[1,1,1,1],padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

class ConvMnistModel(object):
	def __init__(self,inputs,outputs,keep_prob,activation_function=tf.nn.relu):
		self.inputs  = inputs
		self.outputs = outputs
		self.acf = activation_function

		#conv layer1
		with tf.name_scope("conv_layer1"):
			self.conv_w1 = weight_variable([conv_layer1_size,conv_layer1_size,1,conv_layer1_channel],name="conv_w1")
			self.conv_b1 = biases_variable([conv_layer1_channel],name="conv_b1")
			
			self.conv_in1 = tf.reshape(self.inputs,[-1,input_size,input_size,1])

			#h_conv1 is (None,28,28,32)
			self.h_conv1 = self.acf(conv2d(self.conv_in1,self.conv_w1) + self.conv_b1)
	
			#h_pool1 is (None,14,14,32)
			self.h_pool1 = max_pool_2x2(self.h_conv1)

		#conv layer2
		with tf.name_scope("conv_layer2"):
			self.conv_w2 = weight_variable([conv_layer2_size,conv_layer2_size,conv_layer1_channel,conv_layer2_channel],name="conv_w2")
			self.conv_b2 = biases_variable([conv_layer2_channel],name="conv_b2")
		
			#h_conv2 is (None,14,14,64)
			self.h_conv2 = self.acf(conv2d(self.h_pool1,self.conv_w2) + self.conv_b2)
	
			#h_pool2 is (None,7,7,64)
			self.h_pool2 = max_pool_2x2(self.h_conv2)

		#densely connected layer
		with tf.name_scope("dcl_layer"):
			self.dcl_w1 = weight_variable([dcl_layer1_input_size*dcl_layer1_input_size*conv_layer2_channel,dcl_layer1_size],name="dcl_w1")
			self.dcl_b1 = biases_variable([dcl_layer1_size],name="dcl_b1")
			self.pool2_flat = tf.reshape(self.h_pool2,[-1,7*7*conv_layer2_channel])

			#h_fc1 is (None,1024)
			self.h_fc1 = self.acf(tf.matmul(self.pool2_flat,self.dcl_w1)+self.dcl_b1)
			self.h_fc1_drop = tf.nn.dropout(self.h_fc1,keep_prob)
		
		with tf.name_scope("output_layer"):
			self.dcl_w2 = weight_variable([dcl_layer1_size,output_size],name="dcl_w2")
			self.dcl_b2 = biases_variable([output_size],"dcl_b2")

			#no softmax,
			#y is [None,10]
			self.y = tf.matmul(self.h_fc1_drop,self.dcl_w2) + self.dcl_b2
			tf.summary.histogram("output",self.y)


	def cost(self):
		self.cross_entropy = tf.reduce_mean(
				tf.nn.softmax_cross_entropy_with_logits(
					labels = self.outputs,logits = self.y
				)
			)
		return self.cross_entropy

	def train_op(self):
		return tf.train.AdamOptimizer(learning_rate).minimize(self.cross_entropy)



	def accuracy_op(self):
		self.correct_pred = tf.equal(tf.argmax(self.y,1),tf.argmax(self.outputs,1))
		return tf.reduce_mean(tf.cast(self.correct_pred,tf.float32))


X  = tf.placeholder(tf.float32,[None,input_size*input_size])
y_ = tf.placeholder(tf.float32,[None,10])
keep_prob = tf.placeholder(tf.float32)

model = ConvMnistModel(X,y_,keep_prob)

with tf.name_scope("loss"):
	cross_entropy = model.cost()
	tf.summary.scalar('loss',cross_entropy)

with tf.name_scope("train"):
	train_op = model.train_op()


train_accuracy_op = model.accuracy_op()
test_accuracy_op = model.accuracy_op()

gl_init = tf.global_variables_initializer()
lo_init = tf.local_variables_initializer()

saver = tf.train.Saver()

print ("begin training")
with tf.Session() as sess:

	
	
	merged = tf.summary.merge_all()
	sum_writer = tf.summary.FileWriter("./logs/",sess.graph)

	sess.run(gl_init)
	sess.run(lo_init)

	for epoch in range (epoches):
		batch_xs,batch_ys = mnist.train.next_batch(batch_size)

#		train_op = model.train_op
		sess.run(train_op,feed_dict={
				X:batch_xs,y_:batch_ys,keep_prob:0.5
			}
		)

		if ( epoch % 100) == 0:
			train_acc = sess.run(train_accuracy_op,feed_dict={
					X:batch_xs,y_:batch_ys,keep_prob:1.0
				}
			)
			print  ("step %d,training accuracy is %g" %(epoch,train_acc) )
			result = sess.run(merged,feed_dict={
				X:batch_xs,y_:batch_ys,keep_prob:1.0
				})
			sum_writer.add_summary(result,epoch)
			saver.save(sess,check_point_file,global_step=epoch)


	test_accuracy = sess.run(test_accuracy_op,feed_dict={
					X:mnist.test.images,y_:mnist.test.labels,keep_prob:1.0
				}
			)

	print  ("finish training,test accuracy is %g" %(test_acc) )
