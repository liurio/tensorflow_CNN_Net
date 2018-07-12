import os
import tensorflow as tf
import numpy as numpy
from tensorflow.examples.tutorials.mnist import input_data

BATCH_SIZE=100
LEARNING_RATE_BASE=0.8
LEARNING_RATE_DECAY=0.99
REGULARAZTION_RATE=0.0001
TRAIN_STEPS=30000
MOVING_AVERAGE_DECAY=0.99

def train(mnist):
	 x = tf.placeholder(tf.float32, [BATCH_SIZE, LeNet_inference.IMAGE_SIZE,
                                    LeNet_inference.IMAGE_SIZE,
                                    LeNet_inference.NUM_CHANNELS], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, LeNet_inference.OUTPUT_NODE], name='y-input')
    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)##定义一个正则化项，是为了防止过拟合
    global_step = tf.Variable(0,trainable=False)

    y=LeNet_inference.inference(x,train,regularizer)

    #滑动平均模型，使模型在测试数据上更健壮
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    #来判断输出值与真实值之间的误差，采用损失函数
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(y,tf.argmax(y_,1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    loss= cross_entropy_mean+tf.add_n(tf.get_collection('losses'))
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE#基础学习率，学习率在这个基础上递减
    	,global_step,mnist.train.num_examples / BATCH_SIZE,#过完所有的训练数据需要的迭代次数
    	LEARNING_RATE_DECAY#衰减率
    	)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)

    #在训练网络模型时，每过一遍数据通过反向传播来更新参数
    with tf.control_dependencies([train_step,variables_averages_op]):
    	train_op=tf.no_op(name='train')

    saver = tf.train.Saver()
    with tf.Session() as sess:
    	tf.initialize_all_variables().run()
    	for i in range(TRAIN_STEPS):
    		xs, ys=mnist.train.next_batch(BATCH_SIZE)
    		_, loss_value, step = sess.run([train_op,loss,global_step],feed_dict={x:xs,y_:ys})

    		if i%1000==0:
    			print("after %d training step(s), loss on training batch is %g." % (step,loss_value))
    			saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME),global_step=global_step)

def main(argv=None):
	mnist = input_data.read_data_sets("D:/note/tensorflow/tensorboard/tensorflow/mnist/input_data',one_hot=True")
	train(mnist)

if __name__=='__main__':
	tf.app.run()

