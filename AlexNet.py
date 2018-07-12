import time
import tensorflow as tf
import kaggle_mnist_input as loader
import os
import csv

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('training_epoch', 30, "training epoch")
tf.app.flags.DEFINE_integer('batch_size', 128, "batch size")
tf.app.flags.DEFINE_integer('training_size', 200000, "batch size")
tf.app.flags.DEFINE_integer('validation_interval', 100, "validation interval")

tf.app.flags.DEFINE_float('dropout_keep_prob', 0.5, "dropout keep prob")
tf.app.flags.DEFINE_float('learning_rate', 0.001, "learning rate")
tf.app.flags.DEFINE_float('rms_decay', 0.9, "rms optimizer decay")
tf.app.flags.DEFINE_float('weight_decay', 0.0005, "l2 regularization weight decay")
tf.app.flags.DEFINE_string('train_path', '/tmp/train.csv', "path to download training data")
tf.app.flags.DEFINE_string('test_path', '/tmp/test.csv', "path to download test data")
tf.app.flags.DEFINE_integer('validation_size', 2000, "validation size in training data")
tf.app.flags.DEFINE_string('save_name', os.getcwd() + '/var.ckpt', "path to save variables")
tf.app.flags.DEFINE_boolean('is_train', True, "True for train, False for test")
tf.app.flags.DEFINE_string('test_result', 'result.csv', "test file path")

image_size=28
image_channel = 1
label_cnt =10

def alexnet():
	#conv layer 1
	conv1_weights = tf.Variable(tf.random_normal([7,7,image_channel,96],dtype=tf.float32,stddev=0.1))
	conv1_biases = tf.Variable(tf.constant(0.0,shape=[96],dtype=tf.float32))
	#卷积
	conv1_relu = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(inputs, conv1_weights,[1,3,3,1],padding='SAME'),conv1_biases))
	#正则化
	conv1_norm = tf.nn.local_response_normalization(conv1_relu, depth_radius=2,alpha=0.0001, beta=0.75, bias=1.0)
	#池化
	conv1_pool = tf.nn.max_pool(conv1_norm,ksize=[1,2,2,1],strides=[1,1,1,1],padding='VALID')


	#conv layer 2
	conv2_weights = tf.Variable(tf.random_normal([5,5,96,256],dtype=tf.float32,stddev=0.1))
	conv2_biases = tf.Variable(tf.constant(0.0,shape=[256],dtype=tf.float32))
	#卷积
	conv2_relu = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1_pool, conv2_weights,[1,1,1,1],padding='SAME'),conv2_biases))
	#正则化
	conv2_norm = tf.nn.local_response_normalization(conv2_relu, depth_radius=2,alpha=0.0001, beta=0.75, bias=1.0)
	#池化
	conv2_pool = tf.nn.max_pool(conv2_norm,ksize=[1,2,2,1],strides=[1,1,1,1],padding='VALID')

	#conv layer 3
	conv3_weights = tf.Variable(tf.random_normal([3,3,256,384],dtype=tf.float32,stddev=0.1))
	conv3_biases = tf.Variable(tf.constant(0.0,shape=[384],dtype=tf.float32))
	#卷积
	conv3_relu = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2_pool, conv3_weights,[1,1,1,1],padding='SAME'),conv3_biases))

	#conv layer 4
	conv4_weights = tf.Variable(tf.random_normal([3,3,384,384],dtype=tf.float32,stddev=0.1))
	conv4_biases = tf.Variable(tf.constant(0.0,shape=[384],dtype=tf.float32))
	#卷积
	conv4_relu = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv3_relu, conv4_weights,[1,1,1,1],padding='SAME'),conv4_biases))

	#conv layer 5
	conv5_weights = tf.Variable(tf.random_normal([3,3,384,256],dtype=tf.float32,stddev=0.1))
	conv5_biases = tf.Variable(tf.constant(0.0,shape=[256],dtype=tf.float32))
	#卷积
	conv5_relu = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv4_relu, conv5_weights,[1,1,1,1],padding='SAME'),conv5_biases))
	#正则化
	conv5_norm = tf.nn.local_response_normalization(conv2_relu, depth_radius=2,alpha=0.0001, beta=0.75, bias=1.0)
	#池化
	conv5_pool = tf.nn.max_pool(conv2_norm,ksize=[1,3,3,1],strides=[1,2,2,1],padding='VALID')

	#fc layer 1
	fc1_weights = tf.Variable(tf.random_normal([4096,4096],dtype=tf.float32,stddev=0.1))
	fc1_biases = tf.Variable(tf.constant([1.0,shape=[4096],dtype=tf.float32]))
	conv5_reshape = tf.reshape(conv5_pool,[-1,fc1_weights.getshape().as_list()[0]])
	fc1_relu = tf.nn.relu(tf.nn.bias_add(tf.matmul(conv5_reshape,fc1_weights),fc1_biases))
	fc1_dropout = tf.nn.dropout(fc1_relu,dropout_keep_prob)

	#fc layer 2
	fc2_weights = tf.Variable(tf.random_normal([4096,4096],dtype=tf.float32,stddev=0.1))
	fc2_biases = tf.Variable(tf.constant([1.0,shape=[4096],dtype=tf.float32]))
	fc2_relu = tf.nn.relu(tf.nn.bias_add(tf.matmul(fc1_dropout,fc2_weights),fc2_biases))
	fc2_dropout = tf.nn.dropout(fc2_relu,dropout_keep_prob)

	#fc layer 3 -out
	fc3_weights = tf.Variable(tf.random_normal([4096,label_cnt],dtype=tf.float32,stddev=0.1))
	fc3_biases = tf.Variable(tf.constant([1.0,shape=[label_cnt],dtype=tf.float32]))
	logits = tf.nn.bias_add(tf.matmul(fc2_dropout,fc3_weights),fc3_biases)
	return logits


inputs = tf.placeholder("float",[None,image_size,image_size,image_channel])
labels = tf.placeholder("float",[None,label_cnt])
dropout_keep_prob = tf.placeholder("float",None)
learning_rate_ph = tf.placeholder("float",None)
global_step = tf.Variable(0,trainable=False)
#构造网络模型
logits = alexnet()

##构造损失函数，为了防止过拟合，加入了正则化项
#loss
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits,labels))
#L2 正则化
regularizers = (tf.nn.l2_loss(conv1_weights)+tf.nn.l2_loss(conv1_biases)+
				tf.nn.l2_loss(conv2_weights)+tf.nn.l2_loss(conv2_biases)+
				tf.nn.l2_loss(conv3_weights)+tf.nn.l2_loss(conv3_biases)+
				tf.nn.l2_loss(conv4_weights)+tf.nn.l2_loss(conv4_biases)+
				tf.nn.l2_loss(conv5_weights)+tf.nn.l2_loss(conv5_biases)+
				tf.nn.l2_loss(fc1_weights)+tf.nn.l2_loss(fc1_biases)+
				tf.nn.l2_loss(fc2_weights)+tf.nn.l2_loss(fc2_biases)+
				tf.nn.l2_loss(fc3_weights)+tf.nn.l2_loss(fc3_biases))
loss = loss + FLAGS.weight_decay*regularizers

##accuracy
predict = tf.argmax(logits,1)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predict,tf.argmax(labels,1)),tf.float32))

#开始训练
train = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(loss,global_step=global_step)

# tf saver
saver = tf.train.Saver()
if os.path.isfile(FLAGS.save_name):
    saver.restore(sess, FLAGS.save_name)

with tf.Session() as sess:
	tf.initialize_all_variables().run()
	for i in range(FLAGS.training_size):
    		xs, ys=mnist.train.next_batch(FLAGS.batch_size)
    		_, loss_value, step = sess.run([train,loss,global_step],feed_dict={x:xs,y_:ys})

    		if i%1000==0:
    			print("after %d training step(s), loss on training batch is %g." % (step,loss_value))
    			saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME),global_step=global_step)



