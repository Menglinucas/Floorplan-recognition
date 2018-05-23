import os
import sys
import tensorflow as tf
import numpy as np
import cv2
import glob
import random
import pdb

# basic parameters
layers = 8
stride = 1
pool = 2
learning_rate = 1.0e-4
epochs = 10000
train_batch_size = 20
img_hsize = 128 
img_wsize = 128
num_channels = 3
num_classes = 5
train_rate = 0.9

# sample data preprocess
def pre_data():
	files = []
	dirs = glob.glob("recg_train_data3/*")
	for dir in dirs:
		for fl in glob.glob(dir+"\*"):
			image = cv2.imread(fl,flags=0)
			if isinstance(image,np.ndarray) == True:
				files.append(fl)
	files_num = len(files)
	random.shuffle(files)
	files_train = files[0:int(train_rate*files_num)]
	files_valid = files[int(train_rate*files_num):files_num]
	return files_train,files_valid

# extract batches
def next_batch(files_sel,batch_size):
	# batch_files = [files_sel[random.randint(0,len(files_sel)-1)] for _ in range(batch_size)]
	indexs = random.sample(range(0,len(files_sel)-1),batch_size)
	images = []
	labels = []
	for index in indexs:
		fl = files_sel[index]
		if num_channels == 1:
			image = cv2.imread(fl,flags=0)
		else:
			image = cv2.imread(fl,flags=1)
		image = cv2.resize(image,(img_wsize,img_hsize),0,0,cv2.INTER_AREA)
		image = image.astype(np.float32)
		image = np.multiply(image,1.0/255.0)
		images.append(image)
		label = np.zeros(num_classes)
		if os.path.dirname(fl) == "recg_train_data3\\no":
			index = 0
		elif os.path.dirname(fl) == "recg_train_data3\\BlackSimple":
			index = 1
		elif os.path.dirname(fl) == "recg_train_data3\\BlackComplicate":
			index = 2
		elif os.path.dirname(fl) == "recg_train_data3\\ColorSimple":
			index = 3
		else:  # os.path.dirname(fl) == "recg_train_data3\\ColorComplicate"
			index = 4
		label[index] = 1
		labels.append(label)
	images = np.array(images)
	labels = np.array(labels)
	return images, labels

# construct networks
# layers
features1 = num_channels
x = tf.placeholder(tf.float32,[None,img_hsize,img_wsize,features1])

features2 = 32

features3 = 64

features4 = 128

features5 = 256

features6 = 512

features_fc1 = 800

features_fc2 = num_classes
y_ = tf.placeholder(tf.float32,[None,num_classes])

# linkers
def weight_init(shape):	#weight initialization function, wight number ~ nodes
	init_value = tf.truncated_normal(shape,stddev=0.05)
	return tf.Variable(init_value)
def bias_init(shape):	#bias initialization function, bias number ~ latter features
	init_value = tf.constant(0.05,shape=shape)
	return tf.Variable(init_value)

filter1 = 5
W_conv1 = weight_init([filter1,filter1,features1,features2])
b_conv1 = bias_init([features2])

filter2 = 5
W_conv2 = weight_init([filter2,filter2,features2,features3])
b_conv2 = bias_init([features3])

filter3 = 5
W_conv3 = weight_init([filter3,filter3,features3,features4])
b_conv3 = bias_init([features4])

filter4 = 5
W_conv4 = weight_init([filter4,filter4,features4,features5])
b_conv4 = bias_init([features5])

filter5 = 5
W_conv5 = weight_init([filter5,filter5,features5,features6])
b_conv5 = bias_init([features6])

W_fc1 = weight_init([img_hsize//pow(pool,(layers-3))*img_wsize//pow(pool,(layers-3))*features6,features_fc1])
b_fc1 = bias_init([features_fc1])

W_fc2 = weight_init([features_fc1, features_fc2])
b_fc2 = bias_init([features_fc2])

# operation
def conv2d(x,W,stride):
	return tf.nn.conv2d(x,W,strides=[1,stride,stride,1],padding='SAME')
def max_pool_2x2(x,pool):
	return tf.nn.max_pool(x,ksize=[1,pool,pool,1],strides=[1,pool*stride,pool*stride,1],padding='SAME')

h_conv1 = tf.nn.relu(conv2d(x,W_conv1,stride)+b_conv1)
h_pool1 = max_pool_2x2(h_conv1,pool)

h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2,stride)+b_conv2)
h_pool2 = max_pool_2x2(h_conv2,pool)

h_conv3 = tf.nn.relu(conv2d(h_pool2,W_conv3,stride)+b_conv3)
h_pool3 = max_pool_2x2(h_conv3,pool)

h_conv4 = tf.nn.relu(conv2d(h_pool3,W_conv4,stride)+b_conv4)
h_pool4 = max_pool_2x2(h_conv4,pool)

h_conv5 = tf.nn.relu(conv2d(h_pool4,W_conv5,stride)+b_conv5)
h_pool5 = max_pool_2x2(h_conv5,pool)

h_pool5_flat = tf.reshape(h_pool5,[-1,img_hsize//pow(pool,(layers-3))*img_wsize//pow(pool,(layers-3))*features6])
h_fc1 = tf.nn.relu(tf.matmul(h_pool5_flat,W_fc1)+b_fc1)
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)

# loss function
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))

# optimizer
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

# evaluation
correct_prediction = tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

# Training
init = tf.global_variables_initializer()
saver = tf.train.Saver()
with tf.Session() as sess:
	if sys.argv[1] == 'train':
		sess.run(init)
		files_train, files_valid = pre_data()
		for i in range(epochs):
			batch = next_batch(files_train,train_batch_size)
			sess.run(train_step,feed_dict={x:batch[0],y_:batch[1],keep_prob:1.0})
			train_accuracy = accuracy.eval(feed_dict = {x:batch[0],y_:batch[1],keep_prob:0.8})
			print('step %d, training accuracy %g'%(i+1,train_accuracy))
			if (i+1)%100 == 0:
				saver.save(sess,'RECG_model3/recg.cpk',global_step=i+1)
		#validation
		batch = next_batch(files_train,len(files_valid))
		valid_accuracy = accuracy.eval(feed_dict = {x:batch[0],y_:batch[1],keep_prob:1.0})
		print ('valid accuracy %g'%valid_accuracy)
	elif sys.argv[1] == 'predict':
		saver.restore(sess, tf.train.latest_checkpoint('RECG_model3/'))
		files = glob.glob("recg_test_data/*")

		images = []
		for fl in files:
			image = cv2.imread(fl,flags=0)
			if isinstance(image,np.ndarray) == False:
				files.remove(files[files.index(fl)])
			else:
				if num_channels == 1:
					image = cv2.imread(fl,flags=0)
				else:
					image = cv2.imread(fl,flags=1)
				image = cv2.resize(image,(img_wsize,img_hsize),0,0,cv2.INTER_AREA)
				image = image.astype(np.float32)
				image = np.multiply(image,1.0/255.0)
				images.append(image)
		images = np.array(images)
		result = sess.run(y_conv, feed_dict={x:images,keep_prob:1.0})
		result = np.argmax(result,axis=1)
		for i in range(len(files)):
			print(files[i],'\t',result[i,])
	else:
		saver.restore(sess, tf.train.latest_checkpoint('RECG_model3/'))
		fl = sys.argv[1]
		image = cv2.imread(fl,flags=0)
		if isinstance(image,np.ndarray) == False:
			print("Maybe not an image!")
		else:
			if num_channels == 1:
				image = cv2.imread(fl,flags=0)
			else:
				image = cv2.imread(fl,flags=1)
			image = cv2.resize(image,(img_wsize,img_hsize),0,0,cv2.INTER_AREA)
			image = image.astype(np.float32)
			image = np.multiply(image,1.0/255.0)
			image = image.reshape((1,image.shape[0],image.shape[1],image.shape[2]))
			result = sess.run(y_conv, feed_dict={x:image,keep_prob:1.0})
			result = np.argmax(result,axis=1)
			result = result[0,]
			print(fl,'\t',result)