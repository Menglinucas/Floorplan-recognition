import os
import sys
import tensorflow as tf
import numpy as np
import cv2
import glob
import random
import pdb

# basic parameters
layers = 5
stride = 1
pool = 2
learning_rate = 1.0e-4
epochs = 2000
train_batch_size = 5
img_hsize = 64
img_wsize = 64
num_channels = 3
num_classes = 2
train_rate = 0.9

# construct networks
# layers
features1 = num_channels
x = tf.placeholder(tf.float32,[None,img_hsize,img_wsize,features1])

features2 = 32

features3 = 64

features_fc1 = 800

features_fc2 = num_classes
y_ = tf.placeholder(tf.int32,[None,img_hsize,img_wsize,1])

features_t1 = features1

features_t2 = features1

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

W_fc1 = weight_init([img_hsize//pow(pool,(layers-3))*img_wsize//pow(pool,(layers-3))*features3,features_fc1])
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

h_pool2_flat = tf.reshape(h_pool2,[-1,img_hsize//pow(pool,(layers-3))*img_wsize//pow(pool,(layers-3))*features3])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1,name='h_fc1')
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2,name='y_conv')

# get weight data
saver = tf.train.Saver()
with tf.Session() as sess1:
	saver.restore(sess1, tf.train.latest_checkpoint('RECG_model/'))
	W1_data = sess1.run(W_conv1)
	b1_data = sess1.run(b_conv1)
	W2_data = sess1.run(W_conv2)
	b2_data = sess1.run(b_conv2)
hconv1_data = tf.nn.relu(conv2d(x,W1_data,stride)+b1_data)
hpool1_data = max_pool_2x2(hconv1_data,pool)
# hconv2_data = tf.nn.relu(conv2d(hpool1_data,W2_data,stride)+b2_data)
# hpool2_data = max_pool_2x2(hconv2_data,pool)

# pool2 to t_pool1 layer
W_t1 = weight_init([5,5,features2,features3])
b_t1 = bias_init([features2])
conv_t1 = tf.nn.conv2d_transpose(h_pool2,W_t1,tf.shape(h_pool1),strides=[1,2,2,1],padding='SAME')+b_t1
fuse_1 = tf.add(conv_t1,hpool1_data)

# pool1 to t_original image layer
W_t2 = weight_init([filter1,filter1,num_classes,features2])
b_t2 = bias_init([num_classes])
x_shape = tf.shape(x)
deconv_shape = tf.stack([x_shape[0],x_shape[1],x_shape[2],num_classes])
conv_t2 = tf.nn.conv2d_transpose(fuse_1,W_t2,deconv_shape,strides=[1,2,2,1],padding='SAME')+b_t2
annotation_pred = tf.expand_dims(tf.argmax(conv_t2,axis=3), dim=3)

loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=conv_t2,
	labels=tf.squeeze(y_,squeeze_dims=[3])))

trainable_var = tf.trainable_variables()
optimizer = tf.train.AdamOptimizer(learning_rate)
grads= optimizer.compute_gradients(loss,trainable_var)
train_op = optimizer.apply_gradients(grads)

# load data
train_images = glob.glob('cutout_train_data/images/*')
train_annotations = glob.glob('cutout_train_data/annotations/*')
valid_images = glob.glob('cutout_valid_data/images/*')
valid_annotations = glob.glob('cutout_valid_data/annotations/*')
# prepare for minibatch
def next_batch(img_names,ann_names,batch_size):
	indexs = [random.randint(0,len(img_names)-1) for _ in range(batch_size)]
	images = []
	annots = []
	for index in indexs:
		image = cv2.imread(img_names[index],flags=1)
		annot = cv2.imread(ann_names[index],flags=0)
		image = cv2.resize(image,(img_wsize,img_hsize),0,0,cv2.INTER_LINEAR)
		image = image.astype(np.float32)
		image = np.multiply(image,1.0/255.0)
		images.append(image)
		annot = cv2.resize(annot,(img_wsize,img_hsize),0,0,cv2.INTER_LINEAR)
		annot = annot.reshape((img_wsize,img_hsize,1))
		annot = annot.astype(np.float32)
		annot = np.multiply(annot,1.0/255.0)
		annots.append(annot)
	images = np.array(images)
	annots = np.array(annots)
	return images, annots

# training
init = tf.global_variables_initializer()
saver = tf.train.Saver()
with tf.Session() as sess:
	sess.run(init)
	for i in range(epochs):
		train_batch = next_batch(train_images,train_annotations,train_batch_size)
		feed_dict = {x:train_batch[0],y_:train_batch[1],keep_prob:1.0}
		sess.run(train_op,feed_dict=feed_dict)
		if (i+1)%200 == 0:
			train_loss = sess.run(loss,feed_dict=feed_dict)
			print('step %d, train loss: %g'%(i+1,train_loss))
			saver.save(sess,'CUTOUT_model/recg.cpk',global_step=i+1)
		# save the predication of the last epoch
		if (i+1) == epochs:
			pred = sess.run(annotation_pred,feed_dict)
			pred = np.squeeze(pred,axis=3)
			for j in range(train_batch_size):
				img_save = pred[j]*255
				cv2.imwrite(os.path.join('CUTOUT_model/img/',str(j)+'.jpg'),img_save)
# 		# validation
# 		if (i+1)%100 == 0:
# 			valid_batch = next_batch(valid_images,valid_annotations,train_batch_size)
# 			valid_loss = sess.run(loss,feed_dict={x:valid_batch[0],y_:valid_batch[1],keep_prob:1.0})
# 			print("%s ---> Validation_loss: %g" % (datetime.datetime.now(), valid_loss))