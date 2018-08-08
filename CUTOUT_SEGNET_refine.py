# This code is used for image segmentation,
#				 and calculate the square level of a floor plan. 
# usage: 
# 1. in python evironment
# 	import CUTOUT_SEGNET_refine as cutout
# 	train:       cutout.train(continueTrain=False, startEpoch=1, nEpoch=epochs)
# 	testFile:    cutout.testFile(fl,model)
# 	application: (1) sess,x,y_predict,keep_prob = cutout.loadModel(modelPath)
#	                 cutout.predict(sess,x,y_predict,keep_prob,filename)
#	             (2) sess,x,y_predict,keep_prob = cutout.loadModel(modelPath)
# 	                 cutout.squareLevel(sess,x,y_predict,keep_prob,imageByte)
# 2. in win powershell
# 	train:       python CUTOUT_SEGNET_refine.py train
# 	continue train: python CUTOUT_SEGNET_refine.py train startEpoch totalEpoch
# 	testFile:    python CUTOUT_SEGNET_refine.py testFile modelPath
# *********************************************************************************
# Note the paths of training data, storing model, test data and the test result ! *
# *********************************************************************************
import os, sys, glob, shutil, time
import cv2, random, numpy as np
import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile

# basic parameters
layers = 8
stride = 1
pool = 2
learning_rate = 1.0e-4
epochs = 200000
train_batch_size = 3
img_hsize = 128 
img_wsize = 128
num_channels = 3
num_classes = 2
train_rate = 1
# layers
features1 = num_channels
features2 = 32
features3 = 64
features4 = 128
features5 = 256
features6 = 512
features_fc1 = 800
features_fc2 = num_classes

# prepare for minibatch
def next_batch(img_names,ann_names,batch_size):
	# indexs = [random.randint(0,len(img_names)-1) for _ in range(batch_size)]
	indexs = random.sample(range(0,len(img_names)-1),batch_size)
	images = []
	annots = []
	for index in indexs:
		image = cv2.imread(img_names[index],flags=1)
		annot = cv2.imread(ann_names[index],flags=0)
		image = cv2.resize(image,(img_wsize,img_hsize),0,0,cv2.INTER_AREA)
		image = image.astype(np.float32)
		image = np.multiply(image,1.0/255.0)
		images.append(image)
		annot = cv2.resize(annot,(img_wsize,img_hsize),0,0,cv2.INTER_AREA)
		annot = annot.reshape((img_hsize,img_wsize,1))
		annot = annot.astype(np.float32)
		annot = np.multiply(annot,1.0/255.0)
		annots.append(annot)
	images = np.array(images)
	annots = np.array(annots)
	return images, annots
# post processing
def postProcess(img):
	# close operation
	element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
	img = cv2.morphologyEx(img,cv2.MORPH_CLOSE,element,iterations=3)
	# outline operation
	img = img.astype(np.uint8)
	# img.convertTo(img,CV_8U)
	img, contours, hierarchy = cv2.findContours(img,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)
	maxArea = 0
	for index,contour in enumerate(contours):
		area = cv2.contourArea(contours[index],oriented=False)
		if area > maxArea:
			maxArea = area
			myIndex = index
	img = np.zeros(img.shape,np.uint8)
	img = cv2.drawContours(img,contours,myIndex,255,-1)
	# open operation
	element = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6))
	img = cv2.morphologyEx(img,cv2.MORPH_OPEN,element,iterations=3)
	return img
# square level,  (row, column), corner position
def squareCorner(img):
	# find contour
	_, contours, hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
	imgArea = cv2.contourArea(contours[0],oriented=False)
	# predLength = cv2.arcLength(contours[0],closed=False)
	# box
	imgBoxPoints = cv2.boundingRect(contours[0])
	boxCenter = (imgBoxPoints[1]+imgBoxPoints[3]/2, imgBoxPoints[0]+imgBoxPoints[2]/2)
	imgBoxArea = imgBoxPoints[2] * imgBoxPoints[3]
	# imgArea / imgBoxArea
	imgRatio = imgArea/(imgBoxArea+1e-5)*100
	imgBox = np.zeros(img.shape,np.uint8)
	imgBox[imgBoxPoints[1]:(imgBoxPoints[1]+imgBoxPoints[3]), \
			imgBoxPoints[0]:(imgBoxPoints[0]+imgBoxPoints[2])] = 255
	boxSubtractImg = imgBox - img
	_, contours, hierarchy = cv2.findContours(boxSubtractImg,cv2.RETR_EXTERNAL, \
												cv2.CHAIN_APPROX_NONE)
	# mass center, area
	mc = [None]*len(contours)
	contourAreas = [None]*len(contours)
	for i in range(len(contours)):
		mu = cv2.moments(contours[i])
		mc[i] = (mu['m01']/(mu['m00']+1.e-5), mu['m10']/(mu['m00']+1e-5))
		contourAreas[i] = mu['m00']
	maxAreaIndex = contourAreas.index(max(contourAreas))
	cornerLevel = contourAreas[maxAreaIndex] / (imgBoxArea+1e-5) * 100
	# up? down?
	if mc[maxAreaIndex][0] < (boxCenter[0] - imgBoxPoints[3]/10):
		upDown = 'Up';
	elif mc[maxAreaIndex][0] > (boxCenter[0] + imgBoxPoints[3]/10):
		upDown = 'Down';
	else:
		upDown = 'Center'
	# right? left?
	if mc[maxAreaIndex][1] < (boxCenter[1] - imgBoxPoints[2]/10):
		leftRight = 'Left'
	elif mc[maxAreaIndex][1] > (boxCenter[1] + imgBoxPoints[2]/10):
		leftRight = 'Right'
	else:
		leftRight = 'Center'
	return imgRatio, upDown+leftRight, cornerLevel
# linkers
def weight_init(shape):	#weight initialization function, wight number ~ nodes
	init_value = tf.truncated_normal(shape,stddev=0.05)
	return tf.Variable(init_value)
def bias_init(shape):	#bias initialization function, bias number ~ latter features
	init_value = tf.constant(0.05,shape=shape)
	return tf.Variable(init_value)
# operation
def conv2d(x,W,stride):
	return tf.nn.conv2d(x,W,strides=[1,stride,stride,1],padding='SAME')
def max_pool_2x2(x,pool):
	return tf.nn.max_pool(x,ksize=[1,pool,pool,1],strides=[1,pool*stride,pool*stride,1],padding='SAME')

# set a graph
def createGraph():
	# set a graph, g
	g = tf.Graph()
	# construct the graph, g, for our networks
	with g.as_default():
		# placeholders
		with tf.name_scope("inputLayer"):
			x = tf.placeholder(tf.float32,[None,img_hsize,img_wsize,features1],name="xInput")
		with tf.name_scope("labelLayer"):
			y_ = tf.placeholder(tf.int32,[None,img_hsize,img_wsize,1],name="labelData")
		# keep proportion
		with tf.name_scope("dropParameter"):
			keep_prob = tf.placeholder(tf.float32,name='keepProb')
		# convolutional layers
		with tf.name_scope('convLayers'):
			# filters, w, b
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
			# convolving operation
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
		with tf.name_scope("flattenLayer"):
			h_pool5_flat = tf.reshape(h_pool5,[-1,img_hsize//pow(pool,(layers-3))*img_wsize//pow(pool,(layers-3))*features6])
		with tf.name_scope("classifyLayer"):
			h_fc1 = tf.nn.relu(tf.matmul(h_pool5_flat,W_fc1)+b_fc1)
			h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)
		# y-predict
		with tf.name_scope("convPredict"):
			y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)
		# upsample
		with tf.name_scope('deconvLayers'):
			# pool5 to t_pool3 layer
			W_t3 = weight_init([filter5,filter5,features4,features6])
			b_t3 = bias_init([features4])
			conv_t3 = tf.nn.conv2d_transpose(h_pool5,W_t3,tf.shape(h_pool3),strides=[1,4,4,1],padding='SAME')+b_t3
			fuse_3 = tf.add(conv_t3,h_pool3)
			# pool3 to t_pool1 layer
			W_t1 = weight_init([filter3,filter3,features2,features4])
			b_t1 = bias_init([features2])
			conv_t1 = tf.nn.conv2d_transpose(fuse_3,W_t1,tf.shape(h_pool1),strides=[1,4,4,1],padding='SAME')+b_t1
			fuse_1 = tf.add(conv_t1,h_pool1)
			# pool1 to t_original image layer
			W_t0 = weight_init([filter1,filter1,num_classes,features2])
			b_t0 = bias_init([num_classes])
			x_shape = tf.shape(x)
			deconv_shape = tf.stack([x_shape[0],x_shape[1],x_shape[2],num_classes])
			conv_t0 = tf.nn.conv2d_transpose(fuse_1,W_t0,deconv_shape,strides=[1,2,2,1],padding='SAME')+b_t0
		with tf.name_scope('prediction'):
			annotation_pred = tf.expand_dims(tf.argmax(conv_t0,axis=3), dim=3, name="yPredict")
		# loss function
		with tf.name_scope("loss"):
			loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=conv_t0,
				labels=tf.squeeze(y_,squeeze_dims=[3])),name='lossValue')
		# optimizer
		with tf.name_scope("optimization"):
			trainable_var = tf.trainable_variables()
			optimizer = tf.train.AdamOptimizer(learning_rate)
			grads= optimizer.compute_gradients(loss,trainable_var)
			train_op = optimizer.apply_gradients(grads,name='optimizer')
	return g

def train(continueTrain=False, startEpoch=1, nEpoch=epochs):
	# launch a graph
	g = createGraph()
	# load data
	train_images = glob.glob('cutout_train_data/images/*')
	train_annotations = glob.glob('cutout_train_data/annotations/*')
	valid_images = glob.glob('cutout_valid_data/images/*')
	valid_annotations = glob.glob('cutout_valid_data/annotations/*')
	modelPath = "E:/FP-python/CUTOUT_model"
	# training
	with tf.Session(graph=g) as sess:
		# merge the summaries
		merged = tf.summary.merge_all()
		# initialize the variables
		init = tf.global_variables_initializer()
		sess.run(init)
		# create a tensorboard writer
		writer = tf.summary.FileWriter(os.path.join(modelPath,"logs"),sess.graph)
		# create a saver to save variables in training
		saver = tf.train.Saver(tf.global_variables(),max_to_keep=1)
		# continueTrain?
		if continueTrain:
			# load the model
			saver.restore(sess, tf.train.latest_checkpoint(modelPath))
		# placeholder
		x = sess.graph.get_tensor_by_name('inputLayer/xInput:0')
		y_ = sess.graph.get_tensor_by_name('labelLayer/labelData:0')
		keep_prob = sess.graph.get_tensor_by_name('dropParameter/keepProb:0')
		loss = sess.graph.get_tensor_by_name('loss/lossValue:0')
		# operation
		y_predict = sess.graph.get_operation_by_name('prediction/yPredict')
		opt = sess.graph.get_operation_by_name('optimization/optimizer')
		for i in range(startEpoch,startEpoch+nEpoch):
			train_batch = next_batch(train_images,train_annotations,train_batch_size)
			feed_dict = {x:train_batch[0],y_:train_batch[1],keep_prob:1.0}
			sess.run(opt,feed_dict=feed_dict)
			if i%50 == 0:
				train_loss = sess.run(loss,feed_dict=feed_dict)
				print('step %d, train loss: %g'%(i,train_loss))
				saver.save(sess,os.path.join(modelPath,'cutout.ckp'),global_step=i)
		# save the last step as pb
		constant_graph = graph_util.convert_variables_to_constants(sess,sess.graph_def,['prediction/yPredict'])
		with gfile.FastGFile(os.path.join(modelPath,'cutout.pb'), mode='wb') as f:
			f.write(constant_graph.SerializeToString())

# test one file
def testFile(fl,modelPath="CUTOUT_model"):
	# launch a graph
	g = createGraph()
	####################
	## g = tf.Graph() ##
	####################
	# testing image
	test0 = cv2.imread(fl,flags=1)
	# set a session, sess, in g for training
	with tf.Session(graph=g) as sess:
		# create a saver to save variables in restoring
		saver = tf.train.Saver(tf.global_variables())
		# load the model
		saver.restore(sess, tf.train.latest_checkpoint(modelPath))
		#########################################################################
		## with gfile.FastGFile(os.path.join(modelPath,'recg.pb'), 'rb') as f: ##
		## 	graph_def = tf.GraphDef()                                          ##
		## 	graph_def.ParseFromString(f.read())                                ##
		## 	tf.import_graph_def(graph_def,name='')                             ##
		#########################################################################
		# placeholder
		x = sess.graph.get_tensor_by_name('inputLayer/xInput:0')
		keep_prob = sess.graph.get_tensor_by_name('dropParameter/keepProb:0')
		y_predict = sess.graph.get_tensor_by_name('prediction/yPredict:0')
		# resize to (128,128)
		test = cv2.resize(test0,(img_wsize,img_hsize),0,0,cv2.INTER_AREA)
		test = test.astype(np.float32)
		test = np.multiply(test,1.0/255.0)
		# convert to the shape for tensorflow placeholder
		test = test.reshape((1,test.shape[0],test.shape[1],test.shape[2]))
		# predicting
		pred = sess.run(y_predict,feed_dict={x:test,keep_prob:1.0})
		# recover the shape for displaying image
		test = np.squeeze(test,axis=0) * 255
		pred = np.squeeze(pred,axis=0) * 255
		pred = pred.astype(np.float32)
		#post process
		pred = postProcess(pred)
		# recover predication to the original size
		pred = cv2.resize(pred,(test0.shape[1],test0.shape[0]),0,0,cv2.INTER_CUBIC)
		pred = pred.reshape((pred.shape[0],pred.shape[1],1))
		# to be binary
		_, pred = cv2.threshold(pred,0,255,cv2.THRESH_BINARY)
		# display
		cv2.imshow('pred',pred)
		cv2.waitKey(0)
		# cv2.imwrite("E:/FP-python/test/predictTest.jpg",pred)

# load the model
def loadModel(modelPath="CUTOUT_model"):
	# launch a graph
	g = createGraph()
	####################
	## g = tf.Graph() ##
	####################
	# set an interactive session
	sess = tf.Session(graph=g)
	# get the variables in g
	with g.as_default():
		vars = tf.global_variables()
	# create a saver to save variables in restoring
	saver = tf.train.Saver(vars)
	# load the model
	saver.restore(sess, tf.train.latest_checkpoint(modelPath))
	##########################################################################
	## with g.as_default():                                                 ##
	## 	with gfile.FastGFile(os.path.join(modelPath,'recg.pb'), 'rb') as f: ##
	## 		graph_def = tf.GraphDef()                                       ##
	## 		graph_def.ParseFromString(f.read())                             ##
	##		tf.import_graph_def(graph_def,name='')                          ##
	##########################################################################
	# placeholder
	x = sess.graph.get_tensor_by_name('inputLayer/xInput:0')
	keep_prob = sess.graph.get_tensor_by_name('dropParameter/keepProb:0')
	y_predict = sess.graph.get_tensor_by_name('prediction/yPredict:0')
	return sess,x,y_predict,keep_prob

def predict(sess,x,y_predict,keep_prob,fileName):
	# predicting image
	img0 = cv2.imread(fileName,flags=1)
	if isinstance(img0,np.ndarray) == False:
		print("Maybe not an image!")
		return 0
	# resize to (128,128)
	img = cv2.resize(img0,(img_wsize,img_hsize),0,0,cv2.INTER_AREA)
	img = img.astype(np.float32)
	img = np.multiply(img,1.0/255.0)
	# convert to the shape for tensorflow placeholder
	img = img.reshape((1,img.shape[0],img.shape[1],img.shape[2]))
	# predict
	pred = sess.run(y_predict,feed_dict={x:img,keep_prob:1.0})
	pred = np.squeeze(pred,axis=0) * 255
	pred = pred.astype(np.float32)
	# post process
	pred = postProcess(pred)
	# recover predication to the original size
	pred = cv2.resize(pred,(img0.shape[1],img0.shape[0]),0,0,cv2.INTER_CUBIC)
	pred = pred.reshape((pred.shape[0],pred.shape[1],1))
	# to be binary
	_, pred = cv2.threshold(pred,0,255,cv2.THRESH_BINARY)
	# squareLevel, corner position
	try:
		imgRatio, cornerPosition, cornerLevel = squareCorner(pred)
	except ValueError:
		print("some value error!")
		return
	print("square level: %.4f" % imgRatio)
	print("corner position: %s" % cornerPosition)
	print("corner level: %.1f" % cornerLevel)
	print("shape:", img0.shape)
	# if exist label, calculate the accuracy of square level
	img1 = cv2.imread("test/annotations/annot-"+fileName.split('/')[1],flags=0)
	if isinstance(img1,np.ndarray) == True:
		# open operation
		element = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6))
		img1 = cv2.morphologyEx(img1,cv2.MORPH_CLOSE,element,iterations=3)
		imgRatio1, cornerPosition1, cornerLevel1 = squareCorner(img1)
		accuracy = 100-abs(imgRatio-imgRatio1)/imgRatio1*100
		print("label square level: %.1f" % imgRatio1)
		print("the predicting accuracy is: %.1f" % accuracy)
	# display prediction
	cv2.imshow('pred',pred)
	cv2.waitKey(0)
	return

def squareLevel(sess,x,y_predict,keep_prob,imageByte):
	img0 = cv2.imdecode(np.fromstring(imageByte,np.uint8),flags=1)
	if isinstance(img0,np.ndarray) == False:
		print("Maybe not an image!")
		return
	# resize to (128,128)
	img = cv2.resize(img0,(img_wsize,img_hsize),0,0,cv2.INTER_AREA)
	img = img.astype(np.float32)
	img = np.multiply(img,1.0/255.0)
	# convert to the shape for tensorflow placeholder
	img = img.reshape((1,img.shape[0],img.shape[1],img.shape[2]))
	# predict
	pred = sess.run(y_predict,feed_dict={x:img,keep_prob:1.0})
	pred = np.squeeze(pred,axis=0) * 255
	pred = pred.astype(np.float32)
	# post process
	pred = postProcess(pred)
	# recover predication to the original size
	pred = cv2.resize(pred,(img0.shape[1],img0.shape[0]),0,0,cv2.INTER_CUBIC)
	pred = pred.reshape((pred.shape[0],pred.shape[1],1))
	# to be binary
	_, pred = cv2.threshold(pred,0,255,cv2.THRESH_BINARY)
	# squareLevel
	# find contour
	_, contours, hierarchy = cv2.findContours(pred,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
	imgArea = cv2.contourArea(contours[0],oriented=False)
	# box
	imgBoxPoints = cv2.boundingRect(contours[0])
	boxCenter = (imgBoxPoints[1]+imgBoxPoints[3]/2, imgBoxPoints[0]+imgBoxPoints[2]/2)
	imgBoxArea = imgBoxPoints[2] * imgBoxPoints[3]
	# imgArea / imgBoxArea
	imgRatio = imgArea/(imgBoxArea+1e-5)*100
	return round(imgRatio,4)

def squareLevel2(sess,x,y_predict,keep_prob,img0):
	# img0 = cv2.imdecode(np.fromstring(imageByte,np.uint8),flags=1)
	# if isinstance(img0,np.ndarray) == False:
	# 	print("Maybe not an image!")
	# 	return
	# resize to (128,128)
	img = cv2.resize(img0,(img_wsize,img_hsize),0,0,cv2.INTER_AREA)
	img = img.astype(np.float32)
	img = np.multiply(img,1.0/255.0)
	# convert to the shape for tensorflow placeholder
	img = img.reshape((1,img.shape[0],img.shape[1],img.shape[2]))
	# predict
	pred = sess.run(y_predict,feed_dict={x:img,keep_prob:1.0})
	pred = np.squeeze(pred,axis=0) * 255
	pred = pred.astype(np.float32)
	# post process
	pred = postProcess(pred)
	# recover predication to the original size
	pred = cv2.resize(pred,(img0.shape[1],img0.shape[0]),0,0,cv2.INTER_CUBIC)
	pred = pred.reshape((pred.shape[0],pred.shape[1],1))
	# to be binary
	_, pred = cv2.threshold(pred,0,255,cv2.THRESH_BINARY)
	# squareLevel
	# find contour
	_, contours, hierarchy = cv2.findContours(pred,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
	imgArea = cv2.contourArea(contours[0],oriented=False)
	# box
	imgBoxPoints = cv2.boundingRect(contours[0])
	boxCenter = (imgBoxPoints[1]+imgBoxPoints[3]/2, imgBoxPoints[0]+imgBoxPoints[2]/2)
	imgBoxArea = imgBoxPoints[2] * imgBoxPoints[3]
	# imgArea / imgBoxArea
	imgRatio = imgArea/(imgBoxArea+1e-5)*100
	return round(imgRatio,4),img0.shape

def main():
	if len(sys.argv) == 1:
		print("please add the argument: train or test?")
	elif sys.argv[1] == 'train':
		train(continueTrain=False, startEpoch=1, nEpoch=epochs)
	elif sys.argv[1] == 'continuetrain':
		train(continueTrain=True, startEpoch=int(sys.argv[2]), nEpoch=int(sys.argv[3]))
	elif sys.argv[1] == 'testFile':
		testFile(sys.argv[2],modelPath="CUTOUT_model")
	else:
		print("please check your command!")

if __name__ == '__main__':
	main()