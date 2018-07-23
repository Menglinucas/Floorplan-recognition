# This code is used for classifying images. 
# 			classes: floorplan, inDoor, outDoor, else
# usage: 
# 1. in python evironment
# 	import RECG_CNN2_InOutDoor_refine2 as recg
# 	train:       recg.train(continueTrain=False, startEpoch=1, nEpoch=epochs)
# 	test:        recg.test(modelPath)
# 	testFile:    recg.testFile(fl,model)
# 	application: sess,x,y_predict,keep_prob = recg.loadModel(modelPath)
# 	             recg.classify(sess,x,y_predict,keep_prob,inputImage)
# 2. in win powershell
# 	train:       python RECG_CNN2_InOutDoor_refine2.py train
# 	continue train: python RECG_CNN2_InOutDoor_refine2.py train startEpoch totalEpoch
# 	test:        python RECG_CNN2_InOutDoor_refine2.py test
# 	testFile:    python RECG_CNN2_InOutDoor_refine2.py testFile modelPath
# *********************************************************************************
# Note the paths of training data, storing model, test data and the test result !
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
epochs = 2000
train_batch_size = 50
img_hsize = 128 
img_wsize = 128
num_channels = 3
num_classes = 4
train_rate = 0.9
classes = ['floorPlan','inDoor','outDoor','else']
# layer parameters
features1 = num_channels
features2 = 32
features3 = 64
features4 = 100
features5 = 150
features6 = 200
features_fc1 = 300
features_fc2 = num_classes

# sample data preprocess
def pre_data(trainDataPath):
	files = []
	dirs = glob.glob(trainDataPath+"/*")
	for dir in dirs:
		for fl in glob.glob(dir+"/*"):
			image = cv2.imread(fl,flags=0)
			if isinstance(image,np.ndarray) == True:
				files.append(fl)
	files_num = len(files)
	random.shuffle(files)
	files_train = files[0:int(train_rate*files_num)]
	files_valid = files[int(train_rate*files_num):files_num]
	return files_train,files_valid

# extract batches
def next_batch(files_sel,batch_size,trainSubdir):
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
		indexLabel = trainSubdir.index(os.path.dirname(fl))
		label[indexLabel] = 1
		labels.append(label)
	images = np.array(images)
	labels = np.array(labels)
	return images, labels

# convolving operation function
def conv2d(x,W,stride):
	return tf.nn.relu(tf.nn.conv2d(x,W,strides=[1,stride,stride,1],padding='SAME'))
def max_pool_2x2(x,pool):
	return tf.nn.max_pool(x,ksize=[1,pool,pool,1],strides=[1,pool*stride,pool*stride,1],padding='SAME')

# linkers
def weight_init(shape):	#weight initialization function, wight number ~ nodes
	init_value = tf.truncated_normal(shape,stddev=0.05)
	return tf.Variable(init_value,name='weight')
def bias_init(shape):	#bias initialization function, bias number ~ latter features
	init_value = tf.constant(0.05,shape=shape)
	return tf.Variable(init_value,name='bias')

# create a graph
def createGraph():
	# set a graph, g
	g = tf.Graph()
	# construct the graph, g, for our networks
	with g.as_default():
		# with g.device("/gpu:0"):
		# placeholders
		# input data
		with tf.name_scope("inputLayer"):
			x = tf.placeholder(tf.float32,[None,img_hsize,img_wsize,features1],name='xInput')
		# true outdata
		with tf.name_scope("trueData"):
			y_ = tf.placeholder(tf.float32,[None,num_classes],name='yTrue')
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
			# h_conv5_drop = tf.nn.dropout(h_conv5,keep_prob)
			h_pool5 = max_pool_2x2(h_conv5,pool)
		with tf.name_scope("flattenLayer"):
			h_pool5_flat = tf.reshape(h_pool5,[-1,img_hsize//pow(pool,(layers-3))*img_wsize//pow(pool,(layers-3))*features6])
		with tf.name_scope("classfyLayer"):
			h_fc1 = tf.nn.relu(tf.matmul(h_pool5_flat,W_fc1)+b_fc1)
			h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)
		# y-predict
		with tf.name_scope("prediction"):
			y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2,name='yPredict')
		# loss function
		with tf.name_scope("loss"):
			cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
			# cross_entropy = -tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv,labels=y_))
		# optimizer
		with tf.name_scope("optimization"):
			train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy,name='optimizer')
		# evaluation
		with tf.name_scope("estimation"):
			correct_prediction = tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
			accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32),name='accuracy')
		# summaries
		tf.summary.image('image',x,10)
		tf.summary.scalar('accuracy',accuracy)
		tf.summary.histogram('weight',W_conv1)
		tf.summary.histogram('bias',b_conv1)
	return g

def train(continueTrain=False, startEpoch=1, nEpoch=epochs):
	# launch a graph
	g = createGraph()
	# paths of train data and model
	trainDataPath = "E:/FP-python/recg_train_data3_InOutDoor_refine"
	modelPath = "E:/FP-python/RECG_model3_InOutDoor"
	if not os.path.exists(trainDataPath): 
		print("theirs no train data!")
		return 0
	if not os.path.exists(modelPath): os.mkdir(modelPath)
	# subdir of train data, relative to classes
	trainSubdir = [os.path.join(trainDataPath,tsd) for tsd in classes]
	# train data preprocess
	files_train, files_valid = pre_data(trainDataPath)
	# set a session, sess, in g for training
	with tf.Session(graph=g) as sess:
		# merge the summaries
		merged = tf.summary.merge_all()
		# initialize the variables
		init = tf.global_variables_initializer()
		sess.run(init)
		# create a tensorboard writer
		writer = tf.summary.FileWriter("RECG_model3_InOutDoor/logs",sess.graph)
		# create a saver to save variables in training
		saver = tf.train.Saver(tf.global_variables(),max_to_keep=5)
		# continueTrain?
		if continueTrain:
			# load the model
			saver.restore(sess, tf.train.latest_checkpoint(modelPath))
		# placeholder
		x = sess.graph.get_tensor_by_name('inputLayer/xInput:0')
		y_ = sess.graph.get_tensor_by_name('trueData/yTrue:0')
		keep_prob = sess.graph.get_tensor_by_name('dropParameter/keepProb:0')
		accuracy = sess.graph.get_tensor_by_name('estimation/accuracy:0')
		# operation
		y_predict = sess.graph.get_operation_by_name('prediction/yPredict')
		opt = sess.graph.get_operation_by_name('optimization/optimizer')
		# training
		for i in range(startEpoch,startEpoch+nEpoch):
			batch = next_batch(files_train,train_batch_size,trainSubdir)
			sess.run(opt,feed_dict={x:batch[0],y_:batch[1],keep_prob:1.0})
			# or: sess.run('optimizer',feed_dict={x:batch[0],y_:batch[1],keep_prob:1.0})
			train_accuracy = accuracy.eval(feed_dict={x:batch[0],y_:batch[1],keep_prob:1.0})
			# or: train_accuracy = sess.run(accuracy,feed_dict={x:batch[0],y_:batch[1],keep_prob:1.0})
			# or: train_accuracy = sess.run('accuracy:0',feed_dict={x:batch[0],y_:batch[1],keep_prob:1.0})
			print('step %d, training accuracy %g'%(i,train_accuracy))
			# write to logs
			rs = sess.run(merged,feed_dict={x:batch[0],y_:batch[1],keep_prob:1.0})
			writer.add_summary(rs,i)
			# save as ckp
			if i%50 == 0:
				saver.save(sess,os.path.join(modelPath,'recg.ckp'),global_step=i)
			#validation
			batch = next_batch(files_valid,train_batch_size,trainSubdir)
			valid_accuracy = sess.run(accuracy,feed_dict = {x:batch[0],y_:batch[1],keep_prob:1.0})
			print ('valid accuracy %g'%valid_accuracy)
		# save the last step as pb
		constant_graph = graph_util.convert_variables_to_constants(sess,sess.graph_def,['prediction/yPredict'])
		with gfile.FastGFile(os.path.join(modelPath,'recg.pb'), mode='wb') as f:
			f.write(constant_graph.SerializeToString())

def test(modelPath="RECG_model3_InOutDoor"):
	# launch a graph
	g = createGraph()
	####################
	## g = tf.Graph() ##
	####################
	# paths of test data and results
	testPath = "E:/FP-python/recg_test_data"
	resultPath = "E:/FP-python/resultFP"
	if not os.path.exists(testPath): os.mkdir(testPath)
	if not os.path.exists(resultPath): os.mkdir(resultPath)
	# path of each class
	resultDirs = ["/floorPlan","/inDoor","/outDoor","/else"]
	resultDirs = [os.path.join(resultPath,rpi) for rpi in resultDirs]
	# make dir?
	for resultDir in resultDirs:
		if not os.path.exists(resultDir):
			os.mkdir(resultDir)
	# predicting dir
	files = glob.glob(testPath+"/*")
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
		# start time
		tStart = time.time()
		# predicting
		for fl in files:
			image = cv2.imread(fl,flags=0)
			if isinstance(image,np.ndarray) == False:
				# files.remove(files[files.index(fl)])
				print("Maybe not an image!")
				return
			if num_channels == 1:
				image = cv2.imread(fl,flags=0)
			else:
				image = cv2.imread(fl,flags=1)
			image = cv2.resize(image,(img_wsize,img_hsize),0,0,cv2.INTER_AREA)
			image = image.astype(np.float32)
			image = np.multiply(image,1.0/255.0)
			image = np.expand_dims(image,axis=0)
			result = sess.run(y_predict, feed_dict={x:image,keep_prob:1.0})
			result = np.argmax(result,axis=1)[0,]
			# copy file
			shutil.copy(fl,resultDirs[result])
			# print in progress
			print(fl,'\t',classes[result]) 
		# the end time
		tEnd = time.time()
		# print the used time and file number per second
		print(tEnd-tStart)
		print(len(files)/(tEnd-tStart))

def testFile(fl,modelPath):
	# launch a graph
	g = createGraph()
	####################
	## g = tf.Graph() ##
	####################
	# preprocess the image
	image = cv2.imread(fl,flags=0)
	if isinstance(image,np.ndarray) == False:
		print("Maybe not an image!")
		return
	if num_channels == 1:
		image = cv2.imread(fl,flags=0)
	else:
		image = cv2.imread(fl,flags=1)
	image = cv2.resize(image,(img_wsize,img_hsize),0,0,cv2.INTER_AREA)
	image = image.astype(np.float32)
	image = np.multiply(image,1.0/255.0)
	image = image.reshape((1,image.shape[0],image.shape[1],image.shape[2]))
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
		result = sess.run(y_predict, feed_dict={x:image,keep_prob:1.0})
		result = np.argmax(result,axis=1)[0,]
	print(classes[result])
	return classes[result]

# for production, loadModel/classify
def loadModel(modelPath="RECG_model3_InOutDoor"):
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
def classify(sess,x,y_predict,keep_prob,imageByte):
	image = cv2.imdecode(np.fromstring(imageByte,np.uint8),flags=1)
	if isinstance(image,np.ndarray) == False:
		print("Maybe not an image!")
		return
	image = cv2.resize(image,(img_wsize,img_hsize),0,0,cv2.INTER_AREA)
	image = image.astype(np.float32)
	image = np.multiply(image,1.0/255.0)
	image = image.reshape((1,image.shape[0],image.shape[1],image.shape[2]))
	result = sess.run(y_predict, feed_dict={x:image,keep_prob:1.0})
	result = np.argmax(result,axis=1)[0,]
	return classes[result]

def main():
	if len(sys.argv) == 1:
		print("please add the argument: train or test?")
	elif sys.argv[1] == 'train':
		train(continueTrain=False, startEpoch=1, nEpoch=epochs)
	elif sys.argv[1] == 'continuetrain':
		train(continueTrain=True, startEpoch=int(sys.argv[2]), nEpoch=int(sys.argv[3]))
	elif sys.argv[1] == 'test':
		test(modelPath="RECG_model3_InOutDoor")
	elif sys.argv[1] == 'testFile':
		testFile(sys.argv[2],modelPath="RECG_model3_InOutDoor")
	else:
		print("please check your command!")

if __name__ == '__main__':
	main()