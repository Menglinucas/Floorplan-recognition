# This code is used for classifying images. 
#			classes: inDoor, outDoor, floorplan, else
#			(floorplan: blackSimple, blackComplicate, colorSimple, colorComplicate)
# usage:
#	import RECG_CNN2_InOutDoor as recg
#	train:       recg.train()
#	load Model:  recg.loacModel(modelPath)
#	test:        recg.test()
#	application: recg.classify(inputImage)
# Note the paths of training data, storing model, test data and the test result !
# *********************************************************************************
import os, sys, glob, shutil, time
import cv2, random, numpy as np
import tensorflow as tf

# basic parameters
layers = 8
stride = 1
pool = 2
learning_rate = 1.0e-4
epochs = 20000
train_batch_size = 20
img_hsize = 128 
img_wsize = 128
num_channels = 3
num_classes = 7
train_rate = 0.9
# classes
classes = ['inDoor','outDoor','else','blackSimple','blackComplicate',
			'colorSimple','colorComplicate']

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

def loadModel(modelPath):
	# load the model
	saver.restore(sess, tf.train.latest_checkpoint(modelPath))

# constructing networks
# layers
features1 = num_channels
features2 = 32
features3 = 64
features4 = 128
features5 = 256
features6 = 512
features_fc1 = 800
features_fc2 = num_classes

# placeholders
# input data
x = tf.placeholder(tf.float32,[None,img_hsize,img_wsize,features1])
# true outdata
y_ = tf.placeholder(tf.float32,[None,num_classes])
# keep proportion
keep_prob = tf.placeholder(tf.float32)

# linkers
def weight_init(shape):	#weight initialization function, wight number ~ nodes
	init_value = tf.truncated_normal(shape,stddev=0.05)
	return tf.Variable(init_value)
def bias_init(shape):	#bias initialization function, bias number ~ latter features
	init_value = tf.constant(0.05,shape=shape)
	return tf.Variable(init_value)

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

# convolving operation function
def conv2d(x,W,stride):
	return tf.nn.relu(tf.nn.conv2d(x,W,strides=[1,stride,stride,1],padding='SAME'))
def max_pool_2x2(x,pool):
	return tf.nn.max_pool(x,ksize=[1,pool,pool,1],strides=[1,pool*stride,pool*stride,1],padding='SAME')

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
h_pool5_flat = tf.reshape(h_pool5,[-1,img_hsize//pow(pool,(layers-3))*img_wsize//pow(pool,(layers-3))*features6])
h_fc1 = tf.nn.relu(tf.matmul(h_pool5_flat,W_fc1)+b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

# y-predict
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)

# loss function
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
# cross_entropy = -tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv,labels=y_))

# optimizer
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

# evaluation
correct_prediction = tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

# Training
init = tf.global_variables_initializer()
saver = tf.train.Saver()
sess = tf.Session()
# with tf.Graph.device("/gpu:0"):

def train():
	sess.run(init)
	# paths of train data and model
	trainDataPath = "E:/FP-python/recg_train_data3_InOutDoor"
	modelPath = "E:/FP-python/RECG_model3_InOutDoor"
	if not os.path.exists(trainDataPath): 
		print("theirs no train data!")
		return 0
	if not os.path.exists(modelPath): os.mkdir(modelPath)
	# subdir of train data, relative to classes. Note: linux '/', wins '\\'
	trainSubdir = [trainDataPath+'\\'+tsd for tsd in classes]
	# train data preprocess
	files_train, files_valid = pre_data(trainDataPath)
	# training
	for i in range(epochs):
		batch = next_batch(files_train,train_batch_size,trainSubdir)
		sess.run(train_step,feed_dict={x:batch[0],y_:batch[1],keep_prob:0.9})
		# train_accuracy = accuracy.eval(feed_dict = {x:batch[0],y_:batch[1],keep_prob:1.0})
		train_accuracy = sess.run(accuracy,feed_dict = {x:batch[0],y_:batch[1],keep_prob:1.0})
		print('step %d, training accuracy %g'%(i+1,train_accuracy))
		if (i+1)%100 == 0:
			saver.save(sess,modelPath+'/recg.cpk',global_step=i+1)
		#validation
		batch = next_batch(files_valid,train_batch_size,trainSubdir)
		valid_accuracy = accuracy.eval(feed_dict = {x:batch[0],y_:batch[1],keep_prob:1.0})
		print ('valid accuracy %g'%valid_accuracy)

def test():
	# paths of test data and results
	testPath = "E:/FP-python/recg_test_data"
	resultPath = "E:/FP-python/resultFP"
	if not os.path.exists(testPath): os.mkdir(testPath)
	if not os.path.exists(resultPath): os.mkdir(resultPath)
	# path of each class
	resultDirs = ["/inDoor","/outDoor","/else","/floorplan","/floorplan","/floorplan","/floorplan"]
	resultDirs = [resultPath+rpi for rpi in resultDirs]
	# make dir?
	for resultDir in resultDirs:
		if not os.path.exists(resultDir):
			os.mkdir(resultDir)
	# predicting dir
	files = glob.glob(testPath+"/*")
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
		result = sess.run(y_conv, feed_dict={x:image,keep_prob:1.0})
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

# classes
classesApp = ['inDoor','outDoor','else','floorplan','floorplan',
		'floorplan','floorplan']
def testFile(fl):
	image = cv2.imread(fl,flags=0)
	if isinstance(image,np.ndarray) == False:
		print("Maybe not an image!")
		retrun
	if num_channels == 1:
		image = cv2.imread(fl,flags=0)
	else:
		image = cv2.imread(fl,flags=1)
	image = cv2.resize(image,(img_wsize,img_hsize),0,0,cv2.INTER_AREA)
	image = image.astype(np.float32)
	image = np.multiply(image,1.0/255.0)
	image = image.reshape((1,image.shape[0],image.shape[1],image.shape[2]))
	result = sess.run(y_conv, feed_dict={x:image,keep_prob:1.0})
	result = np.argmax(result,axis=1)[0,]
	return classesApp[result]

def classify(imageByte):
	image = cv2.imdecode(np.fromstring(imageByte,np.uint8),flags=1)
	if isinstance(image,np.ndarray) == False:
		print("Maybe not an image!")
		return
	image = cv2.resize(image,(img_wsize,img_hsize),0,0,cv2.INTER_AREA)
	image = image.astype(np.float32)
	image = np.multiply(image,1.0/255.0)
	image = image.reshape((1,image.shape[0],image.shape[1],image.shape[2]))
	result = sess.run(y_conv, feed_dict={x:image,keep_prob:1.0})
	result = np.argmax(result,axis=1)[0,]
	return classesApp[result]

def main():
	if len(sys.argv) == 1:
		print("please add the argument: train or test?")
	elif sys.argv[1] == 'train':
		train()
	elif sys.argv[1] == 'test':
		loadModel(sys.argv[2])
		test()
	else:
		print("please check your command!")

if __name__ == '__main__':
	main()