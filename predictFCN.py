# predict module
# import predictFCN
# sess = predictFCN.restoreModel(modelPath="CUTOUT_model")
# predictFCN.predict(sess, fileName="test/test.jpg")

# if session exist, delete it
# for key in globals().keys():
# 	if not key.startswith("__"):
# 		globals().pop(key)
# if 'sess' in vars():
# 	del sess

import tensorflow as tf
import numpy as np
import cv2

# basic parameters
layers = 8
stride = 1
pool = 2
img_hsize = 128 
img_wsize = 128
num_channels = 3
num_classes = 2

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
y_ = tf.placeholder(tf.int32,[None,img_hsize,img_wsize,1])

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
annotation_pred = tf.expand_dims(tf.argmax(conv_t0,axis=3), dim=3)

# post processing
def postProcess(img):
	# open operation
	element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3));
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
	element = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10));
	img = cv2.morphologyEx(img,cv2.MORPH_OPEN,element,iterations=3)
	return img

# restore the model
def restoreModel(modelPath):
	sess = tf.Session()
	# restore the model
	saver = tf.train.Saver()
	saver.restore(sess,tf.train.latest_checkpoint(modelPath))
	return sess

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
	imgRatio = round(imgArea/(imgBoxArea+1e-5)*100)
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
	cornerLevel = round(contourAreas[maxAreaIndex] / (imgBoxArea+1e-5) * 100)
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

def predict(sess,fileName):
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
	
	# saver =  tf.train.import_meta_graph("CUTOUT_model/cutout.cpk-1000000.meta")
	pred = sess.run(annotation_pred,feed_dict={x:img,keep_prob:1.0})
	pred = np.squeeze(pred,axis=0) * 255
	pred = pred.astype(np.float32)
	# cv2.imshow('pred',pred)
	# cv2.waitKey(0)
	# post process
	pred = postProcess(pred)
	# recover predication to the original size
	pred = cv2.resize(pred,(img0.shape[1],img0.shape[0]),0,0,cv2.INTER_CUBIC)
	pred = pred.reshape((pred.shape[0],pred.shape[1],1))
	# to be binary
	_, pred = cv2.threshold(pred,0,255,cv2.THRESH_BINARY)
	cv2.imwrite("test/annot.jpg",pred)
	# squareLevel, corner position
	imgRatio, cornerPosition, cornerLevel = squareCorner(pred)
	print("square level: %d" % imgRatio)
	print("corner position: %s" % cornerPosition)
	print("corner level: %d" % cornerLevel)
	return imgRatio, cornerPosition, cornerLevel

def main():
	print("Note two steps: restoreModel, predict")

if __name__ == '__main__':
	main()

