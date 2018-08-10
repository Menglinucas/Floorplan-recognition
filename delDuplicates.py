'''
Using pHash + resizeHash
Checking duplicates of inDoor, outDoor and floorplan pictures.
Retain the best one, and delete the duplicates.
Usage:
	import delDuplicates as delD
1. for static data base
	two steps one by one.
	step 1: get hashes of every pictures in data base
			cmd: delD.hashForDB(fileName)
			in: file name; 
			out: hash
	step 2: check and delete duplicates
			cmd: delD.delDupl(imageList,hashList0)
			in: image name list, hash list; 
			out: image name list after removing the duplicates
2. for real-time update
	cmd: processDynamic(hashList0,imageByte)
	in: hashList0 -- hash list in the unit of 'community' or 'city', or some else
		imageByte -- real time image
	out: 'add' -- add to data base
		 'drop' -- drop
		 ('rep',n) -- replace the image relative to the n-th hash in hashList0 with the current image.
'''

import numpy as np
import cv2
######################################################################
############################# primary method #########################
######################################################################
# get the hash
# img0--color image, rturn a string(hexadecimal hash + scale + width)
def getHash(img0):
	# size
	height = img0.shape[0]
	width = img0.shape[1]
	# height/width
	hw = round(min(width,height)/max(width,height)*9999)
	# convert to gray image
	img = cv2.cvtColor(img0,cv2.COLOR_BGR2GRAY)
	# resize to 64 x 64
	img = cv2.resize(img,(64,64))
	# dct conversion
	dct = cv2.dct(np.float32(img))
	# extract roi
	dct = dct[0:8,0:8]
	# solve the hash
	hash = []
	# 1, for primary frequency, pHash
	avg = np.mean(dct)
	for i in range(dct.shape[0]):
		for j in range(dct.shape[1]):
			if dct[i,j] > avg:
				hash.append(str(1))
			else:
				hash.append(str(0))
	# 2, for local detail
	# resize to 8 x 8
	img = cv2.resize(img0,(8,8))
	# convert to gray image
	img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	avg = np.mean(img)
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			if img[i,j] > avg:
				hash.append(str(1))
			else:
				hash.append(str(0))
	return hex(int(''.join(hash),2))[2:]+'s'+str(hw).zfill(4)+'w'+str(width)

# duple dictionary
def duplDict(list):
	duplList = []
	nonDuplList = []
	dictList = {}
	for ele in list:
		duplList.append(ele)
		if ele not in nonDuplList:
			nonDuplList.append(ele)
		else:
			ind = len(duplList)-1
			dictList[ele] = dictList.get(ele,[list.index(ele)])
			dictList[ele].append(ind)
	return dictList

######################################################################
############################ for application #########################
######################################################################
	
###### for static.
# 1. calculate the hashes for all pictures in the data base
def hashForDB(fileName):
	with open(fileName,'rb') as f:
		imageByte = f.read()
	# if imageByte can be parsed ?
	image = cv2.imdecode(np.fromstring(imageByte,np.uint8),flags=1)
	if isinstance(image,np.ndarray) == False:
		print("This image can not be parsed by openCV!")
		return
	return getHash(image)
# 2. checking duplicates
def delDupl(imageList,hashList0):
	hashList = [ele[:37] for ele in hashList0]
	nonDuplHashDict = duplDict(hashList)
	toDelList = []
	for hash in nonDuplHashDict.keys():
		duplNums = nonDuplHashDict[hash]
		widthList = [int(hashList0[i][38:]) for i in duplNums]
		maxIndex = widthList.index(max(widthList))
		duplNums.pop(maxIndex)
		toDelList = toDelList + duplNums
	return [imageList[i] for i in toDelList]
	
###### for dynamic. 
# drop, add or replace?
def processDynamic(hashList0,imageByte):
	# if imageByte can be parsed ?
	image = cv2.imdecode(np.fromstring(imageByte,np.uint8),flags=1)
	if isinstance(image,np.ndarray) == False:
		print("Maybe not an image!")
		return
	# if duplicates?
	imgHash = getHash(image)
	hashList = [ele[:37] for ele in hashList0]
	if imgHash[:37] not in hashList:  # add
		res = 'add'
	else:
		index = hashList.index(imgHash[:37])
		if int(imgHash[38:]) <= int(hashList0[index][38:]):  # drop
			res = 'drop'
		else:  # replace
			res = ('rep',index)
	return res

######################################################################
########################### Hamming distance #########################
######################################################################
# the Hamming distance between two hashes
def hammingDistance(hexHash1,hexHash2):
	binHash1 = bin(int(hexHash1.replace('s','').replace('w',''),16))
	binHash2 = bin(int(hexHash2.replace('s','').replace('w',''),16))
	num = 0
	for index in range(len(binHash1)):
		if binHash1[index] != binHash2[index]:
			num += 1 
	return num

######################################################################
################################# test ###############################
######################################################################

# test duplicates of some files
def testGetDupl(path="test/replicate/*"):
	import glob
	import os
	imageNames0 = glob.glob(path)
	hashes = []
	imageNames = []
	for imageName in imageNames0:
		img = cv2.imread(imageName,flags=1)
		if isinstance(img,np.ndarray) == False:
			continue
		hash = getHash(img)[:32]
		hashes.append(hash)
		imageNames.append(imageName)
	dictHashes = duplDict(hashes)
	# with open('checking.txt','w') as f:
		# for key in dictHashes.keys():
			# f.write(str(dictHashes[key])+'\n')
	return imageNames,dictHashes,hashes
	
# test for static, delete duplicates in data base
def testDelDupl(path="test/replicate/*"):
	import time
	tStart = time.time()
	imageList,_,hashList0 = testGetDupl(path)
	tEnd = time.time()
	print("哈希值计算效率：",len(imageList)/(tEnd-tStart))
	tStart = time.time()
	res = delDupl(imageList,hashList0)
	tEnd = time.time()
	print("静态查、去重效率：",len(imageList)/(tEnd-tStart))
	return
	
# test for dynamical process	
def testProcessDynamic(fileName,path="test/replicate/*"):
	_,_,hashList0 = testGetDupl(path)
	with open(fileName,'rb') as f:
		imageByte = f.read()
	res = processDynamic(hashList0,imageByte)
	return res
def testDynamicEff(path="test/replicate/*"):
	import time
	imageNames,_,hashList0 = testGetDupl(path)
	tStart = time.time()
	for fileName in imageNames:
		with open(fileName,'rb') as f:
			imageByte = f.read()
		res = processDynamic(hashList0,imageByte)
	tEnd = time.time()
	print("动态查、去重效率：",len(imageNames)/(tEnd-tStart))
	return
	
# test summary
def testSummary(path="recg_train_data3_InOutDoor_refine/floorplan/*"):
	testDelDupl(path)
	testDynamicEff(path)
	
# x--paths; y--replicate dictionaries
def showDupl(x,y):
	n = 0
	for key in y.keys():
		for i in y[key]:
			cv2.imshow('a'+str(i),cv2.imread(x[i],flags=1))
			if y[key].index(i)==0:
				cv2.moveWindow('a'+str(i),100,100);
			elif y[key].index(i)==1:
				cv2.moveWindow('a'+str(i),700,100);
			elif y[key].index(i)==2:
				cv2.moveWindow('a'+str(i),1300,100);
			elif y[key].index(i)==3:
				cv2.moveWindow('a'+str(i),100,500);
			elif y[key].index(i)==4:
				cv2.moveWindow('a'+str(i),500,500);
			else:
				cv2.moveWindow('a'+str(i),900,500);
		cv2.waitKey(1200);cv2.destroyAllWindows()
		n = n+1
		yield n
def showDupl2(x,iters):
	for i in iters:
		cv2.imshow('a'+str(i),cv2.imread(x[i],flags=1))
		if iters.index(i)==0:
			cv2.moveWindow('a'+str(i),100,100);
		elif iters.index(i)==1:
			cv2.moveWindow('a'+str(i),500,100);
		elif iters.index(i)==2:
			cv2.moveWindow('a'+str(i),900,100);
		elif iters.index(i)==3:
			cv2.moveWindow('a'+str(i),100,500);
		elif iters.index(i)==4:
			cv2.moveWindow('a'+str(i),500,500);
		else:
			cv2.moveWindow('a'+str(i),900,500);
	cv2.waitKey(0)
	return	