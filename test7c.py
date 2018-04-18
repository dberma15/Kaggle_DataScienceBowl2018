import pandas as pd
from sklearn.model_selection import train_test_split
from keras import backend as K
from keras.losses import binary_crossentropy
from keras.preprocessing.image import ImageDataGenerator
import cv2
import os
import numpy as np
from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Reshape, LeakyReLU
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose, ZeroPadding2D, AveragePooling2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate, multiply
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
import pickle
import tensorflow as tf
from keras.optimizers import Adam
from skimage.morphology import label
import time
import scipy.misc
import datetime
import keras_rcnn
import keras_rcnn.models
import keras_rcnn.preprocessing
import skimage
import hashlib

def getContour(mask):
	### generate bound
	mask_pad=np.pad(mask,((1,1),(1,1)),'constant')
	h,w=mask.shape
	contour=np.zeros((h,w),dtype=np.bool)
	for i in range(3):
		for j in range(3):
			if i==j==1:
				continue
			edge=(np.float32(mask)-np.float32(mask_pad[i:i+h,j:j+w]))>0
			contour=np.logical_or(contour,edge)
	return contour
	
def make_df_bb(train_path, test_path, img_size):
	train_ids = next(os.walk(train_path))[1] #Generates all file anmes in the tree
	test_ids = next(os.walk(test_path))[1]
	#initialize storage for the data
	X_train = np.zeros((len(train_ids), img_size, img_size, 3), dtype=np.uint8)
	Y_train0 = np.zeros((len(train_ids), img_size, img_size, 1), dtype=np.bool)
	Y_train_contour = np.zeros((len(train_ids), img_size, img_size, 1), dtype=np.bool)
	Y_train_bbox = np.zeros((len(train_ids), img_size, img_size, 3), dtype=np.bool)
	xtrdata=[]
	for i, id_ in enumerate(train_ids):
		path = train_path + id_
		img = cv2.imread(path + '/images/' + id_ + '.png')
		img_size2=img.shape[0]
		img = cv2.resize(img, (img_size, img_size))
		X_train[i] = img

		mask0 = np.zeros((img_size, img_size, 1), dtype=np.bool)
		contour = np.zeros((img_size, img_size, 1), dtype=np.bool)
		boundingboxes=[]
		boundingboxcenter = np.zeros((img_size, img_size, 3), dtype=np.bool)

		for mask_file in next(os.walk(path + '/masks/'))[2]:
			mask_ = cv2.imread(path + '/masks/' + mask_file, 0)
			mask_ = cv2.resize(mask_, (img_size, img_size))
			boundbox = skimage.measure.regionprops (mask_)
			for box in boundbox:
				boxed = box.bbox
				break
			diffheight=boxed[2]-boxed[0]
			diffwidth=boxed[3]-boxed[1]
			xcenter=round((boxed[2]+boxed[0])/2)
			ycenter=round((boxed[3]+boxed[1])/2)
			boundingboxcenter[xcenter-1,ycenter-1,0]=1
			boundingboxcenter[xcenter-1,ycenter-1,1]=diffheight/img_size
			boundingboxcenter[xcenter-1,ycenter-1,2]=diffwidth/img_size
			contour_ = getContour(mask_[:,:])
			contour_ = np.reshape(contour_,(contour_.shape[0],contour_.shape[1],1))
			contour = np.maximum(contour, contour_)	
			mask_ = mask_[:, :, np.newaxis]
			mask0 = np.maximum(mask0, mask_)

		hashchecksum=hashlib.md5(open(path + '/images/' + id_ + '.png','rb').read()).hexdigest()
		xtrdata.append({'image':{'checksum':hashchecksum,
				  'pathname':path + '/images/' + id_ + '.png',
				  'shape': {'r':img_size2,'c':img_size2,'channels':img.shape[2]}},
				  'objects':boundingboxes})
		Y_train0[i] = mask0
		Y_train_contour[i] = np.reshape(contour,(contour.shape[0],contour.shape[1],1))
		Y_train_bbox[i] = np.reshape(boundingboxcenter,(contour.shape[0],contour.shape[1],3))
	# pickle.dump(xtrdata,open('xtrdata.pkl','wb'))
	X_test = np.zeros((len(test_ids), img_size, img_size, 3), dtype=np.uint8)
	sizes_test = []
	for i, id_ in enumerate(test_ids):
		path = test_path + id_
		img = cv2.imread(path + '/images/' + id_ + '.png')
		sizes_test.append([img.shape[0], img.shape[1]])
		img = cv2.resize(img, (img_size, img_size))
		X_test[i] = img
	return X_train, Y_train0, Y_train_contour, X_test, sizes_test, Y_train_bbox
	
def GatedUnit(f1,f2):
	f1 = Conv2D(2, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (f1)
	f1 = BatchNormalization()(f1)
	f2 = Conv2D(2, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (f2)
	f2 = BatchNormalization()(f2)
	f2 = Conv2DTranspose(2, (2, 2), strides=(2, 2), padding='same') (f2)
	M = multiply([f1,f2])
	return(M)

def GatedRefinementUnit(M,R):
	M = Conv2D(2, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (M)
	M = BatchNormalization()(M)
	R = Conv2D(2, (1, 1), activation='relu', kernel_initializer='he_normal', padding='same') (R)
	RM = concatenate([M,R])
	RM = Conv2D(2, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (RM)
	RM = Conv2DTranspose(2, (2, 2), strides=(2, 2), padding='same') (RM)
	return(RM)
	
def ResNetShorten(resnetModel,layerstoremove,num='a'):
	resnetModel.trainable=False
	for i in range(layerstoremove):
		resnetModel.layers.pop()
		resnetModel.outputs=[resnetModel.layers[-1].output]
		resnetModel.layers[-1].outbound_nodes=[]
	if num!='a':
		for layer in resnetModel.layers:
			layer.name = layer.name + "_"+str(num)
	return resnetModel
	
def Unet(img_size):
	resnetfile='/home/bermads1/Pretrained_Models/ResNet50_weights_tf_dim_ordering_tf_kernels_notop_256x256.h5'
	
	resnetModel8=load_model(resnetfile)
	resnetModel8=ResNetShorten(resnetModel8,1,8)
	for layer in resnetModel8.layers:
		layer.trainable=False
	
	resnetModel16=load_model(resnetfile)
	resnetModel16=ResNetShorten(resnetModel16,1)
	resnetModel16=ResNetShorten(resnetModel16,33,16)
	for layer in resnetModel16.layers:
		layer.trainable=False
	
	resnetModel32=load_model(resnetfile)
	resnetModel32=ResNetShorten(resnetModel32,1)
	resnetModel32=ResNetShorten(resnetModel32,33)
	resnetModel32=ResNetShorten(resnetModel32,61,32)
	for layer in resnetModel32.layers:
		layer.trainable=False
	
	resnetModel64=load_model(resnetfile)
	resnetModel64=ResNetShorten(resnetModel64,1)
	resnetModel64=ResNetShorten(resnetModel64,33)
	resnetModel64=ResNetShorten(resnetModel64,61)
	resnetModel64=ResNetShorten(resnetModel64,42,64)
	for layer in resnetModel64.layers:
		layer.trainable=False
	resnet64pad = ZeroPadding2D(padding=((1,0),(1,0)))(resnetModel64.outputs[0])

	resnetModel128=load_model(resnetfile)
	resnetModel128=ResNetShorten(resnetModel128,1)
	resnetModel128=ResNetShorten(resnetModel128,33)
	resnetModel128=ResNetShorten(resnetModel128,61)
	resnetModel128=ResNetShorten(resnetModel128,42)
	resnetModel128=ResNetShorten(resnetModel128,33,128)
	for layer in resnetModel128.layers:
		layer.trainable=False
	
	resnetModel256=load_model(resnetfile)
	resnetModel256=ResNetShorten(resnetModel256,1)
	resnetModel256=ResNetShorten(resnetModel256,33)
	resnetModel256=ResNetShorten(resnetModel256,61)
	resnetModel256=ResNetShorten(resnetModel256,42)
	resnetModel256=ResNetShorten(resnetModel256,33)
	resnetModel256=ResNetShorten(resnetModel256,3,256)
	for layer in resnetModel256.layers:
		layer.trainable=False
	c1 = Conv2D(16, (3, 3), kernel_initializer='he_normal', padding='same') (resnetModel256.outputs[0])
	c1 = BatchNormalization(axis=3)(c1)
	c1 = Activation('elu')(c1)
	c1 = Dropout(0.1) (c1)
	c1 = Conv2D(16, (3, 3), kernel_initializer='he_normal', padding='same') (c1)
	c1 = BatchNormalization(axis=3)(c1)
	c1 = Activation('elu')(c1)
	c1 = Dropout(0.1) (c1)
	c1 = Conv2D(16, (3, 3), kernel_initializer='he_normal', padding='same') (c1)
	c1 = BatchNormalization(axis=3)(c1)
	c1 = Activation('elu')(c1)

	c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (resnetModel8.outputs[0])
	c5 = Dropout(0.3) (c5)
	c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c5)

	u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c5)
	u6 = concatenate([u6, resnetModel16.outputs[0]])
	c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u6)
	c6 = Dropout(0.2) (c6)
	c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c6)

	u7 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c6)
	u7 = concatenate([u7, resnetModel32.outputs[0]])
	c7 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u7)
	c7 = Dropout(0.2) (c7)
	c7 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c7)

	u8 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c7)
	u8 = concatenate([u8, resnet64pad])
	c8 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u8)
	c8 = Dropout(0.2) (c8)
	c8 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c8)

	u9 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c8)
	u9 = concatenate([u9, resnetModel128.outputs[0]])
	c9 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u9)
	c9 = Dropout(0.1) (c9)
	c9 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c9)

	u10 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c9)
	u10 = concatenate([u10, c1], axis=3)
	c10 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u10)
	c10 = Dropout(0.1) (c10)
	c10 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c10)

	outputs = Conv2D(1, (1, 1), activation='sigmoid')(c10)
	outputs_contour = Conv2D(1, (1, 1), activation='sigmoid')(c10)
	outputs_boxcenter = Conv2D(1, (1, 1), activation='sigmoid')(c10)
	outputs_boxdims = Conv2D(2, (1, 1), activation='sigmoid')(c10)
	inputs=[resnetModel8.inputs[0],resnetModel16.inputs[0],resnetModel32.inputs[0],resnetModel64.inputs[0],resnetModel128.inputs[0],resnetModel256.inputs[0]]
	model = Model(inputs=inputs, outputs=[outputs,outputs_contour,outputs_boxcenter,outputs_boxdims])
	return model
	
def mean_iou(y_true, y_pred):
	prec = []
	for t in np.arange(0.5, 1.0, 0.05):
		y_pred_ = tf.to_int32(y_pred > t)
		score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
		K.get_session().run(tf.local_variables_initializer())
		with tf.control_dependencies([up_opt]):
			score = tf.identity(score)
		prec.append(score)
	return K.mean(K.stack(prec))

def dice_coef(y_true, y_pred):
	smooth = 1.
	y_true_f = K.flatten(y_true)
	y_pred_f = K.flatten(y_pred)
	intersection = K.sum(y_true_f * y_pred_f)
	return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def bce_dice_loss(y_true, y_pred):
	return 0.5 * binary_crossentropy(y_true, y_pred) - dice_coef(y_true, y_pred)
	

def rle_encoding(x):
	dots = np.where(x.T.flatten() == 1)[0]
	run_lengths = []
	prev = -2
	for b in dots:
		if (b>prev+1): run_lengths.extend((b + 1, 0))
		run_lengths[-1] += 1
		prev = b
	return run_lengths

def prob_to_rles(x, cutoff=0.5):
	lab_img = label(x > cutoff)
	for i in range(1, lab_img.max() + 1):
		yield rle_encoding(lab_img == i)
		
		
if __name__ == "__main__":
	img_size = 256
	batch_size = 16
	train_path = 'stage1_train/'
	test_path = 'stage1_test/'
	
	X_train, Y_train0, Y_train_contour, X_test, sizes_test, Y_train_bbox = make_df_bb(train_path, test_path, img_size)
	from sklearn.model_selection import StratifiedKFold
	kfold=StratifiedKFold(n_splits=10, shuffle=True, random_state=7)

	X_train = X_train/255.0
	X_test = X_test/255.0

	msk=np.random.random(len(Y_train0))<.9
	xtr=X_train[msk]
	xval=X_train[~msk]
	ytr_bbox=Y_train_bbox[msk]
	yval_bbox=Y_train_bbox[~msk]
	ytr_cntr0=Y_train_contour[msk]
	yval_cntr0=Y_train_contour[~msk]
	ytr0=Y_train0[msk]
	yval0=Y_train0[~msk]

	X=np.zeros((len(X_train),1))
	y=np.zeros((len(Y_train0),1))
	model = Unet(img_size)

	model.compile(optimizer='adam', loss=[bce_dice_loss, bce_dice_loss, bce_dice_loss,'mean_squared_error'], metrics=[mean_iou, 'acc'])

	pickle.dump([xtr,ytr0,ytr_cntr0,ytr_bbox[...,0:1],ytr_bbox[...,1:3]],open("7cBoundingBoxTraining.pkl",'wb'))					

	traindata=[xtr,xtr,xtr,xtr,xtr,xtr]
	valdata=[xval,xval,xval,xval,xval,xval]
	model.fit(traindata,[ytr0,ytr_cntr0,ytr_bbox[...,0:1],ytr_bbox[...,1:3]], class_weight={0:1,1:1000},batch_size=16, epochs=25, validation_data=(valdata,[yval0,yval_cntr0,yval_bbox[...,0:1],yval_bbox[...,1:3]]))

	ts=time.time()
	st=datetime.datetime.fromtimestamp(ts).strftime("%Y%m%d %H%M%S")

	modelname="model_dsbowl2018_test7c_"+st+".h5"
	model.save(modelname)

	testdata=[X_test,X_test,X_test,X_test,X_test,X_test]
	preds_test, preds_contour, preds_boxescenters, preds_boxesdims = model.predict(testdata, verbose=1)
	preds_boxes=np.concatenate([preds_boxescenters,preds_boxesdims],axis=3)
	pickle.dump([preds_test, preds_contour, preds_boxes,X_test],open("7cBoundingBoxPreds.pkl",'wb'))					

	preds_test_upsampled = []
	preds_test_contour_upsampled = []
	for i in range(len(preds_test)):
		upsampled=cv2.resize(preds_test[i], (sizes_test[i][1], sizes_test[i][0]))
		
		preds_test_upsampled.append(upsampled)
	
	test_ids = next(os.walk(test_path))[1]
	new_test_ids = []
	rles = []
	for n, id_ in enumerate(test_ids):
		rle = list(prob_to_rles(preds_test_upsampled[n]))
		rles.extend(rle)
		new_test_ids.extend([id_] * len(rle))
	sub = pd.DataFrame()
	sub['ImageId'] = new_test_ids
	sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))

	filename='sub-dsbowl2018_test7c_twooutputs_'+st+'.csv'
	sub.to_csv(filename, index=False)
	
