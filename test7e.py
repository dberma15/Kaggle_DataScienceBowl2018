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
	Y_train1 = np.zeros((len(train_ids), img_size, img_size, 1), dtype=np.uint8)
	Y_train2 = np.zeros((len(train_ids), img_size, img_size, 1), dtype=np.bool)
	Y_trainoverlap = np.zeros((len(train_ids), img_size, img_size, 1), dtype=np.bool)
	Y_train_contour = np.zeros((len(train_ids), img_size, img_size, 1), dtype=np.bool)
	Y_train_bbox = np.zeros((len(train_ids), img_size, img_size, 3), dtype=np.bool)
	xtrdata=[]
	for i, id_ in enumerate(train_ids):
		path = train_path + id_
		img = cv2.imread(path + '/images/' + id_ + '.png')
		img_size2=img.shape[0]
		img = cv2.resize(img, (img_size, img_size))
		X_train[i] = img
		masktemp = cv2.imread(path + '/masks/' + next(os.walk(path + '/masks/'))[2][0], 0)
		mask2 = np.zeros((masktemp.shape[0], masktemp.shape[1], 1), dtype=np.bool)
		mask1 = np.zeros((masktemp.shape[0], masktemp.shape[1], 1), dtype=np.uint8)
		mask0 = np.zeros((masktemp.shape[0], masktemp.shape[1], 1), dtype=np.bool)
		contour = np.zeros((masktemp.shape[0], masktemp.shape[1], 1), dtype=np.bool)
		for mask_file in next(os.walk(path + '/masks/'))[2]:
			mask_ = cv2.imread(path + '/masks/' + mask_file, 0)
			contour_ = getContour(mask_[:,:])
			contour_ = np.reshape(contour_,(contour_.shape[0],contour_.shape[1],1))
			contour = np.maximum(contour, contour_)	
			mask_ = mask_[:, :, np.newaxis]
			mask0 = np.maximum(mask0, mask_)
			mask_=np.uint8(mask_/255)
			masktemp=skimage.morphology.dilation(mask_)
			masktemp = mask1+masktemp
			mask1 = mask1+mask_
			if((masktemp==2).any()):
				mask1[np.where(masktemp==2)]=2
			contour_ = getContour(mask_[:,:,0])
			contour_ = np.reshape(contour_,(contour_.shape[0],contour_.shape[1],1))
			mask_=mask_-contour_
			mask2 = np.maximum(mask2, mask_)
		Y_train0[i,...,0] = cv2.resize(mask0[...,0],(img_size,img_size))
		Y_train1[i,...,0] = cv2.resize(mask1[...,0],(img_size,img_size))
		Y_train2[i,...,0] = cv2.resize(mask2[...,0],(img_size,img_size))
		Y_train_contour[i,...,0] = cv2.resize(np.uint8(contour[...,0]),(img_size,img_size))
	X_test = np.zeros((len(test_ids), img_size, img_size, 3), dtype=np.uint8)
	sizes_test = []
	for i, id_ in enumerate(test_ids):
		path = test_path + id_
		img = cv2.imread(path + '/images/' + id_ + '.png')
		sizes_test.append([img.shape[0], img.shape[1]])
		img = cv2.resize(img, (img_size, img_size))
		X_test[i] = img
	return X_train, Y_train0, Y_train1, Y_train2, Y_train_contour, X_test, sizes_test#, Y_train_bbox
	
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

def Unet(img_size):
	inputs = Input((img_size, img_size, 3))
	s = Lambda(lambda x: x / 255)(inputs)
	
	c1 = Conv2D(16, (3, 3), kernel_initializer='he_normal', padding='same') (s)
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
	p1 = MaxPooling2D((2, 2)) (c1)
	a1 = AveragePooling2D((2,2))(c1)
	p1 = concatenate([p1,a1])
	
	c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p1)
	c2 = Dropout(0.1) (c2)
	c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c2)
	c2 = Dropout(0.1) (c2)
	c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c2)
	p2 = MaxPooling2D((2, 2)) (c2)
	a2 = AveragePooling2D((2,2))(c2)
	p2 = concatenate([p2,a2])
	
	c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p2)
	c3 = Dropout(0.2) (c3)
	c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c3)
	c3 = Dropout(0.2) (c3)
	c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c3)
	p3 = MaxPooling2D((2, 2)) (c3)
	a3 = AveragePooling2D((2,2))(c3)
	p3 = concatenate([p3,a3])
	
	c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p3)
	c4 = Dropout(0.2) (c4)
	c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c4)
	c4 = Dropout(0.2) (c4)
	c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c4)
	p4 = MaxPooling2D(pool_size=(2, 2)) (c4)
	a4 = AveragePooling2D((2,2))(c4)
	p4 = concatenate([p4,a4])
	
	c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p4)
	c5 = Dropout(0.3) (c5)
	c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c5)
	
	u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c5)
	u6 = concatenate([u6, c4])
	c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u6)
	c6 = Dropout(0.2) (c6)
	c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c6)
	
	q7 = GatedUnit(p3, p4)
	j7 = GatedRefinementUnit(q7,c6)
	u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6)
	u7 = concatenate([u7, c3,j7])
	c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u7)
	c7 = Dropout(0.2) (c7)
	c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c7)
	
	q8 = GatedUnit(p2, p3)
	j8 = GatedRefinementUnit(q8,c7)
	u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7)
	u8 = concatenate([u8, c2,j8])
	c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u8)
	c8 = Dropout(0.1) (c8)
	c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c8)
	
	u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)
	u9 = concatenate([u9, c1], axis=3)
	c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u9)
	c9 = Dropout(0.1) (c9)
	c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c9)
	
	outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
	outputs_contour = Conv2D(1, (1, 1), activation='sigmoid')(c9)
	outputs_box = Conv2D(1, (1, 1), activation='sigmoid')(c9)
	model = Model(inputs=[inputs], outputs=[outputs,outputs_contour,outputs_box])
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
	extradata_path='nuclei/'
	datapickle='data256.pkl'
		
	X_train, Y_train0, Y_train1, Y_train2, Y_train_contour, X_test, sizes_test = make_df_bb(train_path, test_path, img_size)
	
	from keras.utils import np_utils
	Y_train1[np.where(Y_train1==1)]=0
	Y_train1[np.where(Y_train1==2)]=1
		
	msk=np.random.random(len(Y_train0))<.9
	xtr=X_train[msk]
	xval=X_train[~msk]
	ytr0=Y_train0[msk]
	yval0=Y_train0[~msk]
	ytr1=Y_train1[msk]
	yval1=Y_train1[~msk]
	ytr_cntr=Y_train_contour[msk]
	yval_cntr=Y_train_contour[~msk]

	model = Unet(img_size)
	model.compile(optimizer='adam', loss=[bce_dice_loss,bce_dice_loss,bce_dice_loss], metrics=[mean_iou, 'acc'])
	
	ts=time.time()
	st=datetime.datetime.fromtimestamp(ts).strftime("%Y%m%d %H%M%S")

	model.fit(xtr,[ytr0,ytr_cntr,ytr1],batch_size=16, epochs=50, validation_data=(xval,[yval0,yval_cntr,yval1]))

	modelname="model_dsbowl2018_test7e_"+st+".h5"
	model.save(modelname)


	preds_test, preds_contour,preds_double = model.predict(X_test, verbose=1)
	pickle.dump([preds_test, preds_contour,preds_double,X_test],open("7eBoundingBoxPreds.pkl",'wb'))					

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

	filename='sub-dsbowl2018_test7b_twooutputs_'+st+'.csv'
	sub.to_csv(filename, index=False)
	
	