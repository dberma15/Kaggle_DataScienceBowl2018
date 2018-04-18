import pandas as pd
from sklearn.model_selection import train_test_split
from keras import backend as K
from keras.losses import binary_crossentropy
from keras.preprocessing.image import ImageDataGenerator
import cv2
import os
import numpy as np
from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation
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
import image_slicer
import re

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
	xtrdata=[]
	for i, id_ in enumerate(train_ids):
		path = train_path + id_
		img = cv2.imread(path + '/images/' + id_ + '.png')
		img_size2=img.shape[0]
		img = cv2.resize(img, (img_size, img_size))
		X_train[i] = img

		mask0 = np.zeros((img_size, img_size, 1), dtype=np.bool)
		boundingboxes=[]
		for mask_file in next(os.walk(path + '/masks/'))[2]:
			mask_ = cv2.imread(path + '/masks/' + mask_file, 0)
			mask_ = cv2.resize(mask_, (img_size, img_size))
			mask_ = mask_[:, :, np.newaxis]
			mask0 = np.maximum(mask0, mask_)

		Y_train0[i] = mask0

	X_test = np.zeros((len(test_ids), img_size, img_size, 3), dtype=np.uint8)
	sizes_test = []
	for i, id_ in enumerate(test_ids):
		path = test_path + id_
		img = cv2.imread(path + '/images/' + id_ + '.png')
		sizes_test.append([img.shape[0], img.shape[1]])
		img = cv2.resize(img, (img_size, img_size))
		X_test[i] = img
	return X_train, Y_train0, X_test, sizes_test


def read_extra(extradata_path, img_size):
	train_ids = next(os.walk(extradata_path))[2] #Generates all file anmes in the tree
	masks=[train_id for train_id in train_ids if re.search('mask',train_id)]
	masks.sort()
	images=[train_id for train_id in train_ids if re.search('original',train_id)]
	images.sort()

	#initialize storage for the data
	X_extra = np.zeros((len(train_ids), img_size, img_size, 3), dtype=np.uint8)
	Y_extra = np.zeros((len(train_ids), img_size, img_size, 1), dtype=np.bool)

	temp = np.zeros((1, img_size, img_size, 3), dtype=np.uint8)
	temp_mask = np.zeros((1, img_size, img_size, 1), dtype=np.bool)
	temp_contour = np.zeros((1, img_size, img_size, 1), dtype=np.bool)
	for i, id_ in enumerate(masks):
		maskid=masks[i]
		imageid=images[i]
		contourid=contours[i]
		img = cv2.imread(extradata_path + '/' + imageid)
		mask = cv2.imread(extradata_path + '/' + maskid,0)
		if img.shape[0]>512:
			tiles=image_slicer.slice(extradata_path+'/'+imageid,16, save=False )
			tiles_mask=image_slicer.slice(extradata_path+'/'+maskid,16, save=False )
			X_extra[i] = cv2.resize(np.array(tiles[0].image),(img_size,img_size))
			Y_extra[i,...,0] = cv2.resize(np.array(tiles_mask[0].image,dtype=np.uint8),(img_size,img_size))
			for counter, (tile, tile_mask) in enumerate(zip(tiles, tiles_mask)):
				if(counter>0):
					temp[0,...] = cv2.resize(np.array(tile.image), (img_size, img_size))
					temp_mask[0,...,0] = cv2.resize(np.array(tile_mask.image,dtype=np.uint8), (img_size, img_size))
					X_extra=np.concatenate([X_extra,temp],axis=0)
					Y_extra=np.concatenate([Y_extra,np.array(temp_mask)])
		else:
			img = cv2.resize(img, (img_size, img_size))
			mask = cv2.resize(mask, (img_size, img_size))
			X_extra[i] = img
			Y_extra[i] = mask[:,:,0:1]
	return X_extra, Y_extra
	
def GatedUnit(f1,f2):
	f1 = Conv2D(3, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (f1)
	f1 = BatchNormalization()(f1)
	f2 = Conv2D(3, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (f2)
	f2 = BatchNormalization()(f2)
	f2 = Conv2DTranspose(3, (2, 2), strides=(2, 2), padding='same') (f2)
	M = multiply([f1,f2])
	return(M)

def GatedRefinementUnit(M,R):
	M = Conv2D(3, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (M)
	M = BatchNormalization()(M)
	R = Conv2D(3, (1, 1), activation='relu', kernel_initializer='he_normal', padding='same') (R)
	RM = concatenate([M,R])
	RM = Conv2D(3, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (RM)
	RM = Conv2DTranspose(3, (2, 2), strides=(2, 2), padding='same') (RM)
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
	
	model = Model(inputs=[inputs], outputs=outputs)
	return model
	
def generator(xtr, xval, ytr, yval, batch_size):
	data_gen_args = dict(horizontal_flip=True,
						 vertical_flip=True,
						 rotation_range=90.,
						 width_shift_range=0.1,
						 height_shift_range=0.1,
						 zoom_range=0.1)
	image_datagen = ImageDataGenerator(**data_gen_args)
	mask_datagen = ImageDataGenerator(**data_gen_args)
	image_datagen.fit(xtr, seed=7)
	mask_datagen.fit(ytr, seed=7)
	image_generator = image_datagen.flow(xtr, batch_size=batch_size, seed=7)
	mask_generator = mask_datagen.flow(ytr, batch_size=batch_size, seed=7)
	train_generator = zip(image_generator, mask_generator)

	#Creates an identity generator
	val_gen_args = dict()
	image_datagen_val = ImageDataGenerator(**val_gen_args)
	mask_datagen_val = ImageDataGenerator(**val_gen_args)
	image_datagen_val.fit(xval, seed=7)
	mask_datagen_val.fit(yval, seed=7)
	image_generator_val = image_datagen_val.flow(xval, batch_size=batch_size, seed=7)
	mask_generator_val = mask_datagen_val.flow(yval, batch_size=batch_size, seed=7)
	val_generator = zip(image_generator_val, mask_generator_val)

	return train_generator, val_generator


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
	datapickle='data256.pickle'
		
	if(os.path.isfile(datapickle)):
		X_extra,Y_extra, X_train,Y_train0, X_test, sizes_test=pickle.load(open(datapickle,'rb'))
	else:
		X_extra,Y_extra=read_extra(extradata_path, img_size)
		X_train, Y_train0, X_test, sizes_test = make_df_bb(train_path, test_path, img_size)
		pickle.dump([X_extra, Y_extra, X_train, Y_train0, X_test, sizes_test],open(datapickle,'wb'))
		
	X_train=np.concatenate([X_train,X_extra])

	Y_train0=np.concatenate([Y_train0,Y_extra])
	
	msk=np.random.random(len(Y_train0))<.9
	xtr=X_train[msk]
	xval=X_train[~msk]
	ytr0=Y_train0[msk]
	yval0=Y_train0[~msk]

	X=np.zeros((len(X_train),1))
	y=np.zeros((len(Y_train0),1))
	
	model = Unet(img_size)
	model.compile(optimizer='adam', loss=bce_dice_loss,  metrics=[mean_iou, 'acc'])

	train_generator0, val_generator0 = generator(xtr, xval, ytr0, yval0, batch_size)

	model.fit_generator(train_generator0, steps_per_epoch=len(xtr)/6, epochs=30,
							validation_data=val_generator0, validation_steps=len(xval)/batch_size)
	
	preds_test = model.predict(X_test, verbose=1)

	
	ts=time.time()
	st=datetime.datetime.fromtimestamp(ts).strftime("%Y%m%d %H%M%S")

	modelname="model_dsbowl2018_"+st+".h5"
	model.save(modelname)

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
	filename='sub-dsbowl2018_'+st+'.csv'
	sub.to_csv(filename, index=False)
	
	
	