from keras import backend as K
import time
from multiprocessing.dummy import Pool
K.set_image_data_format('channels_first')
import cv2
import os
import glob
import numpy as np
from numpy import genfromtxt
import tensorflow as tf
from fr_utils import *
from inception_network import *
from face_functions import *
from keras.models import load_model
import sys 

def triplet_loss_function(y_true,y_pred,alpha = 0.3):
	anchor = y_pred[0]
	positive = y_pred[1]
	negative = y_pred[2]
	pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1)
	neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)
	basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
	loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))
	return loss


if __name__=='__main__':

	speak('compiling Model.....', 1)
	model = model(input_shape = (3,96,96))
	model.compile(optimizer = 'adam', loss = triplet_loss_function, metrics = ['accuracy'])
	speak('model compile sucessful', 1)
	speak('loading weights into model, this might take sometime sir!', 1)

	load_weights_from_FaceNet(model) 
	speak('loading weights sequence complete sir!')

	while True:
		speak('model ready to roll sir!')
		decision = input("Initiate face_recognition sequence press Y/N: ")
		#decision = sys.argv[1]
		if decision == ('y' or 'Y'):
			speak('initialising web cam', 2)
			print('initialising webcam')
			image = webcam('temp.jpg')
			database = prepare_database(model)
			speak('Initialising face recognition sequence, sir!', 2.5)
			face = recognise_face("temp.jpg", database, model)
			print(face)

			if face != '0':
				speak('Welcome to the future Sir '+ face, 2)
			os.remove("temp.jpg")
		
		if decision == ('n' or 'N'):
			speak('Face recognition sequence closing....', 2)
			print("Face_recognition sequence closing....")
			break





