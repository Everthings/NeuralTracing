from NTlib.transfer.TransferCenterBN import TransferCenterBN
import tensorflow as tf
import scipy.io as sio
import numpy as np
from scipy import misc
from sys import argv
import glob
version = argv[1]
input_placer = tf.placeholder(tf.float32,(None,None,None,None))
gold_placer = tf.placeholder(tf.float32,(None,2))
learn_rate_placer = tf.placeholder(tf.float32)
keep_prob = tf.placeholder(tf.float32)
with tf.Session() as sess:
	tft = TransferCenterBN(sess,input_placer,gold_placer,learn_rate_placer,keep_prob,
						  valid_flag = False,valid_num = 2048,
						  wd = 1e-1, predict_flag = False,
						  model_path = './model/2d/version_0')
	tft.saver = tf.train.Saver()
	model_path = './model/2d/version_0'
	print("Load Again")
	tft.saver.restore(sess,model_path + '/transfer_model_'+version+'.ckpt')
	img3D_mat = sio.loadmat('./data/neuron/test/GMR_57C10_AD_01-1xLwt_attp40_4stop1-m-A02-20111101_1_D3-left_optic_lobe.v3draw.extract_11.v3dpbd_raw.mat')
	image3D = img3D_mat['img']
	#image3D = image3D[70:430,:280,305:485] * 255
	image3D = image3D[160:340,70:390,140:300] * 255
	print(np.max(image3D))
	tft.process_stack(image3D,'1')
	misc.imsave('test1_gld0.png',np.max(image3D[10:-10,10:-10,10:-10],axis = 0)*255)
	misc.imsave('test1_gld1.png',np.max(image3D[10:-10,10:-10,10:-10],axis = 1)*255)
	misc.imsave('test1_gld2.png',np.max(image3D[10:-10,10:-10,10:-10],axis = 2)*255)

	img3D_mat = sio.loadmat('./data/neuron/test/GMR_57C10_AD_01-1xLwt_attp40_4stop1-f-A01-20110325_3_A1-right_optic_lobe.v3draw.extract_5.v3dpbd_raw.mat')
	image3D = img3D_mat['img']
	image3D = image3D[160:410,0:150,150:350] * 255
	tft.process_stack(image3D,'3')
	misc.imsave('test3_gld0.png',np.max(image3D[10:-10,10:-10,10:-10],axis = 0)*255)
	misc.imsave('test3_gld1.png',np.max(image3D[10:-10,10:-10,10:-10],axis = 1)*255)
	misc.imsave('test3_gld2.png',np.max(image3D[10:-10,10:-10,10:-10],axis = 2)*255)
	"""
	counter = 4
	for path in glob.glob('./data/neuron/train/*_raw.mat'):
		image_path = "./data/neuron/train"
		image_name = path[len(image_path)+1:-8]
		curr_image = sio.loadmat(path)
		image3D = curr_image['img'] * 255
		tft.predict_large(image3D,str(counter),folder = './test_result/')
		counter+=1
		misc.imsave('./test_result/'+'test'+str(counter)+'_gld0.png',np.max(image3D[10:-10,10:-10,10:-10],axis = 0)*255)
		misc.imsave('./test_result/'+'test'+str(counter)+'_gld1.png',np.max(image3D[10:-10,10:-10,10:-10],axis = 1)*255)
		misc.imsave('./test_result/'+'test'+str(counter)+'_gld2.png',np.max(image3D[10:-10,10:-10,10:-10],axis = 2)*255)
	"""