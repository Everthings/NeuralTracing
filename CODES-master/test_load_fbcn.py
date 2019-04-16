import tensorflow as tf
from NTlib.train.CenterFbUpTrainBN2D import CenterFbUpTrainBN2D
import numpy as np
from scipy import misc
from sys import argv
import scipy.io as sio
input_placer = tf.placeholder(tf.float32,(None,None,None,None))
gold_placer = tf.placeholder(tf.float32,(None,2))
learn_rate_placer = tf.placeholder(tf.float32)
keep_prob = tf.placeholder(tf.float32)

with tf.Session() as sess:
	wd = float(argv[2])
	lr = float(argv[3])
	print('Init Test')
	fu = CenterFbUpTrainBN2D(sess, input_placer, gold_placer, keep_prob,
						learn_rate_placer, wd = wd,train_flag = False,
						angle_xy = 30)
	#variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope = "fg_"+str(30)+'_'+str(30))
	fu.saver = tf.train.Saver()

	init_op = tf.initialize_all_variables()
	fu.sess.run(init_op)
	print('Loading')
	fu.saver.restore(fu.sess,'./model/2d/version_'+'0'+'/model_center_'+argv[1]+'.ckpt')
	#line_indexes_1 = [[x - (41 - 1)//2,0,0] for x in range(41)]
	
	line_indexes_1 = [[x - (41-1)//2,0,0] for x in range(41)] + \
					 [[x - (41-1)//2,-1,0] for x in range(41)] + \
					 [[x - (41-1)//2,1,0] for x in range(41)] + \
					 [[x - (41-1)//2,0,1] for x in range(41)] + \
					 [[x - (41-1)//2,0,-1] for x in range(41)]
	angles_xy = np.pi / 6
	angles_xz = np.pi / 6
	tmp_rot = np.array(
			   [[np.cos(angles_xy) * np.cos(angles_xz),
				 -np.sin(angles_xy),
				 np.cos(angles_xy) * np.sin(angles_xz)
				],
				[np.sin(angles_xy) * np.cos(angles_xz),
				 np.cos(angles_xy),
				 np.sin(angles_xy) * np.sin(angles_xz)
				],
				[-np.sin(angles_xz),
				 0,
				 np.cos(angles_xz)
				]
			   ])
	result_1 = np.int_(np.dot(line_indexes_1,tmp_rot))
	image3D = np.zeros((63,63,63))
	noise = np.random.normal(0,0.1,(63,63,63))
	brightness = np.random.random() * (0.8 - 0.2) + 0.2
	image3D[result_1[:,0] + 31, result_1[:,1] + 31, result_1[:,2] + 31] = 255
	image3D = (image3D + noise * 255)*brightness
	image2D = np.max(image3D,axis = 2)
	misc.imsave('testi.png',image2D)
	"""
	img3D_mat = sio.loadmat('./data/neuron/test.mat')
	image3D = img3D_mat['img']
	image3D = image3D[200:300,50:200,325:455] * 255
	misc.imsave('testi0.png',np.max(image3D,axis = 0))
	misc.imsave('testi1.png',np.max(image3D,axis = 1))
	misc.imsave('testi2.png',np.max(image3D,axis = 2))
	
	print(np.max(image3D))
	img3D_mat = sio.loadmat('./data/neuron/test/GMR_57C10_AD_01-1xLwt_attp40_4stop1-m-A02-20111101_1_D3-left_optic_lobe.v3draw.extract_11.v3dpbd_raw.mat')
	image3D = img3D_mat['img']
	#image3D = image3D[70:430,:280,305:485] * 255
	image3D = image3D[160:340,70:390,140:300] * 255
	"""
	print(image2D.shape)
	fu.test(image2D,argv[1])

	
