import tensorflow as tf
from NTlib.train.CenterFbUpTrainBN2D import CenterFbUpTrainBN2D
import numpy as np
from scipy import misc
from sys import argv

input_placer = tf.placeholder(tf.float32,(None,None,None,None))
gold_placer = tf.placeholder(tf.float32,(None,2))
learn_rate_placer = tf.placeholder(tf.float32)
keep_prob = tf.placeholder(tf.float32)

with tf.Session() as sess:
	wd = float(argv[1])
	wdo = float(argv[2])
	lr = float(argv[3])
	print('Init Train')
	fu = CenterFbUpTrainBN2D(sess, input_placer, gold_placer, keep_prob,
					learn_rate_placer, wd = wd, wdo = wdo, angle_xy = 60)
	#variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope = "fg_"+str(30)+'_'+str(30))
	fu.saver = tf.train.Saver()

	
	print('Train')
	fu.train(epic_num = 3,keep_prob = 1.0, batch_size = 128,
		learning_rate = lr)
	fu.saver.save(fu.sess,'./model/2d/version_'+'0'+'/model_center_'+str(60)+'.ckpt')
	#line_indexes_1 = [[x - (41 - 1)//2,0,0] for x in range(41)]
	"""
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
	image3D = np.zeros((64,64,64))
	noise = np.random.normal(0,0.1,(64,64,64))
	brightness = np.random.random() * (0.8 - 0.2) + 0.2
	image3D[result_1[:,0] + 32, result_1[:,1] + 32, result_1[:,2] + 32] = 255
	image3D = (image3D + noise * 255)*brightness
	misc.imsave('testi0.png',np.max(image3D,axis = 0))
	misc.imsave('testi1.png',np.max(image3D,axis = 1))
	misc.imsave('testi2.png',np.max(image3D,axis = 2))

	#img3D_mat = sio.loadmat('./data/neuron/test.mat')
	#image3D = img3D_mat['img']
	#image3D = image3D[200:300,50:200,325:455] * 255
	#print(np.max(image3D))
	fu.test(image3D)
	"""
