from NTlib.train.FbUpTrainBN2D import FbUpTrainBN2D
import tensorflow as tf
from scipy import misc
import numpy as np
import pickle
from sys import argv
np.random.seed(17)
tf.set_random_seed(17)
WINDOW_SIZE = 11
input_placer = tf.placeholder(tf.float32,(None,None,None,None))
gold_placer = tf.placeholder(tf.float32,(None,2))
learn_rate_placer = tf.placeholder(tf.float32)
keep_prob = tf.placeholder(tf.float32)
args = [1,30,30,5,128,0.01,0.5]
#np.random.seed()
for i in range(1,len(argv)):
	args[i-1] = argv[i]

version = args[0]#1
angle_xy = int(args[1])#30
epic_num = int(args[2])#5
batch_size = int(args[3])#128
learn_rate = float(args[4])#0.01
kp = float(args[5])#0.5

ft = FbUpTrainBN2D(None,input_placer,gold_placer,
				keep_prob,learn_rate_placer,
				angle_xy = angle_xy,
				wd = 1e-5)

with tf.Session() as sess:
	ft.sess = sess
	ft.saver = tf.train.Saver()
	init_op = tf.initialize_all_variables()
	ft.sess.run(init_op)
	ft.train(epic_num = epic_num,keep_prob = kp,loop_size = 2000,
		batch_size = batch_size,learning_rate = learn_rate)
	ft.saver.save(ft.sess,'./model/2d/version_'+version+'/model_foreground_'+str(angle_xy)+'.ckpt')



