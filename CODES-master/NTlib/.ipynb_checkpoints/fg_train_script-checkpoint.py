from NTlib.foreground_train import *
import tensorflow as tf
from scipy import misc
import numpy as np
import pickle
from sys import argv
WINDOW_SIZE = 11
input_placer = tf.placeholder(tf.float32,(None,WINDOW_SIZE,WINDOW_SIZE,WINDOW_SIZE))
gold_placer = tf.placeholder(tf.float32,(None,2))
learn_rate_placer = tf.placeholder(tf.float32)
keep_prob = tf.placeholder(tf.float32)
args = [1,30,30,5,128,0.01,0.5]

for i in range(1,len(argv)):
	args = int(argv[i-1])

version = args[0]#1
angle_xy = int(args[1])#30
angle_xz = int(args[2])#30
epic_num = int(args[3])#5
batch_size = int(args[4])#128
learn_rate = float(args[5])#0.01
kp = float(args[6])#0.5

ft = ForegroundTrain(input_placer,gold_placer,learn_rate_placer,keep_prob,
					 angle_xy = angle_xy, angle_xz = angle_xz)

with tf.Session() as sess:
	ft.sess = sess
	ft.saver = tf.train.Saver()
	init_op = tf.initialize_all_variables()
	ft.sess.run(init_op)
	ft.train(epic_num = epic_num,keep_prob = kp, batch_size = batch_size,learning_rate = learn_rate)
	ft.saver.save(ft.sess,'./model/model_foreground_'+str(angle_xy)+'_'+str(angle_xz)+'_'+version+'.ckpt')