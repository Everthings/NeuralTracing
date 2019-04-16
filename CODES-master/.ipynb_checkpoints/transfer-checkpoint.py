from NTlib.transfer.TransferCenterBN import TransferCenterBN
import tensorflow as tf
import numpy as np
from sys import argv
wd = float(argv[1])
wdo = float(argv[2])
lr = float(argv[3])
print('wdo = ',wdo)
np.random.seed(17)
tf.set_random_seed(17)
input_placer = tf.placeholder(tf.float32,(None,None,None,None))
gold_placer = tf.placeholder(tf.float32,(None,2))
learn_rate_placer = tf.placeholder(tf.float32)
keep_prob = tf.placeholder(tf.float32)
with tf.Session() as sess:
	tft = TransferCenterBN(sess,input_placer,gold_placer,learn_rate_placer,keep_prob,
						  valid_flag = True,valid_num = 1024,
						  wd = wdo, wdo = wdo, window_size = 11,
						  model_path = './model/2d/version_0')
	tft.saver = tf.train.Saver()
	tft.train(epic_num = 1, loop_size = 2000, batch_size = 1,learning_rate = lr)
	model_path = './model/2d/version_0'
	tft.saver.save(sess,model_path + '/transfer_model_b1_'+str(wdo)+'.ckpt')
	print("Saved at " + model_path + '/transfer_model_b1_'+str(wdo)+'.ckpt')