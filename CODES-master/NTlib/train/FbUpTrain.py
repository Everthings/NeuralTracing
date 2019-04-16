import tensorflow as tf
from NTlib.nn.FbUp import FbUp
from NTlib.preprocess.batch_generator import generate_batch
import numpy as np
from scipy import misc
from NTlib.preprocess.dilation3D import dilation3D
import pickle

class FbUpTrain(object):
	def __init__(self, sess, input_placer, gold_placer, keep_prob,
					learn_rate_placer, train_flag = True,
					valid_num = 2048, wd = 1e-4,
					angle_xy = 30, angle_xz = 30):
		self.sess = sess
		self.saver = None
		self.train_flag = train_flag
		self.angle_xy = angle_xy
		self.angle_xz = angle_xz

		if self.train_flag:
			self.valid = generate_batch(valid_num, \
				target_angles_xy = angle_xy,
				target_angles_xz = angle_xz,
				big_cubic_len = 39, sml_cubi_len = 15)
		with tf.variable_scope("fg_"+str(angle_xy)+'_'+str(angle_xz)):
			self.get_nn(input_placer, gold_placer,keep_prob,
				learn_rate_placer, wd)

	def get_nn(self, input_placer, gold_placer, keep_prob,
				learn_rate_placer, wd):
		self.x = input_placer
		self.y_ = gold_placer
		self.l_rate = learn_rate_placer
		self.keep_prob = keep_prob
		self.para_dict = dict()
		para_dict = FbUp().generate_flow(input_placer,
												keep_prob,
												wd = wd)
		self.y_conv = para_dict['y_conv']
		self.y_res = para_dict['y_res']
		if self.train_flag:
			self.y_conv_all = self.y_conv
			self.y_conv_2x = dilation3D(self.y_conv_all)
			self.y_res_all = self.y_res
			self.shape = tf.shape(self.y_res)
			width = self.shape[1]
			self.y_conv = self.y_conv[:,3,3,3,:]
			self.y_res = self.y_res[:,3,3,3]

			cross_entropy_mean = -tf.reduce_mean(self.y_ * tf.log(self.y_conv))
			tf.add_to_collection('losses', cross_entropy_mean)
			cross_entropy = tf.add_n(tf.get_collection('losses'), name='total_loss')
			self.train_step = tf.train.AdamOptimizer(self.l_rate).minimize(cross_entropy)
			correct_prediction = tf.equal(tf.argmax(self.y_conv,1), tf.argmax(self.y_,1))
			self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
		else:
			self.y_conv_all = self.y_conv
			self.y_conv_2x = dilation3D(self.y_conv_all)
			self.y_res_all = self.y_res
			self.y_res_2x = dilation3D(self.y_res_all)
			self.para_dict['y_conv'] = self.y_conv_2x
			self.para_dict['y_res'] = self.y_res_2x
			self.para_dict['h_conv4'] = para_dict['h_conv4']
			self.para_dict['h_pool2'] = para_dict['h_pool2']
			self.para_dict['x'] = para_dict['x']
			self.para_dict['h_conv3'] = para_dict['h_conv3']
			self.para_dict['h_conv2'] = para_dict['h_conv2']
			self.para_dict['h_conv_out'] = para_dict['h_conv_out']

	def next_batch(self,batch_size):
		return generate_batch(batch_size,
				target_angles_xy = self.angle_xy,
				target_angles_xz = self.angle_xz,
				big_cubic_len = 39, sml_cubi_len = 15)

	def get_valid_set(self,window_size = 11):
		return self.valid

	def train(self, epic_num = 8,loop_size = 2100,batch_size = 32,learning_rate = 0.01, thres = 0.01,keep_prob = 1.0, tao = 10):
		accuracies = [0.0,0.05,0.1,0.15,0.2,.25,.3,.35,.4,.45,.5]
		tao_counter = 0
		for epic in range(epic_num):
			for i in range(loop_size):
				#ts = time.time()
				batch = self.next_batch(batch_size)
				#te = time.time()
				#print('Gen Batch: ', te-ts)
				if i%100 == 0:
					print(learning_rate)
					valid = self.get_valid_set()
					train_accuracy = self.accuracy.eval(session = self.sess, feed_dict={self.x: np.reshape(valid[0],(valid[0].shape[0],valid[0].shape[1],valid[0].shape[2],valid[0].shape[3],1)),
											   											self.y_: valid[1],
											   											self.keep_prob: 1.0,
											   											self.l_rate: learning_rate})
					
					ca  = self.shape.eval(session = self.sess, feed_dict={self.x: np.reshape(valid[0],(valid[0].shape[0],valid[0].shape[1],valid[0].shape[2],valid[0].shape[3],1)),
											   											self.y_: valid[1],
											   											self.keep_prob: 1.0,
											   											self.l_rate: learning_rate})
					print(ca)
					
					tao_counter += 1
					#print corrects
					print("%d step %d, training accuracy %g"%(epic,i, train_accuracy))
					if np.std(accuracies[-10:]) < thres or tao_counter > tao:
						learning_rate = max(1e-6,learning_rate/2)
						thres /= 2
						if np.std(accuracies[11:]) == 0:
							 print("Converged")
							 break
						accuracies = [0.0,0.05,0.1,0.15,0.2,.25,.3,.35,.4,.45,.5]
						tao_counter = 0

					accuracies.append(train_accuracy)
				#corrects = correct_prediction.eval(feed_dict={x1: batch[0],x2: batch[1], y_: batch[2], keep_prob: 1.0,l_rate: learn_rate * (4.0-epic)})
				#reinforce = append_reinforce(corrects,batch,reinforce)
				#ts = time.time()
				self.train_step.run(session = self.sess,feed_dict={self.x: np.reshape(batch[0],(batch[0].shape[0],batch[0].shape[1],batch[0].shape[2],batch[0].shape[3],1)),
											   			self.y_: batch[1],
											   			self.keep_prob: keep_prob,
											   			self.l_rate: learning_rate})
				#te = time.time()
				#print('Train: ',te-ts)
		print(accuracies[-1])
		with open('log.dict','rb') as fin:
			accuracy = pickle.load(fin)
		accuracy[(self.angle_xy, self.angle_xz)] = accuracies[-1]
		with open('log.dict','wb') as fout:
			pickle.dump(accuracy,fout)

	def test(self,image):
		sx,sy,sz = image.shape
		batch = np.reshape(image,(1,sx,sy,sz))
		prob_result = self.y_conv_all.eval(session = self.sess,
									  	   feed_dict = {
									  	   self.x:np.reshape(batch/255.0,(batch.shape[0],batch.shape[1],batch.shape[2],batch.shape[3],1)),
									  	   self.y_:np.zeros((1,2)),
									  	   self.keep_prob:1.0,
									  	   self.l_rate: 0
									   	   })
		res_result = self.y_conv_2x.eval(session = self.sess,
									  	   feed_dict = {
									  	   self.x:np.reshape(batch/255.0,(batch.shape[0],batch.shape[1],batch.shape[2],batch.shape[3],1)),
									  	   self.y_:np.zeros((1,2)),
									  	   self.keep_prob:1.0,
									  	   self.l_rate: 0
									   	   })
		print(prob_result.shape)
		prob_result[0,0,0,0,1] = 1
		res_result[0,0,0,0] = 1
		misc.imsave('zzhp0.png',np.max(prob_result[0,5:-5,5:-5,5:-5,1],axis = 0)*255)
		misc.imsave('zzhp1.png',np.max(prob_result[0,5:-5,5:-5,5:-5,1],axis = 1)*255)
		misc.imsave('zzhp2.png',np.max(prob_result[0,5:-5,5:-5,5:-5,1],axis = 2)*255)
		misc.imsave('zzhr0.png',np.max(res_result[0,5:-5,5:-5,5:-5,1],axis = 0)*255)
		misc.imsave('zzhr1.png',np.max(res_result[0,5:-5,5:-5,5:-5,1],axis = 1)*255)
		misc.imsave('zzhr2.png',np.max(res_result[0,5:-5,5:-5,5:-5,1],axis = 2)*255)

