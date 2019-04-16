import tensorflow as tf
from NTlib.nn.deep_narrow_net import *
from NTlib.nn.shallow_wide_net import *
from NTlib.preprocess.sampling import *
from NTlib.preprocess.batch_generator import *
import random
import time
import pickle
class CONVDNForegroundTrain(object):
	def __init__(self,input_placer,gold_placer,learn_rate_placer,
			   	 keep_prob,angle_xy = 30,angle_xz = 30,valid_flag = False,valid_num = 2048,
			   	 wd = 0.001, wdo = 0.001, nn = DNNet(), predict_flag = True,
			   	 window_size = 11):
		self.nn = nn
		self.sess = None
		self.saver = None
		self.predict_flag = predict_flag
		self.angle_xy = angle_xy
		self.angle_xz = angle_xz
		self.fg_para_dict = dict()
		self.valid_flag = valid_flag
		self.window_size = window_size
		if valid_flag:
			self.valid = generate_batch(valid_num, target_angles_xy = angle_xy,target_angles_xz = angle_xz)
		with tf.variable_scope("fg_"+str(angle_xy)+'_'+str(angle_xz)):
			self.get_nn(input_placer,gold_placer,learn_rate_placer,
				   		keep_prob,wd = wd, wdo = wdo, window_size = window_size)
	def get_nn(self,input_placer,gold_placer,learn_rate_placer,
			   keep_prob,window_size = 11,
			   fc_1_neuron_num = 16, wd = 0.001, wdo = 0.001):
		self.x = input_placer
		self.y_ = gold_placer
		self.l_rate = learn_rate_placer
		self.keep_prob = keep_prob
		with tf.variable_scope("fs_train_nn"):
			self.para_dict = self.nn.generate_flow(self.x,self.keep_prob,wd = wd, wdo = wdo,window_size = window_size)
			h_final_conv = self.para_dict['h_conv5']
		with tf.variable_scope("fs_fc2"):
			W_fc2 = self.nn.weight_variable([1,1,1,fc_1_neuron_num,2])
			b_fc2 = self.nn.bias_variable([2])
			h_fc1 = tf.nn.relu(tf.nn.bias_add(self.nn.conv2d(h_final_conv, W_fc2) , b_fc2))
		#h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)
		if self.predict_flag:
			self.y_conv = tf.nn.softmax(h_fc1[:,(window_size-1)/2,(window_size-1)/2,(window_size-1)/2,:])
			self.y_res = tf.argmax(self.y_conv,1)
		else:
			curr_shape = tf.shape(h_fc1)
			reshaped_conv = tf.reshape(h_fc1,[-1,2])
			softmaxed_reshaped_conv = tf.nn.softmax(reshaped_conv)
			self.y_conv = tf.reshape(softmaxed_reshaped_conv,curr_shape)
			self.y_res = tf.argmax(self.y_conv,4)
		if self.valid_flag:
			cross_entropy_mean = -tf.reduce_mean(self.y_ * tf.log(self.y_conv))
			tf.add_to_collection('losses', cross_entropy_mean)
			cross_entropy = tf.add_n(tf.get_collection('losses'), name='total_loss')
			self.train_step = tf.train.AdamOptimizer(self.l_rate).minimize(cross_entropy)
			correct_prediction = tf.equal(tf.argmax(self.y_conv,1), tf.argmax(self.y_,1))
			self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
		else:
			print("Not For Training Singly")
		
		self.fg_para_dict['h_fc1'] = h_fc1
		self.fg_para_dict['y_conv'] = self.y_conv
		self.fg_para_dict['y_res'] = self.y_res
		self.fg_para_dict['h_conv5'] = h_final_conv

	def next_batch(self,batch_size):
		return generate_batch(batch_size, target_angles_xy = self.angle_xy,target_angles_xz = self.angle_xz)

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
					tao_counter += 1
					#print corrects
					print("%d step %d, training accuracy %g"%(epic,i, train_accuracy))
					if np.std(accuracies[-10:]) < thres or tao_counter > tao:
						learning_rate = max(1e-6,learning_rate/2)
						thres /= 2
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
	def predict(self,image,batch_size = 64,version = 0):
		sx,sy,sz = image.shape
		batches,pos,neg = sampling(image)
		prob_result = np.zeros((sx * sy * sz,2))
		result = np.zeros((sx * sy * sz))
		counter = 0
		for batch in batch_generator(batches,batch_size):
			num = batch.shape[0]
			#print(np.max(batch))
			prob_result[counter:counter+num,:] = self.y_conv.eval(session = self.sess,
									  	   feed_dict = {
									  	   self.x:np.reshape(batch/255.0,(batch.shape[0],batch.shape[1],batch.shape[2],batch.shape[3],1)),
									  	   self.y_:np.zeros((num,2)),
									  	   self.keep_prob:1.0,
									  	   self.l_rate: 0
									   	   })
			result[counter:counter+num] = self.y_res.eval(session = self.sess,
									  	   feed_dict = {
									  	   self.x:np.reshape(batch/255.0,(batch.shape[0],batch.shape[1],batch.shape[2],batch.shape[3],1)),
									  	   self.y_:np.zeros((num,2)),
									  	   self.keep_prob:1.0,
									  	   self.l_rate: 0
									   	   })
			counter += num
		print(np.max(prob_result))
		prob_result = np.reshape(prob_result[:,1],(sx,sy,sz))
		prob_result = prob_result[5:-5,5:-5,5:-5]
		result = np.reshape(result,(sx,sy,sz))
		result = result[5:-5,5:-5,5:-5]
		misc.imsave('testv'+version+'_0.png',np.max(prob_result,axis = 0)*255)
		misc.imsave('testv'+version+'_1.png',np.max(prob_result,axis = 1)*255)
		misc.imsave('testv'+version+'_2.png',np.max(prob_result,axis = 2)*255)
		misc.imsave('testv'+version+'_r0.png',np.max(result,axis = 0)*255)
		misc.imsave('testv'+version+'_r1.png',np.max(result,axis = 1)*255)
		misc.imsave('testv'+version+'_r2.png',np.max(result,axis = 2)*255)
	def test(self,image):
		sx,sy,sz = image.shape
		batch = np.reshape(image,(1,sx,sy,sz))
		prob_result = self.y_conv.eval(session = self.sess,
									  	   feed_dict = {
									  	   self.x:np.reshape(batch/255.0,(batch.shape[0],batch.shape[1],batch.shape[2],batch.shape[3],1)),
									  	   self.y_:np.zeros((1,2)),
									  	   self.keep_prob:1.0,
									  	   self.l_rate: 0
									   	   })
		print(prob_result.shape)
		misc.imsave('zzh0.png',np.max(prob_result[0,:,:,:,1],axis = 0)*255)
		misc.imsave('zzh1.png',np.max(prob_result[0,:,:,:,1],axis = 1)*255)
		misc.imsave('zzh2.png',np.max(prob_result[0,:,:,:,1],axis = 2)*255)



