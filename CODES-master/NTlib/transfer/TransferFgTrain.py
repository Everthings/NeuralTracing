import tensorflow as tf
from NTlib.nn.deep_narrow_net import *
from NTlib.train.FbUpTrain import FbUpTrain
from NTlib.preprocess.ImageListBatchGenerator import ImageListBatchGenerator
import time
import numpy as np

class TransferFgTrain(object):
	def __init__(self,sess,input_placer,gold_placer,learn_rate_placer,keep_prob,
				 window_size = 21,model_path = './model/version_11',
				 image_path = './data/neuron/train',
				 valid_flag = False,valid_num = 2048,
				 nn = DNNet(), predict_flag = True,
				 wd = 0.001, wdo = 0.001):
		self.sess = sess
		self.batch_generator = ImageListBatchGenerator((window_size - 1)//2,image_path = image_path)
		self.nn = nn
		self.saver = None
		self.predict_flag = predict_flag
		self.fg_para_dict = dict()
		self.valid_flag = valid_flag
		self.window_size = window_size
		if valid_flag:
			self.valid = self.next_batch(valid_num)

		self.get_nn(input_placer,gold_placer,learn_rate_placer,
			   		keep_prob,wd = wd, wdo = wdo, window_size = window_size)

		print("Creation Complete\nIntialization")
		init_op = tf.initialize_all_variables()
		print("Start Intialization")
		sess.run(init_op)
		print("Intialization Done")
		self.restore(model_path)
		print("Saving")
		#self.saver = tf.train.Saver()
		#self.saver.save(self.sess,model_path + '/transfer_model.ckpt')
		#print("Saved at " + model_path + '/transfer_model.ckpt')
	def get_nn(self,input_placer,gold_placer,learn_rate_placer,keep_prob,window_size,
					conv_1_neuron_num = 512,
					conv_2_neuron_num = 256,
					conv_3_neuron_num = 128,
					conv_4_neuron_num = 64,
					conv_5_neuron_num = 32,
					wd = 1e-9, wdo = 1e-9):
		#create sub classes
		self.x = input_placer
		self.y_ = gold_placer
		self.l_rate = learn_rate_placer
		self.keep_prob = keep_prob
		fg_dict = dict()
		y_conv_combined = []
		y_res_combined = []
		for angle_xy in range(30,181,30):
			for angle_xz in range(30,181,30):
				print("Create FG NN %d %d" % (angle_xy,angle_xz))
				fg_dict[(angle_xy,angle_xz)] = \
					CONVDNForegroundTrain(input_placer,gold_placer,\
						learn_rate_placer,keep_prob,\
					 	angle_xy = angle_xy, \
					 	angle_xz = angle_xz,
					 	predict_flag = True,
					 	valid_flag = False,
					 	window_size = window_size)
				fg_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope = "fg_"+str(angle_xy)+'_'+str(angle_xz))
				fg_dict[(angle_xy,angle_xz)].saver = \
					tf.train.Saver(fg_variables)
				#fg_dict[(angle_xy,angle_xz)].saver.restore(self.sess,\
				#		model_path + '/model_foreground_'+str(angle_xy)+'_'+str(angle_xz)+'.ckpt')
				y_conv_combined.append(
					tf.expand_dims(fg_dict[(angle_xy,angle_xz)].fg_para_dict['h_conv5'],5))
		self.fg_dict = fg_dict
		y_conv_combined = tf.concat(5,y_conv_combined)
		dir_out = tf.reshape(y_conv_combined,[-1,window_size,window_size,window_size,36 * 16])
		with tf.variable_scope("transfer"):
			with tf.variable_scope("transfer_layer_1"):
				W_conv1 = self.nn.weight_variable([3,3,3,36*16,conv_1_neuron_num],wd,27 * 36 * 16,27 * conv_1_neuron_num)
				b_conv1 = self.nn.bias_variable([conv_1_neuron_num])
				h_conv1 = tf.nn.relu(tf.nn.bias_add(self.nn.conv2d(dir_out, W_conv1) , b_conv1))	

			with tf.variable_scope("transfer_layer_2"):
				W_conv2 = self.nn.weight_variable([3,3,3,conv_1_neuron_num,conv_2_neuron_num],wd,27 * conv_1_neuron_num,27 * conv_2_neuron_num)
				b_conv2 = self.nn.bias_variable([conv_2_neuron_num])
				h_conv2 = tf.nn.relu(tf.nn.bias_add(self.nn.conv2d(h_conv1, W_conv2) , b_conv2))	
			
			with tf.variable_scope("transfer_layer_3"):
				W_conv3 = self.nn.weight_variable([3,3,3,conv_2_neuron_num,conv_3_neuron_num],wd,27 * conv_2_neuron_num,27 * conv_3_neuron_num)
				b_conv3 = self.nn.bias_variable([conv_3_neuron_num])
				h_conv3 = tf.nn.relu(tf.nn.bias_add(self.nn.conv2d(h_conv2, W_conv3) , b_conv3))
		
			with tf.variable_scope("transfer_layer_4"):
				W_conv4 = self.nn.weight_variable([3,3,3,conv_3_neuron_num,conv_4_neuron_num],wd,27 * conv_3_neuron_num,27 * conv_4_neuron_num)
				b_conv4 = self.nn.bias_variable([conv_4_neuron_num])
				h_conv4 = tf.nn.relu(tf.nn.bias_add(self.nn.conv2d(h_conv3, W_conv4) , b_conv4))
		
			with tf.variable_scope("transfer_layer_5"):
				W_conv5 = self.nn.weight_variable([3,3,3,conv_4_neuron_num,conv_5_neuron_num],wd,27 * conv_4_neuron_num,27 * conv_5_neuron_num)
				b_conv5 = self.nn.bias_variable([conv_5_neuron_num])
				h_conv5 = tf.nn.relu(tf.nn.bias_add(self.nn.conv2d(h_conv4, W_conv5) , b_conv5))	
			
			with tf.variable_scope("transfer_final"):
				W_fc1 = self.nn.weight_variable([1,1,1,conv_5_neuron_num,2])
				b_fc1 = self.nn.bias_variable([2])
				h_fc1 = tf.nn.relu(tf.nn.bias_add(self.nn.conv2d(h_conv5, W_fc1) , b_fc1))
		
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

			transfer_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"transfer")
			
			self.train_step = tf.train.AdamOptimizer(self.l_rate).minimize(cross_entropy)#, var_list = transfer_vars)
			correct_prediction = tf.equal(tf.argmax(self.y_conv,1), tf.argmax(self.y_,1))
			self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
		else:
			print("Transfer Not For Training Singly")
	def restore(self,model_path):
		print("Restoring...")
		for angle_xy in range(30,181,30):
			for angle_xz in range(30,181,30):
				print("Restore %d %d" % (angle_xy,angle_xz))
				self.fg_dict[(angle_xy,angle_xz)].saver.restore(self.sess,\
						model_path + '/model_foreground_'+str(angle_xy)+'_'+str(angle_xz)+'.ckpt')
		print("Done")
	def next_batch(self,batch_size):
		return self.batch_generator.next_batch(batch_size)

	def get_valid_set(self,valid_num):
		return self.batch_generator.next_batch(valid_num)

	def train(self, epic_num = 8,loop_size = 2100,batch_size = 32,learning_rate = 0.01, thres = 0.01,keep_prob = 1.0, tao = 10):
		accuracies = [0.0,0.05,0.1,0.15,0.2,.25,.3,.35,.4,.45,.5]
		tao_counter = 0
		for epic in range(epic_num):
			for i in range(loop_size):
				#ts = time.time()
				batch = self.next_batch(batch_size)
				#te = time.time()
				#print('Gen Batch: ', te-ts)
				if i%10 == 0:
					print(learning_rate)
					train_accuracy = 0
					for v_idx in range(4):
						valid = self.get_valid_set(256)
						train_accuracy += self.accuracy.eval(session = self.sess, feed_dict={self.x: valid[0],
											   											self.y_: valid[1],
											   											self.keep_prob: 1.0,
											   											self.l_rate: learning_rate})
					train_accuracy /= 4
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
				self.train_step.run(session = self.sess,feed_dict={self.x: batch[0],
											   			self.y_: batch[1],
											   			self.keep_prob: keep_prob,
											   			self.l_rate: learning_rate})
				#te = time.time()
				#print('Train: ',te-ts)

	def predict(self,image,batch_size = 512):
		sx,sy,sz = image.shape
		batches,pos,neg = sampling(image)
		result = np.zeros((sx * sy * sz,36))
		max_res = np.zeros((sx * sy * sz))
		counter = 0
		for batch in batch_generator(batches,batch_size):
			num = batch.shape[0]
			#print(np.max(batch))
			
			tmp_cov = self.y_conv_combined.eval(session = self.sess,
									  	   feed_dict = {
									  	   self.x:batch/255.0,
									  	   self.y_:np.zeros((num,2)),
									  	   self.keep_prob:1.0,
									  	   self.l_rate: 0
									   	   })
			tmp_cov = np.reshape(tmp_cov[:,1,:],(num,-1))
			"""
			tmp_cov = self.y_res_combined.eval(session = self.sess,
									  	   feed_dict = {
									  	   self.x:batch/255.0,
									  	   self.y_:np.zeros((num,2)),
									  	   self.keep_prob:1.0,
									  	   self.l_rate: 0
									   	   })
			"""
			#tmp_res = np.max(tmp_res,axis = 1)
			result[counter:counter+num,:] = tmp_cov
			max_res[counter:counter+num] = np.max(tmp_cov,axis = 1)
			counter += num
			print(counter)
		print(np.max(result))
		result = np.reshape(result,(sx,sy,sz,36))
		max_res = np.reshape(max_res,(sx,sy,sz))
		misc.imsave('test_max.png',np.max(max_res[5:-5,5:-5,5:-5],axis = 0)*255)
		misc.imsave('test_max.png',np.max(max_res[5:-5,5:-5,5:-5],axis = 1)*255)
		misc.imsave('test_max.png',np.max(max_res[5:-5,5:-5,5:-5],axis = 2)*255)
		for i in range(36):
			misc.imsave('testr_'+str(i)+'_0.png',np.max(result[5:-5,5:-5,5:-5,i],axis = 0)*255)
			misc.imsave('testr_'+str(i)+'_1.png',np.max(result[5:-5,5:-5,5:-5,i],axis = 1)*255)
			misc.imsave('testr_'+str(i)+'_2.png',np.max(result[5:-5,5:-5,5:-5,i],axis = 2)*255)

	def predict_large(self,image,batch_diameter = 512):
		sx,sy,sz = image.shape
		print(sx,sy,sz)
		#batches,pos,neg = sampling(image)
		result = np.zeros((sx,sy,sz,36))
		max_res = np.zeros((sx,sy,sz))

		for x_start,y_start,z_start,batch in image3D_crop_generator(image,batch_diameter):
			ts = time.time()
			num,x_len,y_len,z_len,_ = batch.shape
			#print(batch.shape)
			
			tmp_cov = self.y_conv_combined.eval(session = self.sess,
									  	   feed_dict = {
									  	   self.x:batch/255.0,
									  	   self.y_:np.zeros((num,2)),
									  	   self.keep_prob:1.0,
									  	   self.l_rate: 0
									   	   })
			tmp_cov = tmp_cov[0,5:-5,5:-5,5:-5,1,:]
			"""
			tmp_cov = self.y_res_combined.eval(session = self.sess,
									  	   feed_dict = {
									  	   self.x:batch/255.0,
									  	   self.y_:np.zeros((num,2)),
									  	   self.keep_prob:1.0,
									  	   self.l_rate: 0
									   	   })
			tmp_cov = tmp_cov[0,5:-5,5:-5,5:-5,:]
			"""
			#tmp_res = np.max(tmp_res,axis = 1)
			result[ x_start:x_start+x_len - 10,
					y_start:y_start+y_len - 10,
					z_start:z_start+z_len - 10, : ] = tmp_cov
			te = time.time()
			print(x_start,y_start,z_start,te-ts)
		print(np.max(result))
		max_res = np.max(result,axis = 3)
		misc.imsave('test_max0.png',np.max(max_res[5:-5,5:-5,5:-5],axis = 0)*255)
		misc.imsave('test_max1.png',np.max(max_res[5:-5,5:-5,5:-5],axis = 1)*255)
		misc.imsave('test_max2.png',np.max(max_res[5:-5,5:-5,5:-5],axis = 2)*255)
		for i in range(36):
			misc.imsave('testr_'+str(i)+'_0.png',np.max(result[5:-5,5:-5,5:-5,i],axis = 0)*255)
			misc.imsave('testr_'+str(i)+'_1.png',np.max(result[5:-5,5:-5,5:-5,i],axis = 1)*255)
			misc.imsave('testr_'+str(i)+'_2.png',np.max(result[5:-5,5:-5,5:-5,i],axis = 2)*255)


