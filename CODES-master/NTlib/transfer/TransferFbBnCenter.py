import tensorflow as tf
from NTlib.preprocess.dilation3D import dilation3D
from NTlib.nn.FbUpBN import FbUpBN
from NTlib.train.FbUpTrainBN import FbUpTrainBN
from NTlib.preprocess.ImageListBatchGenerator import ImageListBatchGenerator
from NTlib.preprocess.sampling import image3D_crop_generator
import time
import numpy as np
from scipy import misc
class TransferFbBn(object):
	def __init__(self,sess,input_placer,gold_placer,learn_rate_placer,keep_prob,
				 window_size = 15,model_path = './model/version_16',
				 image_path = './data/neuron/test',
				 valid_flag = False,valid_num = 2048,
				 nn = FbUpBN(), predict_flag = True,
				 wd = 0.001, wdo = 0.001,
				 load_path = 'model/version_16/transfer_model_b1_1e-06.ckpt'):
		self.sess = sess
		self.batch_generator = None
		self.nn = nn
		self.saver = None
		self.predict_flag = predict_flag
		self.fg_para_dict = dict()
		self.valid_flag = valid_flag
		self.window_size = window_size
		if valid_flag:
			self.batch_generator = ImageListBatchGenerator((window_size - 1)//2,image_path = image_path, neg_radius = 5, shaking = 0)
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
		self.saver = tf.train.Saver()
		self.saver.save(self.sess,model_path + '/transfer_model.ckpt')
		#print("Saved at " + model_path + '/transfer_model.ckpt')
	def get_nn(self,input_placer,gold_placer,learn_rate_placer,keep_prob,window_size,
					conv_1_neuron_num = 256,
					conv_2_neuron_num = 128,
					conv_3_neuron_num = 128,
					conv_4_neuron_num = 64,
					conv_5_neuron_num = 64,
					conv_6_neuron_num = 64,
					conv_7_neuron_num = 64,
					up_conv_1_neuron_num = 256,
					up_conv_1 = 3,
					conv_1 = 1,
					conv_2 = 3,
					conv_3 = 3,
					conv_4 = 3,
					conv_5 = 3,
					conv_6 = 3,
					conv_7 = 1,
					wd = 1e-9, wdo = 1e-9):
		if self.predict_flag:
			padding = 'VALID'
		else:
			padding = 'SAME'
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
					FbUpTrainBN(self.sess,input_placer,gold_placer,\
						keep_prob,learn_rate_placer,\
					 	angle_xy = angle_xy, \
					 	angle_xz = angle_xz,
					 	train_flag = False,
					 	wd = 0e-9)
				fg_variables = tf.get_collection(tf.GraphKeys.VARIABLES,scope = "fg_"+str(angle_xy)+'_'+str(angle_xz))
				fg_dict[(angle_xy,angle_xz)].saver = \
					tf.train.Saver(fg_variables)
				#fg_dict[(angle_xy,angle_xz)].saver.restore(self.sess,\
				#		model_path + '/model_foreground_'+str(angle_xy)+'_'+str(angle_xz)+'.ckpt')
				y_conv_combined.append(
					fg_dict[(angle_xy,angle_xz)].para_dict['h_conv3'])
				#y_res_combined.append(
				#	tf.expand_dims(fg_dict[(angle_xy,angle_xz)].y_res,4))
		FEAT_NUM = 16
		self.fg_dict = fg_dict
		y_conv_combined = tf.concat(4,y_conv_combined)
		dir_out = y_conv_combined

		self.dir_out = dir_out
		with tf.variable_scope("transfer"):

			with tf.variable_scope("transfer_layer_1"):
				W_conv1 = self.nn.weight_variable([conv_1, conv_1, conv_1, 36*FEAT_NUM, conv_1_neuron_num],wd,1 ** 3 * 36 * FEAT_NUM, conv_1 ** 3 * conv_1_neuron_num)
				tmp_1 = self.nn.conv2d(dir_out, W_conv1, padding = padding)
				h_conv1 = tf.nn.relu(self.batch_norm_layer(tmp_1,self.predict_flag))

			with tf.variable_scope("transfer_layer_2"):
				W_conv2 = self.nn.weight_variable([conv_2,conv_2,conv_2,conv_1_neuron_num,conv_2_neuron_num],wd,conv_1 ** 3 * 2*conv_1_neuron_num, conv_2 ** 3 * conv_2_neuron_num)
				tmp_2 = self.nn.conv2d(h_conv1, W_conv2, padding = padding)
				h_conv2 = tf.nn.relu(self.batch_norm_layer(tmp_2,self.predict_flag))	
			
			with tf.variable_scope("transfer_layer_3"):
				W_conv3 = self.nn.weight_variable([conv_3,conv_3,conv_3,conv_2_neuron_num,conv_3_neuron_num],wd,conv_2 ** 3 * conv_2_neuron_num, conv_3 ** 3 * conv_3_neuron_num)
				b_conv3 = self.nn.bias_variable([conv_3_neuron_num])
				h_conv3 = tf.nn.relu(tf.nn.bias_add(self.nn.conv2d(h_conv2, W_conv3, padding = padding) , b_conv3))
			"""
			with tf.variable_scope("transfer_layer_4"):
				W_conv4 = self.nn.weight_variable([conv_4,conv_4,conv_4,conv_3_neuron_num,conv_4_neuron_num],wd,conv_3 ** 3 * conv_3_neuron_num,conv_4 ** 3 * conv_4_neuron_num)
				b_conv4 = self.nn.bias_variable([conv_4_neuron_num])
				if self.predict_flag:
					residual_4 = h_conv3[:,1:8,1:8,1:8,:]
				else:
					residual_4 = h_conv3
				h_conv4 = tf.nn.relu(tf.add(tf.nn.bias_add(self.nn.conv2d(h_conv3, W_conv4, padding = padding) , b_conv4),residual_4))
		
			with tf.variable_scope("transfer_layer_5"):
				W_conv5 = self.nn.weight_variable([conv_5,conv_5,conv_5,conv_4_neuron_num,conv_5_neuron_num],wd,conv_4 ** 3 * conv_4_neuron_num,conv_5 ** 3 * conv_5_neuron_num)
				b_conv5 = self.nn.bias_variable([conv_5_neuron_num])
				if self.predict_flag:
					residual_5 = h_conv4[:,1:6,1:6,1:6,:]
				else:
					residual_5 = h_conv4
				h_conv5 = tf.nn.relu(tf.add(tf.nn.bias_add(self.nn.conv2d(h_conv4, W_conv5, padding = padding) , b_conv5),residual_5))	

			with tf.variable_scope("transfer_layer_6"):
				W_conv6 = self.nn.weight_variable([conv_6,conv_6,conv_6,conv_5_neuron_num,conv_6_neuron_num],wd,conv_5 ** 3 * conv_5_neuron_num,conv_6 ** 3 * conv_6_neuron_num)
				b_conv6 = self.nn.bias_variable([conv_5_neuron_num])
				if self.predict_flag:
					residual_6 = h_conv5[:,1:4,1:4,1:4,:]
				else:
					residual_6 = h_conv5
				h_conv6 = tf.nn.relu(tf.add(tf.nn.bias_add(self.nn.conv2d(h_conv5, W_conv6, padding = padding) , b_conv6),residual_6))

			with tf.variable_scope("transfer_layer_7"):
				W_conv7 = self.nn.weight_variable([conv_7,conv_7,conv_7,conv_6_neuron_num,conv_7_neuron_num],wd,conv_6 ** 3 * conv_6_neuron_num,conv_7 ** 3 * conv_7_neuron_num)
				b_conv7 = self.nn.bias_variable([conv_7_neuron_num])
				h_conv7 = tf.nn.relu(tf.nn.bias_add(self.nn.conv2d(h_conv6, W_conv7, padding = padding) , b_conv7))	

			weight_decay = tf.mul(tf.nn.l2_loss(h_conv7), wd, name='weight_loss')
			tf.add_to_collection('losses', weight_decay)
			"""
			with tf.variable_scope("transfer_final"):
				W_fc1 = self.nn.weight_variable([1,1,1,conv_2_neuron_num,2],wd = wdo)
				b_fc1 = self.nn.bias_variable([2])
				h_fc1 = (tf.nn.bias_add(self.nn.conv2d(h_conv2, W_fc1, padding = padding) , b_fc1))
			self.h_fc1 = h_fc1
		if self.predict_flag:
			final_shape = tf.shape(h_fc1)
			self.final_shape = final_shape
			#self.y_conv = tf.nn.softmax(h_fc1[:,(final_shape[1]-1)/2,(final_shape[2]-1)/2,(final_shape[3]-1)/2,:])
			self.y_conv = tf.nn.softmax(h_fc1[:,2,2,2,:])
			self.y_res = tf.argmax(self.y_conv,1)
			conv_center3x3 = h_fc1
			reshaped_conv_center3x3 = tf.reshape(conv_center3x3,[-1,2])
			softmaxed_reshaped_conv_center3x3 = tf.nn.softmax(reshaped_conv_center3x3)
			self.y_conv_center3x3 = tf.reshape(softmaxed_reshaped_conv_center3x3,
												tf.shape(conv_center3x3))
		else:
			input_shape = tf.shape(input_placer)
			curr_shape = tf.shape(h_fc1)
			reshaped_conv = tf.reshape(h_fc1,[-1,2])
			softmaxed_reshaped_conv = tf.nn.softmax(reshaped_conv)
			self.y_conv = tf.reshape(softmaxed_reshaped_conv,curr_shape)
			self.y_res = tf.argmax(self.y_conv,4)
			self.y_conv_2x = dilation3D(self.y_conv,input_shape)
			tmp = tf.expand_dims(self.y_res,4)
			self.y_res_2x = tf.squeeze(dilation3D(tmp,input_shape))

		if self.valid_flag:
			cross_entropy_mean = -tf.reduce_mean(self.y_ * tf.log(self.y_conv))
			#For Centerline
			center_sum = tf.reduce_sum(self.y_conv_center3x3,[1,2,3])
			#self.line_penalty = tf.mul(wdo,tf.reduce_sum(tf.mul(self.y_[:,1],tf.abs(tf.sub(center_sum[:,1],3)))))
			#self.ce = cross_entropy_mean
			#tf.add_to_collection('losses', self.line_penalty)

			tf.add_to_collection('losses', cross_entropy_mean)
			cross_entropy = tf.add_n(tf.get_collection('losses'), name='total_loss')

			transfer_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"transfer")
			
			self.train_step = tf.train.AdamOptimizer(self.l_rate).minimize(cross_entropy, var_list = transfer_vars)
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

	def get_valid_set(self):
		return self.valid

	def batch_norm_layer(self, inputs, is_training, decay = 0.99):
		epsilon = 1e-3
		scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
		beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
		pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
		pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

		if is_training:
			batch_mean, batch_var = tf.nn.moments(inputs,[0,1,2,3])
			train_mean = tf.assign(pop_mean,
					pop_mean * decay + batch_mean * (1 - decay))
			train_var = tf.assign(pop_var,
					pop_var * decay + batch_var * (1 - decay))
			with tf.control_dependencies([train_mean, train_var]):
				return tf.nn.batch_normalization(inputs,
					batch_mean, batch_var, beta, scale, epsilon)
		else:
			return tf.nn.batch_normalization(inputs,
				pop_mean, pop_var, beta, scale, epsilon)

	def train(self, epic_num = 8,loop_size = 2100,batch_size = 32,learning_rate = 0.01, thres = 0.01,keep_prob = 1.0, tao = 50):
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
					valid = self.get_valid_set()
					print(np.max(valid[0]))
					valid_counter = 0
					for v_idx in range(0,len(valid[0]),256):
						train_accuracy += self.accuracy.eval(session = self.sess, feed_dict={self.x: valid[0][v_idx:min(v_idx+256,len(valid[0]))]/255.0,
											   											self.y_: valid[1][v_idx:min(v_idx+256,len(valid[0]))],
											   											self.keep_prob: 1.0,
											   											self.l_rate: learning_rate})
						valid_counter += 1
						dir_out = self.h_fc1.eval(session = self.sess, feed_dict={self.x: valid[0][v_idx:min(v_idx+256,len(valid[0]))]/255.0,
											   											self.y_: valid[1][v_idx:min(v_idx+256,len(valid[0]))],
											   											self.keep_prob: 1.0,
											   											self.l_rate: learning_rate})
						print(dir_out.shape)
					train_accuracy /= valid_counter
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
				self.train_step.run(session = self.sess,feed_dict={self.x: batch[0]/255.0,
											   			self.y_: batch[1],
											   			self.keep_prob: keep_prob,
											   			self.l_rate: learning_rate})
				#te = time.time()
				#print('Train: ',te-ts)

	def predict_large(self,image,version = '',folder = '',batch_diameter = 128, extra_edge_width = 15):
		sx,sy,sz = image.shape
		print(sx,sy,sz)
		#batches,pos,neg = sampling(image)
		result = np.zeros((sx,sy,sz))
		max_res = np.zeros((sx,sy,sz))

		for x_start,y_start,z_start,batch in image3D_crop_generator(image,batch_diameter,extra_edge_width):
			ts = time.time()
			num,x_len,y_len,z_len,_ = batch.shape
			print(batch.shape)
			
			tmp_res = self.y_res_2x.eval(session = self.sess,
									  	   feed_dict = {
									  	   self.x:batch/255.0,
									  	   self.y_:np.zeros((num,2)),
									  	   self.keep_prob:1.0,
									  	   self.l_rate: 0
									   	   })
			print(tmp_res.shape)
			tmp_res = tmp_res[extra_edge_width:-extra_edge_width,
								extra_edge_width:-extra_edge_width,
								extra_edge_width:-extra_edge_width]
			
			tmp_cov = self.y_conv_2x.eval(session = self.sess,
									  	   feed_dict = {
									  	   self.x:batch/255.0,
									  	   self.y_:np.zeros((num,2)),
									  	   self.keep_prob:1.0,
									  	   self.l_rate: 0
									   	   })
			tmp_cov = tmp_cov[0,extra_edge_width:-extra_edge_width,
								extra_edge_width:-extra_edge_width,
								extra_edge_width:-extra_edge_width,1]
			
			#tmp_res = np.max(tmp_res,axis = 1)
			result[ x_start:x_start+x_len - 2*extra_edge_width,
					y_start:y_start+y_len - 2*extra_edge_width,
					z_start:z_start+z_len - 2*extra_edge_width]\
					 = tmp_cov
			max_res[ x_start:x_start+x_len - 2*extra_edge_width,
					y_start:y_start+y_len - 2*extra_edge_width,
					z_start:z_start+z_len - 2*extra_edge_width]\
					 = tmp_res
			te = time.time()
			print(x_start,y_start,z_start,te-ts)
		print(np.max(result))
		print(np.max(max_res))
		misc.imsave(folder+'test'+version+'_max0.png',np.max(max_res[10:-10,10:-10,10:-10],axis = 0)*255)
		misc.imsave(folder+'test'+version+'_max1.png',np.max(max_res[10:-10,10:-10,10:-10],axis = 1)*255)
		misc.imsave(folder+'test'+version+'_max2.png',np.max(max_res[10:-10,10:-10,10:-10],axis = 2)*255)
		misc.imsave(folder+'test'+version+'_cov0.png',np.max(result[10:-10,10:-10,10:-10],axis = 0)*255)
		misc.imsave(folder+'test'+version+'_cov1.png',np.max(result[10:-10,10:-10,10:-10],axis = 1)*255)
		misc.imsave(folder+'test'+version+'_cov2.png',np.max(result[10:-10,10:-10,10:-10],axis = 2)*255)


