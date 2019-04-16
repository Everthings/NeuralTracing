import tensorflow as tf
from NTlib.nn.FbUpBN2D import FbUpBN2D
from NTlib.train.CenterFbUpTrainBN2D import CenterFbUpTrainBN2D
from NTlib.train.FbUpTrainBN2D import FbUpTrainBN2D
from NTlib.preprocess.ImageBatchGenerator2D import ImageListBatchGenerator
from NTlib.preprocess.sampling import image3D_crop_generator
import time
import numpy as np
from scipy import misc
class TransferCenterBN(object):
	def __init__(self,sess,input_placer,gold_placer,learn_rate_placer,keep_prob,
				 window_size = 11,model_path = './model/2d/version_0',
				 image_path = './data/neuron/test',
				 valid_flag = False,valid_num = 2048,
				 nn = FbUpBN2D(), predict_flag = True,
				 wd = 0.001, wdo = 0.001):
		self.sess = sess
		self.batch_generator = None
		self.nn = nn
		self.saver = None
		self.predict_flag = predict_flag
		self.fg_para_dict = dict()
		self.valid_flag = valid_flag
		self.window_size = window_size
		if valid_flag:
			self.batch_generator = ImageListBatchGenerator((window_size - 1)//2,image_path = image_path)
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
					conv_1_neuron_num = 48,
					conv_2_neuron_num = 24,
					conv_1 = 1,
					conv_2 = 1,
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
			print("Create FG NN %d" % (angle_xy))
			fg_dict[angle_xy] = \
				CenterFbUpTrainBN2D(self.sess,input_placer,gold_placer,\
					keep_prob,learn_rate_placer,\
				 	angle_xy = angle_xy, \
				 	train_flag = False,
				 	wd = 0e-9)
			fg_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope = "fg_"+str(angle_xy))
			cn_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope = "centerline_"+str(angle_xy))
			fg_dict[angle_xy].saver = tf.train.Saver(fg_variables+cn_variables)

			y_conv_combined.append(
				fg_dict[angle_xy].para_dict['h_conv1'])
		self.fg_dict = fg_dict
		#dir_out = tf.concat(3,y_conv_combined)
		dir_out = tf.concat(y_conv_combined, 3)
		self.dir_out = dir_out
		with tf.variable_scope("transfer"):

			with tf.variable_scope("transfer_layer_1"):
				W_conv1 = self.nn.weight_variable([conv_1, conv_1, 6*8, conv_1_neuron_num],wd,1 ** 3 * 6 * 8, conv_1 ** 3 * conv_1_neuron_num,loss_type = 'L2')
				#b_conv1 = self.nn.bias_variable([conv_1_neuron_num])
				tmp_1 = self.nn.conv2d(dir_out, W_conv1)
				h_conv1 = tf.nn.relu(self.batch_norm_layer(tmp_1,self.predict_flag))
			
			with tf.variable_scope("transfer_layer_2"):
				W_conv2 = self.nn.weight_variable([conv_2, conv_2, conv_1_neuron_num, conv_2_neuron_num],wd,1 ** 3 * conv_1_neuron_num, conv_1 ** 3 * conv_2_neuron_num,loss_type = 'L2')
				#b_conv1 = self.nn.bias_variable([conv_1_neuron_num])
				tmp_2 = self.nn.conv2d(h_conv1, W_conv2)
				h_conv2 = tf.nn.relu(self.batch_norm_layer(tmp_2,self.predict_flag))
			"""
			weight_decay = tf.mul(tf.nn.l2_loss(h_conv7), wd, name='weight_loss')
			tf.add_to_collection('losses', weight_decay)
			"""
			with tf.variable_scope("transfer_final"):
				W_fc1 = self.nn.weight_variable([1,1,conv_2_neuron_num,2],wd = wdo,loss_type = 'L2')
				b_fc1 = self.nn.bias_variable([2])
				h_fc1 = (tf.nn.bias_add(self.nn.conv2d(h_conv2, W_fc1) , b_fc1))
		
		if self.predict_flag:
			final_shape = tf.shape(h_fc1)
			self.final_shape = final_shape
			#self.y_conv = tf.nn.softmax(h_fc1[:,(final_shape[1]-1)/2,(final_shape[2]-1)/2,(final_shape[3]-1)/2,:])
			self.y_conv = tf.nn.softmax(h_fc1[:,2,2,:])
			self.y_res = tf.argmax(self.y_conv,1)
		else:
			curr_shape = tf.shape(h_fc1)
			reshaped_conv = tf.reshape(h_fc1,[-1,2])
			softmaxed_reshaped_conv = tf.nn.softmax(reshaped_conv)
			self.y_conv = tf.reshape(softmaxed_reshaped_conv,curr_shape)
			self.y_res = tf.argmax(self.y_conv,3)
			self.y_conv_2x = tf.image.resize_bilinear(self.y_conv,tf.shape(input_placer)[1:3])
			tmp = tf.expand_dims(self.y_res,3)
			self.y_res_2x = tf.squeeze(tf.image.resize_bilinear(tmp,tf.shape(input_placer)[1:3]),[3])

		if self.valid_flag:
			cross_entropy_mean = -tf.reduce_mean(self.y_ * tf.log(self.y_conv))

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
			print("Restore %d" % (angle_xy))
			self.fg_dict[angle_xy].saver.restore(self.sess,\
					model_path + '/model_center_'+str(angle_xy)+'.ckpt')
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
			batch_mean, batch_var = tf.nn.moments(inputs,[0,1,2])
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
				if i%100 == 0:
					
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
						dir_out = self.dir_out.eval(session = self.sess, feed_dict={self.x: valid[0][v_idx:min(v_idx+256,len(valid[0]))]/255.0,
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

	def predict_large(self,image,version = '',batch_diameter = 64, extra_edge_width = 15):
		sx,sy,sz = image.shape
		print(sx,sy,sz)
		#batches,pos,neg = sampling(image)
		result = np.zeros((sx,sy,sz))
		max_res = np.zeros((sx,sy,sz))

		for x_start,y_start,z_start,batch in image3D_crop_generator(image,batch_diameter,extra_edge_width):
			ts = time.time()
			num,x_len,y_len,z_len,_ = batch.shape
			#print(batch.shape)
			
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
		misc.imsave('test'+version+'_max0.png',np.max(max_res[10:-10,10:-10,10:-10],axis = 0)*255)
		misc.imsave('test'+version+'_max1.png',np.max(max_res[10:-10,10:-10,10:-10],axis = 1)*255)
		misc.imsave('test'+version+'_max2.png',np.max(max_res[10:-10,10:-10,10:-10],axis = 2)*255)
		misc.imsave('test'+version+'_cov0.png',np.max(result[10:-10,10:-10,10:-10],axis = 0)*255)
		misc.imsave('test'+version+'_cov1.png',np.max(result[10:-10,10:-10,10:-10],axis = 1)*255)
		misc.imsave('test'+version+'_cov2.png',np.max(result[10:-10,10:-10,10:-10],axis = 2)*255)
	def process_stack(self,image,version):
		sx,sy,sz = image.shape
		print(sx,sy,sz)
		#batches,pos,neg = sampling(image)
		result = np.zeros((sx,sy,sz))
		max_res = np.zeros((sx,sy,sz))
		for zidx in range(1,sz-1):
			curr_2d = 0.25 * (image[:,:,zidx-1] + image[:,:,zidx+1]) + 0.5 * image[:,:,zidx]
			batch = np.reshape(curr_2d,[1,sx,sy,1])
			tmp_cov = self.y_res_2x.eval(session = self.sess,
									  	   feed_dict = {
									  	   self.x:batch/255.0,
									  	   self.y_:np.zeros((1,2)),
									  	   self.keep_prob:1.0,
									  	   self.l_rate: 0
									   	   })
			result[:,:,zidx] = tmp_cov[0,:,:]
		misc.imsave('zhihao_'+version+'_0.png',np.max(result[10:-10,10:-10,:],axis=0)*255)
		misc.imsave('zhihao_'+version+'_1.png',np.max(result[10:-10,10:-10,:],axis=1)*255)
		misc.imsave('zhihao_'+version+'_2.png',np.max(result[10:-10,10:-10,:],axis=2)*255)


