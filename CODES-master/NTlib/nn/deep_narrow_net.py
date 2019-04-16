import tensorflow as tf

class DNNet(object):
	def weight_variable(self,shape,wd = 0.0,
						previous_layer_num = None,
						current_layer_num = None,
						loss_type = 'L2'):
		if previous_layer_num == None:
			initial = tf.truncated_normal(shape, stddev = 0.1)
		else:
			initial = tf.random_uniform(shape,
						minval = -(6**0.5)/((previous_layer_num+current_layer_num)**0.5),
						maxval = (6**0.5)/((previous_layer_num+current_layer_num)**0.5))
		var = tf.Variable(initial,name = 'weight')
		if loss_type == 'L1':
			weight_decay = tf.mul(wd, tf.reduce_sum(tf.abs(var)), name='weight_loss')
		else:
			weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
		tf.add_to_collection('losses', weight_decay)
		return var

	def bias_variable(self,shape):
		initial = tf.constant(0.0, shape = shape)
		return tf.Variable(initial,name = 'bias')

	def conv2d(self,x, W, padding = 'SAME'):
		return tf.nn.conv3d(x, W, strides = [1,1,1,1,1], padding = padding)

	def pool_3x3(self,x,pool_type = 'avg',window_size = 2):
		if pool_type == 'avg':
			return tf.nn.avg_pool3d(x, ksize=[1,window_size,window_size,window_size,1], strides = [1,2,2,2,1],
							 padding = 'VALID')
		else:
			return tf.nn.max_pool3d(x, ksize=[1,window_size,window_size,window_size,1], strides = [1,2,2,2,1],
							 padding = 'VALID')

	def generate_flow(self,x,keep_prob,window_size = 11, wd = 0.001, wdo = 0.001,
						   input_num = 1,
						   neuron_num_1 = 16,
						   neuron_num_2 = 16,
						   neuron_num_3 = 16,
						   neuron_num_4 = 16,
						   neuron_num_5 = 16,
						   neuron_num_6 = 16,
						   neuron_num_7 = 16,
						   neuron_num_8 = 16,
						   conv_1 = 3,
						   conv_2 = 3,
						   conv_3 = 3,
						   conv_4 = 3,
						   conv_5 = 3,
						   conv_6 = 3,
						   conv_7 = 3,
						   conv_8 = 3,
						   pool_type = 'avg'):
		total_para = conv_1 **3 * neuron_num_1 * input_num +\
					 conv_2 **3 * neuron_num_2 * neuron_num_1 +\
					 conv_3 **3 * neuron_num_3 * neuron_num_2 +\
					 conv_4 **3 * neuron_num_4 * neuron_num_3 +\
					 conv_5 **3 * neuron_num_5 * neuron_num_4 +\
					 conv_6 **3 * neuron_num_6 * neuron_num_5 +\
					 conv_7 **3 * neuron_num_7 * neuron_num_6 +\
					 conv_8 **3 * neuron_num_8 * neuron_num_7
		print("Total Para: ",total_para)
		para_dict = dict()

		x_image = x

		with tf.variable_scope("sn_conv1"):
			W_conv1 = self.weight_variable([conv_1,conv_1,conv_1,input_num,neuron_num_1],wd,27 * input_num,27 * neuron_num_1)
			b_conv1 = self.bias_variable([neuron_num_1])

			h_conv1 = tf.nn.relu(tf.nn.bias_add(self.conv2d(x_image, W_conv1) , b_conv1))

		with tf.variable_scope("sn_conv2"):
			W_conv2 = self.weight_variable([conv_2,conv_2,conv_2,neuron_num_1,neuron_num_2],wd,27 * neuron_num_1,27 * neuron_num_2)
			b_conv2 = self.bias_variable([neuron_num_2])

			h_conv2 = tf.nn.relu(tf.nn.bias_add(self.conv2d(h_conv1, W_conv2) , b_conv2))

		with tf.variable_scope("sn_conv3"):
			W_conv3 = self.weight_variable([conv_3,conv_3,conv_3,neuron_num_2,neuron_num_3],wd,27 * neuron_num_2,27 * neuron_num_3)
			b_conv3 = self.bias_variable([neuron_num_3])

			h_conv3 = tf.nn.relu(tf.nn.bias_add(self.conv2d(h_conv2, W_conv3) , b_conv3))
			#h_conv3_pool = self.pool_3x3(h_conv3)
		
		with tf.variable_scope("sn_conv4"):
			W_conv4 = self.weight_variable([conv_4,conv_4,conv_4,neuron_num_3,neuron_num_4],wd,27 * neuron_num_3,27 * neuron_num_4)
			b_conv4 = self.bias_variable([neuron_num_4])

			h_conv4 = tf.nn.relu(tf.nn.bias_add(self.conv2d(h_conv3, W_conv4) , b_conv4))

		with tf.variable_scope("sn_conv5"):
			W_conv5 = self.weight_variable([conv_5,conv_5,conv_5,neuron_num_4,neuron_num_5],wd,27 * neuron_num_4,27 * neuron_num_5)
			b_conv5 = self.bias_variable([neuron_num_5])

			h_conv5 = tf.nn.relu(tf.nn.bias_add(self.conv2d(h_conv4, W_conv5) , b_conv5))
		"""
		with tf.variable_scope("sn_conv6"):
			W_conv6 = self.weight_variable([conv_6,conv_6,conv_6,neuron_num_5,neuron_num_6],wd,27 * neuron_num_5,27 * neuron_num_6)
			b_conv6 = self.bias_variable([neuron_num_6])

			h_conv6 = tf.nn.relu(tf.nn.bias_add(self.conv2d(h_conv5, W_conv6) , b_conv6))

		with tf.variable_scope("sn_conv7"):
			W_conv7 = self.weight_variable([conv_7,conv_7,conv_7,neuron_num_6,neuron_num_7],wd,27 * neuron_num_6,27 * neuron_num_7)
			b_conv7 = self.bias_variable([neuron_num_7])

			h_conv7 = tf.nn.relu(tf.nn.bias_add(self.conv2d(h_conv6, W_conv7) , b_conv7))

		with tf.variable_scope("sn_conv8"):
			W_conv8 = self.weight_variable([conv_8,conv_8,conv_8,neuron_num_7,neuron_num_8],wd,27 * neuron_num_7,27 * neuron_num_8)
			b_conv8 = self.bias_variable([neuron_num_8])

			h_conv8 = tf.nn.relu(tf.nn.bias_add(self.conv2d(h_conv7, W_conv8) , b_conv8))
		"""
		h_fc1 = h_conv5[:,5,5,5,:]
		weight_decay = tf.mul(tf.nn.l2_loss(h_fc1), wdo, name='weight_loss')
		#weight_decay = tf.mul(wd, tf.reduce_sum(tf.abs(h_fc1)), name='weight_loss')
		tf.add_to_collection('losses', weight_decay)
		para_dict['h_fc1'] = h_fc1
		para_dict['h_conv5'] = h_conv5
		return para_dict


