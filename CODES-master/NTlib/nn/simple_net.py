import tensorflow as tf

class SimpleNet(object):
	def weight_variable(self,shape,wd = 0.0,
						previous_layer_num = None,
						current_layer_num = None):
		if previous_layer_num == None:
			initial = tf.truncated_normal(shape, stddev = 0.1)
		else:
			initial = tf.random_uniform(shape,
						minval = -(6**0.5)/((previous_layer_num+current_layer_num)**0.5),
						maxval = (6**0.5)/((previous_layer_num+current_layer_num)**0.5))
		var = tf.Variable(initial,name = 'weight')
		weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
		tf.add_to_collection('losses', weight_decay)
		return var

	def bias_variable(self,shape):
		initial = tf.constant(0.0, shape = shape)
		return tf.Variable(initial,name = 'bias')

	def conv2d(self,x, W, padding = 'SAME'):
		return tf.nn.conv3d(x, W, strides = [1,1,1,1,1], padding = padding)

	def pool_3x3(self,x,pool_type = 'avg',window_size = 3):
		if pool_type == 'avg':
			return tf.nn.avg_pool3d(x, ksize=[1,window_size,window_size,window_size,1], strides = [1,2,2,2,1],
							 padding = 'SAME')
		else:
			return tf.nn.max_pool3d(x, ksize=[1,window_size,window_size,window_size,1], strides = [1,2,2,2,1],
							 padding = 'SAME')

	def generate_flow(self,x,keep_prob,window_size = 11, wd = 0.001, wdo = 0.001,
						   input_num = 1,
						   neuron_num_1 = 4,
						   neuron_num_2 = 4,
						   neuron_num_3 = 8,
						   neuron_num_4 = 8,
						   fc_neuron_num = 16,
						   conv_1 = 3,
						   conv_2 = 3,
						   conv_3 = 3,
						   conv_4 = 3,
						   pool_type = 'avg'):
		#x = tf.placeholder("float", shape = [None,window_size,window_size])
		para_dict = dict()
		#print("Config For the NN\nConv%d-%d\nConv%d-%d\nConv%d-%d\nConv%d-%d\nFC-%d" % (conv_1,neuron_num_1,conv_2,neuron_num_2,conv_3,neuron_num_3,conv_4,neuron_num_4,fc_neuron_num))
		total_para = conv_1 ** 3 * neuron_num_1 \
				   + conv_2 ** 3 * neuron_num_1 * neuron_num_2\
				   + conv_3 ** 3 * neuron_num_2 * neuron_num_3\
				   + conv_4 ** 3 * neuron_num_3 * neuron_num_4\
				   + 0 * window_size ** 3 * neuron_num_4 * fc_neuron_num
		#print("Total Para: Conv %d FC %d" % (total_para,fc_neuron_num))
		x_image = tf.reshape(x, [-1,window_size,window_size,window_size,input_num])

		with tf.variable_scope("sn_conv1"):
			W_conv1 = self.weight_variable([conv_1,conv_1,conv_1,input_num,neuron_num_1],wd,27 * input_num,27 * neuron_num_1)
			b_conv1 = self.bias_variable([neuron_num_1])

			h_conv1 = tf.nn.relu(self.conv2d(x_image, W_conv1) + b_conv1)

		with tf.variable_scope("sn_conv2"):
			W_conv2 = self.weight_variable([conv_2,conv_2,conv_2,neuron_num_1,neuron_num_2],wd,27 * neuron_num_1,27 * neuron_num_2)
			b_conv2 = self.bias_variable([neuron_num_2])

			h_conv2 = tf.nn.relu(self.conv2d(h_conv1, W_conv2) + b_conv2)

		with tf.variable_scope("sn_conv3"):
			W_conv3 = self.weight_variable([conv_3,conv_3,conv_3,neuron_num_2,neuron_num_3],wd,27 * neuron_num_2,27 * neuron_num_3)
			b_conv3 = self.bias_variable([neuron_num_3])

			h_conv3 = tf.nn.relu(self.conv2d(h_conv2, W_conv3) + b_conv3)
		
		with tf.variable_scope("sn_conv4"):
			W_conv4 = self.weight_variable([conv_4,conv_4,conv_4,neuron_num_3,neuron_num_4],wd,27 * neuron_num_3,27 * neuron_num_4)
			b_conv4 = self.bias_variable([neuron_num_4])

			h_conv4 = tf.nn.relu(self.conv2d(h_conv3, W_conv4) + b_conv4)
		
		"""
		with tf.variable_scope("sn_fc1"):
			W_fc1 = self.weight_variable([window_size * window_size * window_size * neuron_num_4, fc_neuron_num],wd,window_size * window_size * window_size * neuron_num_4,fc_neuron_num)
			b_fc1 = self.bias_variable([fc_neuron_num])

			h_pool2_flat = tf.reshape(h_conv4, [-1, window_size * window_size * window_size * neuron_num_4])
			h_pool2_flat_drop = tf.nn.dropout(h_pool2_flat, keep_prob)
			h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat_drop, W_fc1) + b_fc1)
			weight_decay = tf.mul(tf.nn.l2_loss(h_fc1), wdo, name='weight_loss')
			tf.add_to_collection('losses', weight_decay)
		"""
		
		with  tf.variable_scope("sn_fconv"):
			W_fc1 = self.weight_variable([window_size,window_size,window_size,neuron_num_4,fc_neuron_num],wd,window_size ** 3 * neuron_num_4, 1 * fc_neuron_num )
			b_fc1 = self.bias_variable([fc_neuron_num])

			h_conv4_dropped = tf.nn.dropout(h_conv4,keep_prob)

			h_fc1_conv = tf.nn.relu(self.conv2d(h_conv4_dropped,W_fc1,padding = 'VALID') + b_fc1)
			h_fc1 = tf.squeeze(h_fc1_conv,[1,2,3])
			weight_decay = tf.mul(tf.nn.l2_loss(h_fc1), wdo, name='weight_loss')
			tf.add_to_collection('losses', weight_decay)
		
		para_dict['W_conv1'] = W_conv1
		para_dict['b_conv1'] = b_conv1
		para_dict['W_conv2'] = W_conv2
		para_dict['b_conv2'] = b_conv2
		para_dict['W_conv3'] = W_conv3
		para_dict['b_conv3'] = b_conv3
		para_dict['W_conv4'] = W_conv4
		para_dict['b_conv4'] = b_conv4
		para_dict['W_fc1'] = W_fc1
		para_dict['b_fc1'] = b_fc1
		para_dict['h_fc1'] = h_fc1
		return para_dict


