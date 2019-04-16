import tensorflow as tf

class SWNet(object):
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
						   neuron_num_2 = 8,
						   neuron_num_3 = 16,
						   neuron_num_4 = 16,
						   conv_1 = 5,
						   conv_2 = 5,
						   conv_3 = 5,
						   conv_4 = 5,
						   pool_type = 'avg'):

		para_dict = dict()

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

			h_fc1 = h_conv4[:,5,5,5,:]
		para_dict['h_fc1'] = h_fc1
		para_dict['h_conv4'] = h_conv4
		return para_dict


