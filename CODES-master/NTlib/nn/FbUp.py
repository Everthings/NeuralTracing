import tensorflow as tf

class FbUp(object):
	def weight_variable(self,shape,wd = 0.0,
						previous_layer_num = None,
						current_layer_num = None,
						loss_type = 'L2'):
		if False or previous_layer_num == None:
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

	def pool_3x3(self,x,pool_type = 'max',window_size = 3):
		if pool_type == 'avg':
			return tf.nn.avg_pool3d(x, ksize=[1,window_size,window_size,window_size,1], strides = [1,2,2,2,1],
							 padding = 'VALID')
		else:
			return tf.nn.max_pool3d(x, ksize=[1,window_size,window_size,window_size,1], strides = [1,2,2,2,1],
							 padding = 'VALID')

	def generate_flow(self,x,keep_prob,window_size = 21, wd = 1e-6, wdo = 0.001,
						   input_num = 1,
						   neuron_num_1 = 4,
						   neuron_num_2 = 8,
						   neuron_num_3 = 16,
						   neuron_num_4 = 32,
						   conv_1 = 3,
						   conv_2 = 3,
						   conv_3 = 3,
						   conv_4 = 3,
						   pool_type = 'avg'):

		para_dict = dict()

		total_para = conv_1 ** 3 * neuron_num_1 \
				   + conv_2 ** 3 * neuron_num_1 * neuron_num_2\
				   + conv_3 ** 3 * neuron_num_2 * neuron_num_3

		x_image = x

		with tf.variable_scope("fb_conv1"):
			W_conv1 = self.weight_variable([conv_1,conv_1,conv_1,input_num,neuron_num_1],wd,27 * input_num,27 * neuron_num_1)
			b_conv1 = self.bias_variable([neuron_num_1])

			h_conv1 = tf.nn.relu(self.conv2d(x_image, W_conv1) + b_conv1)

		with tf.variable_scope("fb_conv2"):
			W_conv2 = self.weight_variable([conv_2,conv_2,conv_2,neuron_num_1,neuron_num_2],wd,27 * neuron_num_1,27 * neuron_num_2)
			b_conv2 = self.bias_variable([neuron_num_2])

			h_conv2 = tf.nn.relu(self.conv2d(h_conv1, W_conv2) + b_conv2)

			h_pool2 = self.pool_3x3(h_conv2)

		with tf.variable_scope("fb_conv3"):
			W_conv3 = self.weight_variable([conv_3,conv_3,conv_3,neuron_num_2,neuron_num_3],wd,27 * neuron_num_2,27 * neuron_num_3)
			b_conv3 = self.bias_variable([neuron_num_3])

			h_conv3 = tf.nn.relu(self.conv2d(h_pool2, W_conv3) + b_conv3)

		with tf.variable_scope("fb_conv4"):
			W_conv4 = self.weight_variable([conv_4,conv_4,conv_4,neuron_num_3,neuron_num_4],wd,27 * neuron_num_3,27 * neuron_num_4)
			b_conv4 = self.bias_variable([neuron_num_4])

			h_conv4 = tf.nn.relu(self.conv2d(h_conv3, W_conv4) + b_conv4)

			h_conv4 = tf.nn.dropout(h_conv4, keep_prob)

		with tf.variable_scope("fb_out"):
			W_conv_out = self.weight_variable([1,1,1,neuron_num_4,2],wd,27 * neuron_num_4,1 * 2)
			b_conv_out = self.bias_variable([2])

			h_conv_out = tf.nn.relu(self.conv2d(h_conv4, W_conv_out) + b_conv_out)

			curr_shape = tf.shape(h_conv_out)
			reshaped_conv = tf.reshape(h_conv_out,[-1,2])
			softmaxed_reshaped_conv = tf.nn.softmax(reshaped_conv)
			y_conv = tf.reshape(softmaxed_reshaped_conv,curr_shape)
			y_res = tf.argmax(y_conv,4)

		para_dict['W_conv1'] = W_conv1
		para_dict['b_conv1'] = b_conv1
		para_dict['h_conv1'] = h_conv1

		para_dict['W_conv2'] = W_conv2
		para_dict['b_conv2'] = b_conv2
		para_dict['h_conv2'] = h_conv2
		para_dict['h_pool2'] = h_pool2

		para_dict['W_conv3'] = W_conv3
		para_dict['b_conv3'] = b_conv3
		para_dict['h_conv3'] = h_conv3

		para_dict['W_conv4'] = W_conv4
		para_dict['b_conv4'] = b_conv4
		para_dict['h_conv4'] = h_conv4

		para_dict['W_conv_out'] = W_conv_out
		para_dict['b_conv_out'] = b_conv_out
		para_dict['h_conv_out'] = h_conv_out
		para_dict['y_conv'] = y_conv
		para_dict['y_res'] = y_res
		para_dict['x'] = x
		return para_dict


