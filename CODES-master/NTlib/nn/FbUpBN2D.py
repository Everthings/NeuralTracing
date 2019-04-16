import tensorflow as tf

class FbUpBN2D(object):
	def __init__(self, is_training = False):
		self.is_training = is_training
	def batch_norm_layer(self, inputs, decay = 0.99):
		is_training = self.is_training
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
					batch_mean, batch_var, beta, scale, epsilon),batch_mean,batch_var
		else:
			return tf.nn.batch_normalization(inputs,
				pop_mean, pop_var, beta, scale, epsilon),pop_mean,pop_var

	def weight_variable(self,shape,wd = 0.0,
						previous_layer_num = None,
						current_layer_num = None,
						loss_type = 'L1'):
		if False or previous_layer_num == None:
			initial = tf.truncated_normal(shape, stddev = 0.1)
		else:
			initial = tf.random_uniform(shape,
						minval = -(6**0.5)/((previous_layer_num+current_layer_num)**0.5),
						maxval = (6**0.5)/((previous_layer_num+current_layer_num)**0.5))
		var = tf.Variable(initial,name = 'weight')
		if loss_type == 'L1':
			# weight_decay = tf.mul(wd, tf.reduce_sum(tf.abs(var)), name='weight_loss')
			weight_decay = tf.multiply(wd, tf.reduce_sum(tf.abs(var)), name='weight_loss')
		else:
			# weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
			weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
		tf.add_to_collection('losses', weight_decay)
		return var

	def bias_variable(self,shape):
		initial = tf.constant(0.0, shape = shape)
		return tf.Variable(initial,name = 'bias')

	def conv2d(self,x, W, stride = 1, padding = 'SAME'):
		return tf.nn.conv2d(x, W, strides = [1,stride,stride,1], padding = padding)

	def pool_3x3(self,x,pool_type = 'max',window_size = 3):
		if pool_type == 'avg':
			return tf.nn.avg_pool(x, ksize=[1,window_size,window_size,1], strides = [1,2,2,1],
							 padding = 'VALID')
		else:
			return tf.nn.max_pool(x, ksize=[1,window_size,window_size,1], strides = [1,2,2,1],
							 padding = 'VALID')

	def generate_flow(self,x,keep_prob,window_size = 15, wd = 1e-6, wdo = 0.001,
						   input_num = 1,
						   neuron_num_1 = 4,
						   neuron_num_2 = 8,
						   neuron_num_3 = 16,
						   conv_1 = 5,
						   conv_2 = 3,
						   conv_3 = 3,
						   pool_type = 'avg'):

		para_dict = dict()

		with tf.variable_scope("fb_conv1"):
			W_conv1 = self.weight_variable([conv_1,conv_1,input_num,neuron_num_1],wd,conv_1**2 * input_num,conv_1**2 * neuron_num_1)
			#b_conv1 = self.bias_variable([neuron_num_1])
			tmp_1,_,_ = self.batch_norm_layer(self.conv2d(x, W_conv1))
			h_conv1 = tf.nn.relu(tmp_1)

			h_pool1 = self.pool_3x3(h_conv1, pool_type = pool_type)

		with tf.variable_scope("fb_conv2"):
			W_conv2 = self.weight_variable([conv_2,conv_2,neuron_num_1,neuron_num_2],wd,conv_2**2 * neuron_num_1,conv_2**2 * neuron_num_2)
			#b_conv2 = self.bias_variable([neuron_num_2])
			tmp_2,_,_ = self.batch_norm_layer(self.conv2d(h_pool1, W_conv2))
			h_conv2 = tf.nn.relu(tmp_2)

			#h_pool2 = self.pool_3x3(h_conv2, pool_type = pool_type)
		with tf.variable_scope("fb_conv3"):
			W_conv3 = self.weight_variable([conv_3,conv_3,neuron_num_2,neuron_num_3],wd,conv_3**2 * neuron_num_2,conv_3**2 * neuron_num_3)
			#b_conv2 = self.bias_variable([neuron_num_2])
			tmp_3,_,_ = self.batch_norm_layer(self.conv2d(h_conv2, W_conv3))
			h_conv3 = tf.nn.relu(tmp_3)

		with tf.variable_scope("fb_out"):
			W_conv_out = self.weight_variable([1,1,neuron_num_3,2],0.0,1**2 * neuron_num_3,1 * 2)
			b_conv_out = self.bias_variable([2])
			tmp_out = self.conv2d(h_conv3, W_conv_out)
			h_conv_out = tmp_out#tf.nn.relu(self.batch_norm_layer(tmp_out))

			curr_shape = tf.shape(h_conv_out)
			reshaped_conv = tf.reshape(h_conv_out,[-1,2])
			softmaxed_reshaped_conv = tf.nn.softmax(reshaped_conv)
			y_conv = tf.reshape(softmaxed_reshaped_conv,curr_shape)
			y_res = tf.argmax(y_conv,3)

		para_dict['y_conv'] = y_conv
		para_dict['y_res'] = y_res
		para_dict['h_conv3'] = h_conv3
		para_dict['x'] = x
		return para_dict


