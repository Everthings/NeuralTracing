import tensorflow as tf

def dilation3D(input_placer,target_size = None):
	if target_size is None:
		shape = tf.shape(input_placer)
		xy = tf.reshape(input_placer,[shape[0],shape[1],shape[2],shape[3]*shape[4]])
		xy2 = tf.image.resize_bilinear(xy,[shape[1] * 2 + 2, shape[2] * 2 + 2])
		dil = tf.reshape(xy2,[shape[0],shape[1]*2+2,shape[2]*2+2,shape[3],shape[4]])		

		xz = tf.transpose(dil,[0,1,3,2,4])
		xz = tf.reshape(xz,[shape[0],shape[1]*2+2,shape[3],2*(shape[2]+1)*shape[4]])
		xz2 = tf.image.resize_bilinear(xz,[shape[1]*2+2,shape[3]*2+2])
		dil = tf.reshape(xz2,[shape[0], shape[1]*2+2, shape[3]*2+2, 2*shape[2]+2,shape[4]])
		dil = tf.transpose(dil,[0,1,3,2,4])
	else:
		shape = tf.shape(input_placer)
		xy = tf.reshape(input_placer,[shape[0],shape[1],shape[2],shape[3]*shape[4]])
		xy2 = tf.image.resize_bilinear(xy,[target_size[1], target_size[2]])
		dil = tf.reshape(xy2,[shape[0],target_size[1],target_size[2],shape[3],shape[4]])		

		xz = tf.transpose(dil,[0,1,3,2,4])
		xz = tf.reshape(xz,[shape[0],target_size[1],shape[3],target_size[2]*shape[4]])
		xz2 = tf.image.resize_bilinear(xz,[target_size[1],target_size[3]])
		dil = tf.reshape(xz2,[shape[0], target_size[1], target_size[3], target_size[2],shape[4]])
		dil = tf.transpose(dil,[0,1,3,2,4])
	return dil