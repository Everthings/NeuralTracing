import tensorflow as tf
from NTlib.nn.simple_net import *
from NTlib.conv_dn_fg_train import CONVDNForegroundTrain
from NTlib.preprocess.sampling import *
import time
class ConvFgCombine(object):
	def __init__(self,sess,input_placer,gold_placer,learn_rate_placer,keep_prob,window_size = 11,model_path = './model/version_11'):
		self.sess = sess
		self.get_nn(input_placer,gold_placer,learn_rate_placer,keep_prob,window_size)
		print("Creation Complete\nIntialization")
		init_op = tf.initialize_all_variables()
		print("Start Intialization")
		sess.run(init_op)
		print("Intialization Done")
		self.restore(model_path)
		print("Saving")
		self.saver = tf.train.Saver()
		self.saver.save(self.sess,model_path + '/combined_model.ckpt')
		print("Saved at " + model_path + '/combined_model.ckpt')
	def get_nn(self,input_placer,gold_placer,learn_rate_placer,keep_prob,window_size):
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
					 	predict_flag = False,
					 	valid_flag = False,
					 	window_size = window_size)
				fg_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope = "fg_"+str(angle_xy)+'_'+str(angle_xz))
				fg_dict[(angle_xy,angle_xz)].saver = \
					tf.train.Saver(fg_variables)
				#fg_dict[(angle_xy,angle_xz)].saver.restore(self.sess,\
				#		model_path + '/model_foreground_'+str(angle_xy)+'_'+str(angle_xz)+'.ckpt')
				y_conv_combined.append(
					tf.expand_dims(fg_dict[(angle_xy,angle_xz)].fg_para_dict['y_conv'],5))
				y_res_combined.append(
					tf.expand_dims(fg_dict[(angle_xy,angle_xz)].fg_para_dict['y_res'],4))

		y_conv_combined = tf.concat(5,y_conv_combined)
		y_res_combined = tf.concat(4,y_res_combined)
		self.y_conv_combined = y_conv_combined
		self.y_res_combined = y_res_combined
		self.fg_dict = fg_dict
	def restore(self,model_path):
		print("Restoring...")
		for angle_xy in range(30,181,30):
			for angle_xz in range(30,181,30):
				print("Restore %d %d" % (angle_xy,angle_xz))
				self.fg_dict[(angle_xy,angle_xz)].saver.restore(self.sess,\
						model_path + '/model_foreground_'+str(angle_xy)+'_'+str(angle_xz)+'.ckpt')
		print("Done")

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


