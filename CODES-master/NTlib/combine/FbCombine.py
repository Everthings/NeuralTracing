from NTlib.train.FbUpTrainBN import FbUpTrainBN
import tensorflow as tf
from NTlib.preprocess.sampling import *
import time

class FbCombine(object):
	def __init__(self,sess,input_placer,gold_placer,learn_rate_placer,keep_prob,model_path = './model/version_13'):
		self.sess = sess
		self.get_nn(input_placer,gold_placer,learn_rate_placer,keep_prob)
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

	def get_nn(self,input_placer,gold_placer,learn_rate_placer,keep_prob):
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
					 	train_flag = False)
				fg_variables = tf.get_collection(tf.GraphKeys.VARIABLES,scope = "fg_"+str(angle_xy)+'_'+str(angle_xz))
				fg_dict[(angle_xy,angle_xz)].saver = \
					tf.train.Saver(fg_variables)
				#fg_dict[(angle_xy,angle_xz)].saver.restore(self.sess,\
				#		model_path + '/model_foreground_'+str(angle_xy)+'_'+str(angle_xz)+'.ckpt')
				y_conv_combined.append(
					tf.expand_dims(fg_dict[(angle_xy,angle_xz)].y_conv_2x,5))
				y_res_combined.append(
					tf.expand_dims(fg_dict[(angle_xy,angle_xz)].y_res_2x,4))

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

	def predict_large(self,image,batch_diameter = 512, window_radius = 32):
		sx,sy,sz = image.shape
		print(sx,sy,sz)
		#batches,pos,neg = sampling(image)
		result = np.zeros((sx,sy,sz,36))
		max_res = np.zeros((sx,sy,sz))

		for x_start,y_start,z_start,batch in image3D_crop_generator(image,batch_diameter,window_radius = window_radius):
			ts = time.time()
			#batch = np.reshape(batch,(1,sx,sy,sz,1))
			num,x_len,y_len,z_len,_ = batch.shape
			print(batch.shape)
			print('batch max', np.max(batch))
			"""
			tmp_cov = self.y_conv_combined.eval(session = self.sess,
									  	   feed_dict = {
									  	   self.x:batch/255.0,
									  	   self.y_:np.zeros((num,2)),
									  	   self.keep_prob:1.0,
									  	   self.l_rate: 0
									   	   })
			#print('tmp_cov',tmp_cov.shape)
			tmp_cov = tmp_cov[0,window_radius:-window_radius,window_radius:-window_radius,window_radius:-window_radius,1,:]
			#tmp_cov = tmp_cov[0,:,:,:,0,:]
			"""
			tmp_cov = self.y_res_combined.eval(session = self.sess,
									  	   feed_dict = {
									  	   self.x:batch/255.0,
									  	   self.y_:np.zeros((num,2)),
									  	   self.keep_prob:1.0,
									  	   self.l_rate: 0
									   	   })
			tmp_cov = tmp_cov[0,window_radius:-window_radius,window_radius:-window_radius,window_radius:-window_radius,:]
			
			print(tmp_cov.shape)
			#tmp_res = np.max(tmp_res,axis = 1)
			result[ x_start:x_start+x_len - 2 * window_radius,
					y_start:y_start+y_len - 2 * window_radius,
					z_start:z_start+z_len - 2 * window_radius, : ] = tmp_cov
			te = time.time()
			print(x_start,y_start,z_start,te-ts)
		print(np.max(result),np.min(result))
		#print(result.shape)
		max_res = np.max(result,axis = 3)
		misc.imsave('test_max0.png',np.max(max_res[5:-5,5:-5,5:-5],axis = 0)*255)
		misc.imsave('test_max1.png',np.max(max_res[5:-5,5:-5,5:-5],axis = 1)*255)
		misc.imsave('test_max2.png',np.max(max_res[5:-5,5:-5,5:-5],axis = 2)*255)
		for i in range(36):
			misc.imsave('testr_'+str(i)+'_0.png',np.max(result[5:-5,5:-5,5:-5,i],axis = 0)*255)
			misc.imsave('testr_'+str(i)+'_1.png',np.max(result[5:-5,5:-5,5:-5,i],axis = 1)*255)
			misc.imsave('testr_'+str(i)+'_2.png',np.max(result[5:-5,5:-5,5:-5,i],axis = 2)*255)
