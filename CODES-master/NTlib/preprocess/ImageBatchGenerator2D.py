import glob
import scipy.io as sio
import numpy as np
from NTlib.preprocess.swc2np import SWCReader
#from swc2np import SWCReader
from scipy import misc

class ImageGoldPair(object):
	def __init__(self,image,gold,window_radius = 10):
		self.image = np.pad(image,((3*window_radius,3*window_radius),(3*window_radius,3*window_radius),(3*window_radius,3*window_radius)),
				 'constant',constant_values=0)
		self.gold = np.pad(gold,((3*window_radius,3*window_radius),(3*window_radius,3*window_radius),(3*window_radius,3*window_radius)),
				 'constant',constant_values=0)
		gold_xs,gold_ys,gold_zs = self.get_gold_sub(self.gold)
		print('ImageGoldPair',self.image.dtype,self.gold.dtype)
		if np.sum(gold_xs < 3 * window_radius) \
			or np.sum(gold_ys < 3 * window_radius) \
			or np.sum(gold_zs < 3 * window_radius):
			print(np.sum(gold_xs < 3 * window_radius))
			print(np.sum(gold_ys < 3 * window_radius))
			print(np.sum(gold_zs < 3 * window_radius))
			err_i = dict()
			err_i['image'] = self.image
			err_g = dict()
			err_g['gold'] = self.gold
			sio.savemat('err_i.mat',err_i)
			sio.savemat('err_g.mat',err_g)
			raise
		self.gold_coord = np.c_[gold_xs,gold_ys,gold_zs]
	def get_gold_sub(self,gold):
		tmp_sub = np.where(gold > 0)
		return tmp_sub[0], tmp_sub[1], tmp_sub[2]
class ImageListBatchGenerator(object):
	def __init__(self,window_radius,neg_radius = 10,
				shaking = 0,image_path = '../../data/neuron/test',
				random_brightness = 50,
				random_contrast_l = 0.2,
				random_contrast_h = 1.8):
		self.data = []
		self.neg_radius = neg_radius
		self.shaking = shaking
		self.random_brightness = random_brightness
		self.random_contrast_l = random_contrast_l
		self.random_contrast_h = random_contrast_h
		print("Loading Neuron Images")
		self.window_radius = window_radius
		self.load_images_and_swc(image_path,window_radius)
        
	def load_images_and_swc(self,image_path,window_radius):
		counter = 0 
		for path in glob.glob(image_path + '/*_raw.mat'):
			image_name = path[len(image_path)+1:-8]

			print(image_name," Start processing...")

			swc_path = glob.glob(image_path + '/' + image_name + '*swc')[0]
			image_mat = sio.loadmat(path)
			image = np.uint8(image_mat['img'] * 255)
			print(np.max(image))
			gold = SWCReader().swc2mat(image.shape,swc_path)
			gold = np.transpose(gold,(1,0,2))
			#print(image.dtype)
			self.data.append(ImageGoldPair(image,gold,window_radius = window_radius))
			print(image_name," Loaded")
			counter += 1
			if np.sum(gold) > 1000: print('WARNING\n'+image_name+'WARNING')


	def next_batch(self,batch_size,
						pos_percent = .3):
		batch = np.zeros((batch_size,self.window_radius * 2 + 1,self.window_radius * 2 + 1,1))
		label = np.zeros((batch_size,2))
		for i in range(batch_size):
			curr_image_gold = np.random.choice(self.data)
			if np.random.random() < pos_percent:
				patch,curr_label = self.pos_patch(curr_image_gold)
				error_info = "pos "+str(patch.shape)
			else:
				patch,curr_label = self.neg_patch(curr_image_gold)
				error_info = "neg "+str(patch.shape)
			try:
				batch[i,:,:,0] = patch
			except:
				print(error_info)
				raise
			if curr_label:
				label[i,1] = 1
			else:
				label[i,0] = 1
		#print(np.sum(label[:,1]))
		return batch,label

	def pos_patch(self,curr_image_gold):
		total,_ = curr_image_gold.gold_coord.shape
		selected = np.random.randint(0,total)
		x = curr_image_gold.gold_coord[selected,0]
		y = curr_image_gold.gold_coord[selected,1]
		z = curr_image_gold.gold_coord[selected,2]
		#print(curr_image_gold.gold[x,y,z])
		non_shaking_label = curr_image_gold.gold[x,y,z]
		if np.random.random() < 0.33:
			x = x + np.random.randint(-self.shaking,self.shaking+1)
		elif np.random.random() < 0.5:
			y = y + np.random.randint(-self.shaking,self.shaking+1)
		else:
			z = z + np.random.randint(-self.shaking,self.shaking+1)
		if x - self.window_radius < 0 \
			or y - self.window_radius < 0 \
			or z - self.window_radius < 0:
			print(curr_image_gold.gold_coord[selected,0],curr_image_gold.gold_coord[selected,1],curr_image_gold.gold_coord[selected,2])
		curr_patch = curr_image_gold.image[x - self.window_radius : x + self.window_radius + 1,
										   y - self.window_radius : y + self.window_radius + 1,
										   z - 1 : z + 2]
		curr_patch = .25 * (curr_patch[:,:,0] + curr_patch[:,:,2]) + 0.5 * curr_patch[:,:,1]
		transpose_inds = [0,1]
		np.random.shuffle(transpose_inds)

		curr_patch = np.transpose(curr_patch,transpose_inds)
		
		#brightness adjust
		curr_patch = np.float32(curr_patch)
		curr_patch += np.random.random() * self.random_brightness
		curr_patch[curr_patch > 255] = 255
		curr_patch[curr_patch < 0] = 0
		#contrast adjust
		curr_patch = (curr_patch - np.mean(curr_patch)) \
			* (np.random.random() * (self.random_contrast_h - self.random_contrast_l) + self.random_contrast_l)\
			+ np.mean(curr_patch)
		curr_patch[curr_patch > 255] = 255
		curr_patch[curr_patch < 0] = 0
		return curr_patch, non_shaking_label#curr_image_gold.gold[x,y,z]
	def neg_patch(self,curr_image_gold):
		total,_ = curr_image_gold.gold_coord.shape
		selected = np.random.randint(0,total)
		x = curr_image_gold.gold_coord[selected,0] + np.random.randint(- self.neg_radius, self.neg_radius + 1)
		y = curr_image_gold.gold_coord[selected,1] + np.random.randint(- self.neg_radius, self.neg_radius + 1)
		z = curr_image_gold.gold_coord[selected,2] + np.random.randint(- self.neg_radius, self.neg_radius + 1)
		if x - self.window_radius < 0 \
			or y - self.window_radius < 0 \
			or z - self.window_radius < 0:
			print(curr_image_gold.gold_coord[selected,0],curr_image_gold.gold_coord[selected,1],curr_image_gold.gold_coord[selected,2])
		curr_patch = curr_image_gold.image[x - self.window_radius : x + self.window_radius + 1,
										   y - self.window_radius : y + self.window_radius + 1,
										   z - 1 : z + 2]

		curr_patch = .25 * (curr_patch[:,:,0] + curr_patch[:,:,2]) + 0.5 * curr_patch[:,:,1]

		transpose_inds = [0,1]
		np.random.shuffle(transpose_inds)

		curr_patch = np.transpose(curr_patch,transpose_inds)
		
		#brightness adjust
		curr_patch = np.float32(curr_patch)
		curr_patch += np.random.random() * self.random_brightness
		curr_patch[curr_patch > 255] = 255
		curr_patch[curr_patch < 0] = 0
		#contrast adjust
		curr_patch = (curr_patch - np.mean(curr_patch)) \
			* (np.random.random() * (self.random_contrast_h - self.random_contrast_l) + self.random_contrast_l)\
			+ np.mean(curr_patch)
		curr_patch[curr_patch > 255] = 255
		curr_patch[curr_patch < 0] = 0
		return curr_patch, curr_image_gold.gold[x,y,z]



if __name__ == '__main__':
	ig = ImageListBatchGenerator(21)
	for i in range(16):
		patch,label = ig.next_batch(1)
		print(np.max(patch))
		misc.imsave('p'+str(i)+'_0'+str(label[0])+'.png',np.max(patch[0,:,:,:,0],axis=0))
		misc.imsave('p'+str(i)+'_1'+str(label[0])+'.png',np.max(patch[0,:,:,:,0],axis=1))
		misc.imsave('p'+str(i)+'_2'+str(label[0])+'.png',np.max(patch[0,:,:,:,0],axis=2))
