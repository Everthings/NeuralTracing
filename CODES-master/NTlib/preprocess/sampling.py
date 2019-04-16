#This Module is to sampling from each image and output sampled patches
import numpy as np
from scipy import misc
class ImagePatch(object):
	def __init__(self, patch, label):
		self.patch = patch
		self.label = label

def sampling(img_3d, gold_sample = None, sample_dict = {},
			 window_radius = 5, background_default_num = 100):
	pos_num = 0
	neg_num = 0
	size_x, size_y, size_z = img_3d.shape
	img = np.pad(img_3d,((window_radius,window_radius),(window_radius,window_radius),(window_radius,window_radius)),
				 'constant',constant_values=np.amin(img_3d))
	if gold_sample is not None:
		gold = gold_sample & np.array(1,dtype=np.uint8) # 001
		sample = ( gold_sample & np.array(6,dtype=np.uint8) ) >> 1# 110
	else:
		gold = np.zeros((size_x,size_y,size_z))
		sample = None

	if gold is not None:
		gold = np.pad(gold,((window_radius,window_radius),(window_radius,window_radius),(window_radius,window_radius)),
				 	  'constant',constant_values=np.amin(gold))
	if sample is not None:
		sample = np.pad(sample,((window_radius,window_radius),(window_radius,window_radius),(window_radius,window_radius)),
				 'constant')

	#hope it's a linked list
	batch = []
	for x_idx in range(window_radius, window_radius + size_x):
		for y_idx in range(window_radius, window_radius + size_y):
			for z_idx in range(window_radius, window_radius + size_z):
				#background has a majority number, process it separately
				if sample is not None and sample[x_idx,y_idx,z_idx] == 0:
					continue
				current_patch = ImagePatch(img[x_idx - window_radius : x_idx + window_radius + 1,
											   y_idx - window_radius : y_idx + window_radius + 1,
											   z_idx - window_radius : z_idx + window_radius + 1],
											   gold[x_idx,y_idx,z_idx] > 0)
				if sample is not None:
					#sampling for training images
					prob = sample_dict[sample[x_idx,y_idx,z_idx]]
					for promised in range(int(prob)):
						batch.append(current_patch)
						if current_patch.label: pos_num += 1
						else: neg_num += 1
					dice_thres = prob - int(prob)
					if np.random.random() < dice_thres:
						batch.append(current_patch)
						if current_patch.label: pos_num += 1
						else: neg_num += 1
				else:
					#simply generate batch
					batch.append(current_patch)
					if current_patch.label: pos_num += 1
					else: neg_num += 1
	if sample is None:
		#batch.append(ImagePatch(np.zeros((11,11,11)),1))
		return batch,pos_num,neg_num

	if pos_num > neg_num + background_default_num:
		background_num = pos_num - neg_num
	else:
		background_num = background_default_num
	for i in range(background_num):
		x_idx = np.random.randint(window_radius,size_x+window_radius)
		y_idx = np.random.randint(window_radius,size_y+window_radius)
		z_idx = np.random.randint(window_radius,size_z+window_radius)
		current_patch = ImagePatch(img[x_idx - window_radius : x_idx + window_radius + 1,
								   y_idx - window_radius : y_idx + window_radius + 1,
								   z_idx - window_radius : z_idx + window_radius + 1],
								   gold[x_idx,y_idx,z_idx] > 0)
		batch.append(current_patch)
		if current_patch.label: pos_num += 1
		else: neg_num += 1

	return batch,pos_num,neg_num
def batch_generator(batch,batch_size = 1024):
	total = len(batch)
	if total < 1:
		return
	sx,sy,sz = batch[0].patch.shape
	print(sx,sy,sz)
	next_batch = np.zeros((batch_size,sx,sy,sz))
	counter = 0
	for i in range(0,total,batch_size):
		for j in range(min(batch_size,total - i)):
			try:
				next_batch[j,:,:,:] = batch[i+j].patch
			except:
				print(i,j,total,batch[i+j].patch)
		if i + batch_size > total:
			yield next_batch[:total-i,:,:,:]
		else:
			yield next_batch

def sampling_and_batch_generator(img3d, batch_size,
								 window_radius = 5):
	size_x,size_y,size_z = img3d.shape
	img = np.pad(img3d,((window_radius,window_radius),
						(window_radius,window_radius),
						(window_radius,window_radius)),
				 	    'constant',constant_values=np.amin(img3d))
	next_batch = np.zeros((batch_size,window_radius * 2 + 1,
									  window_radius * 2 + 1,
									  window_radius * 2 + 1))
	counter = 0
	for x_idx in range(window_radius, window_radius + size_x):
		for y_idx in range(window_radius, window_radius + size_y):
			for z_idx in range(window_radius, window_radius + size_z):
				if counter == batch_size:
					yield next_batch
					counter = 0
					next_batch = np.zeros((batch_size,window_radius * 2 + 1,
													  window_radius * 2 + 1,
									  				  window_radius * 2 + 1))
				next_batch[counter,:,:,:] = img[x_idx - window_radius : x_idx + window_radius + 1,
								   				y_idx - window_radius : y_idx + window_radius + 1,
								   				z_idx - window_radius : z_idx + window_radius + 1]
				counter += 1
	if counter != 0:
		yield next_batch[:counter,:,:,:]

def image3D_crop_generator(image,batch_diameter,window_radius = 5):
	size_x, size_y, size_z = image.shape
	img = np.pad(image,((window_radius,window_radius),
						(window_radius,window_radius),
						(window_radius,window_radius)),
				 	    'constant',constant_values=0)
	for x_idx in range(window_radius, window_radius + size_x, batch_diameter):
		for y_idx in range(window_radius, window_radius + size_y, batch_diameter):
			for z_idx in range(window_radius, window_radius + size_z, batch_diameter):
				end_x_idx = min(x_idx + batch_diameter, window_radius + size_x)
				end_y_idx = min(y_idx + batch_diameter, window_radius + size_y)
				end_z_idx = min(z_idx + batch_diameter, window_radius + size_z)
				yield 	x_idx-window_radius,\
						y_idx-window_radius,\
						z_idx-window_radius,\
						np.reshape(img[x_idx-window_radius:end_x_idx+window_radius,
									   y_idx-window_radius:end_y_idx+window_radius,
									   z_idx-window_radius:end_z_idx+window_radius],
									   (1, end_x_idx - x_idx + 2 * window_radius,
									   	   end_y_idx - y_idx + 2 * window_radius,
									   	   end_z_idx - z_idx + 2 * window_radius, 1))

