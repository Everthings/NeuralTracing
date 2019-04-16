import tensorflow as tf
import numpy as np
from scipy import misc
ANGLE_RANGE = 40

class BatchGenConfig(object):
	def __init__(self,angles_xy_range, angles_xz_range, shift_x, shift_y, shift_z):
		self.angles_xy_range = angles_xy_range
		self.angles_xz_range = angles_xz_range
		self.shift_x = shift_x
		self.shift_y = shift_y
		self.shift_z = shift_z
def get_config(is_pos, angles_xy = 30,angles_xz = 30, big_cubic_len = 19, sml_cubi_len = 11):
	if is_pos:
		angles_xy_range = [ i / 180 * np.pi for i in range(angles_xy - ANGLE_RANGE//2, angles_xy + ANGLE_RANGE//2)]
		angles_xz_range = [ i / 180 * np.pi for i in range(angles_xz - ANGLE_RANGE//2, angles_xz + ANGLE_RANGE//2)]
		shift_x = [( big_cubic_len - 1 ) // 2]
		shift_y = [( big_cubic_len - 1 ) // 2]
		shift_z = [( big_cubic_len - 1 ) // 2]
	else:
		angles_xy_range = \
			[i / 180 * np.pi for i in range(10,angles_xy - ANGLE_RANGE//4)] + \
			[i / 180 * np.pi for i in range(angles_xy + ANGLE_RANGE//4, 200)]
		angles_xz_range = \
			[i / 180 * np.pi for i in range(10,angles_xz - ANGLE_RANGE//4)] + \
			[i / 180 * np.pi for i in range(angles_xz + ANGLE_RANGE//4, 200)]
		shift_x = [i for i in range(( big_cubic_len - 1 ) // 2 - sml_cubi_len, 
									( big_cubic_len - 1 ) // 2 + sml_cubi_len)]
		shift_y = [i for i in range(( big_cubic_len - 1 ) // 2 - sml_cubi_len, 
									( big_cubic_len - 1 ) // 2 + sml_cubi_len)]
		shift_z = [i for i in range(( big_cubic_len - 1 ) // 2 - sml_cubi_len, 
									( big_cubic_len - 1 ) // 2 + sml_cubi_len)]
	return BatchGenConfig(angles_xy_range, angles_xz_range, shift_x, shift_y, shift_z)

def generate_batch(batch_size = 128,big_cubic_len = 19, sml_cubi_len = 11,
				   pos_percentage = 0.2, target_angles_xy = 30,target_angles_xz = 30,
				   noise_mean = 0, noise_std = 0.1,
				   brightness_min = 0.2, brightness_max = 0.8):
	#line_indexes = np.array([[x - 9 for x in range(big_cubic_len)],
	#						   [0 for y in range(big_cubic_len)],
	#						   [0 for z in range(big_cubic_len)]]) * 5

	line_indexes_1 = [[x - (big_cubic_len-1)//2,0,0] for x in range(big_cubic_len)] * 5
	line_indexes_3 = [[x - (big_cubic_len-1)//2,0,0] for x in range(big_cubic_len)] + \
					 [[x - (big_cubic_len-1)//2,-1,0] for x in range(big_cubic_len)] + \
					 [[x - (big_cubic_len-1)//2,1,0] for x in range(big_cubic_len)] + \
					 [[x - (big_cubic_len-1)//2,0,1] for x in range(big_cubic_len)] + \
					 [[x - (big_cubic_len-1)//2,0,-1] for x in range(big_cubic_len)]
	pos_config = get_config(1,target_angles_xy,target_angles_xz, big_cubic_len, sml_cubi_len)
	neg_config = get_config(0,target_angles_xy,target_angles_xz, big_cubic_len, sml_cubi_len)

	out3D = np.zeros((batch_size,big_cubic_len,big_cubic_len,big_cubic_len))
	label = np.zeros((batch_size,2))

	line_indexes = [None] * batch_size
	angles_xy = [None] * batch_size
	angles_xz = [None] * batch_size
	shift_x = [None] * batch_size
	shift_y = [None] * batch_size
	shift_z = [None] * batch_size
	for i in range(batch_size):
		if np.random.random() <  pos_percentage:
			config = pos_config
			shift_x[i] = np.random.choice(config.shift_x)
			shift_y[i] = np.random.choice(config.shift_y)
			shift_z[i] = np.random.choice(config.shift_z)
			label[i,1] = 1
		else:
			if np.random.random() < 0.75:
				if np.random.random() < 0.33:
					config = neg_config
					shift_x[i] = ( big_cubic_len - 1 ) // 2 + int(np.random.laplace(0,2))
					shift_y[i] = ( big_cubic_len - 1 ) // 2 + int(np.random.laplace(0,2))
					shift_z[i] = ( big_cubic_len - 1 ) // 2 + int(np.random.laplace(0,2))
					label[i,0] = 1
				else:
					config = neg_config
					shift_x[i] = ( big_cubic_len - 1 ) // 2 + 100#int(np.random.laplace(0,.5))
					shift_y[i] = ( big_cubic_len - 1 ) // 2 + 100#int(np.random.laplace(0,.5))
					shift_z[i] = ( big_cubic_len - 1 ) // 2 + 100#int(np.random.laplace(0,.5))
					label[i,0] = 1
			else:
				config = pos_config
				dx = np.random.laplace(0,2)
				dy = np.random.laplace(0,2)
				dz = np.random.laplace(0,2)
				shift_x[i] = ( big_cubic_len - 1 ) // 2 + int(dx + np.sign(dx))
				shift_y[i] = ( big_cubic_len - 1 ) // 2 + int(dy + np.sign(dy))
				shift_z[i] = ( big_cubic_len - 1 ) // 2 + int(dz + np.sign(dz))
				label[i,0] = 2

		angles_xy[i] = np.random.choice(config.angles_xy_range)
		angles_xz[i] = np.random.choice(config.angles_xz_range)
		#shift_x[i] = np.random.choice(config.shift_x)
		#shift_y[i] = np.random.choice(config.shift_y)
		#shift_z[i] = np.random.choice(config.shift_z)


	noise = np.random.normal(noise_mean,noise_std,(batch_size,big_cubic_len,big_cubic_len,big_cubic_len))
	brightness = np.random.random(batch_size) * (brightness_max - brightness_min) + brightness_min

	tmp_rot = [np.array(
			   [[np.cos(angles_xy[idx]) * np.cos(angles_xz[idx]),
				 -np.sin(angles_xy[idx]),
				 np.cos(angles_xy[idx]) * np.sin(angles_xz[idx])
				],
				[np.sin(angles_xy[idx]) * np.cos(angles_xz[idx]),
				 np.cos(angles_xy[idx]),
				 np.sin(angles_xy[idx]) * np.sin(angles_xz[idx])
				],
				[-np.sin(angles_xz[idx]),
				 0,
				 np.cos(angles_xz[idx])
				]
			   ]) for idx in range(batch_size)]
	tmp_rot = np.concatenate(tmp_rot,axis=1)
	result_1 = np.int_(np.dot(line_indexes_1,tmp_rot))
	result_3 = np.int_(np.dot(line_indexes_3,tmp_rot))
	inner_shift_x = 0
	inner_shift_y = 0
	inner_shift_z = 0
	for i in range(batch_size):
		if np.random.random() > 0.5:
			result = result_1
		else:
			result = result_3
			xyz_random = np.random.random()
			if xyz_random < 0.25:
				if np.random.random() > 0.5:
					inner_shift_x = 1
				else:
					inner_shift_x = -1
			elif xyz_random < 0.5:
				if np.random.random() > 0.5:
					inner_shift_y = 1
				else:
					inner_shift_y = -1
			elif xyz_random < .75:
				if np.random.random() > 0.5:
					inner_shift_z = 1
				else:
					inner_shift_z = -1
			else:
				pass

		curr_xs = ( result[:,3*i+0] + shift_x[i] + inner_shift_x)
		valid_x_idxes = np.bitwise_and(curr_xs >= 0, curr_xs < big_cubic_len)
		curr_ys = ( result[:,3*i+1] + shift_y[i] + inner_shift_y)
		valid_y_idxes = np.bitwise_and(curr_ys >= 0, curr_ys < big_cubic_len)
		curr_zs = ( result[:,3*i+2] + shift_z[i] + inner_shift_z)
		valid_z_idxes = np.bitwise_and(curr_zs >= 0, curr_zs < big_cubic_len)
		valid_idxes = np.bitwise_and(np.bitwise_and(valid_x_idxes,valid_y_idxes),valid_z_idxes)
		#print curr_xs[valid_idxes]
		#print result[:,3*i+0]
		#print shift_x[i]
		out3D[i,curr_xs[valid_idxes],curr_ys[valid_idxes], curr_zs[valid_idxes]] = 1
		if label[i,0] == 2:
			if out3D[i,( big_cubic_len - 1 ) // 2, ( big_cubic_len - 1 ) // 2, ( big_cubic_len - 1 ) // 2] == 1:
				label[i,1] = 1
				label[i,0] = 0
			else:
				label[i,0] = 1
		if np.random.random() > 0.5 :
			out3D[i,( big_cubic_len - 1 ) // 2,( big_cubic_len - 1 ) // 2,( big_cubic_len - 1 ) // 2] = 0
	noised_3D = out3D + noise
	for i in range(batch_size):
		noised_3D[i,:,:,:] = noised_3D[i,:,:,:] * brightness[i]
	crop_offset = (big_cubic_len - sml_cubi_len) // 2
	data = noised_3D[:,crop_offset:-crop_offset,\
					   crop_offset:-crop_offset,\
					   crop_offset:-crop_offset]
	data[data > 1] = 1
	data[data < 0] = 0
	return data,label

def generate_batch2D(batch_size = 128,big_cubic_len = 19, sml_cubi_len = 11,
				   pos_percentage = 0.2, target_angles_xy = 30,target_angles_xz = 30,
				   noise_mean = 0, noise_std = 0.1,
				   brightness_min = 0.2, brightness_max = 0.8):
	data,label = generate_batch(batch_size, big_cubic_len, sml_cubi_len,
				   pos_percentage, target_angles_xy,target_angles_xz,
				   noise_mean, noise_std,
				   brightness_min, brightness_max)
	data = np.max(data,axis = 3)
	return data,label


if __name__ == '__main__':
	d,l = generate_batch2D(8,pos_percentage = 0.0,target_angles_xy = 30,target_angles_xz = 0)
	for i in range(8):
		print(d[i,:,:].shape)
		misc.imsave(str(i)+'.png',d[i,:,:])



	