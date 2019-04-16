from NTlib.preprocess.batch_generator import BatchGenConfig,get_config
import tensorflow as tf
import numpy as np
ANGLE_RANGE = 40

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
					shift_x[i] = np.random.choice(config.shift_x)
					shift_y[i] = np.random.choice(config.shift_y)
					shift_z[i] = np.random.choice(config.shift_z)
					label[i,0] = 1
				else:
					config = neg_config
					shift_x[i] = ( big_cubic_len - 1 ) // 2 + 100#int(np.random.laplace(0,.5))
					shift_y[i] = ( big_cubic_len - 1 ) // 2 + 100#int(np.random.laplace(0,.5))
					shift_z[i] = ( big_cubic_len - 1 ) // 2 + 100#int(np.random.laplace(0,.5))
					label[i,0] = 1
			else:
				config = pos_config
				shift_conf = neg_config

				shift_x[i] = np.random.choice(shift_conf.shift_x)
				shift_y[i] = np.random.choice(shift_conf.shift_y)
				shift_z[i] = np.random.choice(shift_conf.shift_z)
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