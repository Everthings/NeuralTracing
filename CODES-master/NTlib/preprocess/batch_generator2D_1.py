import numpy as np
class BatchGenConfig(object):
	def __init__(self,angles_xy_range, shift_x, shift_y,line_coords):
		self.angles_xy_range = angles_xy_range
		self.shift_x = shift_x
		self.shift_y = shift_y
		self.line_coords = line_coords

def generate_line_coords(big_cubic_len):
	line_indexes_1 = [[x - (big_cubic_len-1)//2,0] for x in range(big_cubic_len)] * 5
	line_indexes_3 = [[x - (big_cubic_len-1)//2,0] for x in range(big_cubic_len)] * 3 + \
					 [[x - (big_cubic_len-1)//2,-1] for x in range(big_cubic_len)] + \
					 [[x - (big_cubic_len-1)//2,1] for x in range(big_cubic_len)]
	line_indexes_5 = [[x - (big_cubic_len-1)//2,0] for x in range(big_cubic_len)] + \
					 [[x - (big_cubic_len-1)//2,-1] for x in range(big_cubic_len)] + \
					 [[x - (big_cubic_len-1)//2,1] for x in range(big_cubic_len)] + \
					 [[x - (big_cubic_len-1)//2,2] for x in range(big_cubic_len)] + \
					 [[x - (big_cubic_len-1)//2,-2] for x in range(big_cubic_len)]
	return [line_indexes_1,line_indexes_3,line_indexes_5]
	width = np.random.choice([1,3,5])
	if width == 1:
		return line_indexes_1
	elif width == 3:
		return line_indexes_3
	elif width == 5:
		return line_indexes_5

def get_config(is_pos, angles_lb = 0, angles_hb = 179,big_cubic_len = 19, width_r = 0, shift_range = 11,big_cubic_len = 29):
	if is_pos:
		if angles_lb <= angles_hb:
			angles_xy_range = [ i * np.pi / 180 for i in range(angles_lb,angles_hb+1)]
		else:
			angles_xy_range = [i * np.pi / 180 for i in range(angles_lb,180+1)]\
							+ [i * np.pi / 180 for i in range(0,angles_hb+1)]
		shift_x = [i for i in range((big_cubic_len - 1) // 2 - width_r,\
									(big_cubic_len - 1) // 2 + width_r + 1)]
		shift_y = [( big_cubic_len - 1 ) // 2]
		line_coords = generate_line_coords(big_cubic_len)
	else:
		if angles_lb <= angles_hb:
			angles_xy_range = \
				[i * np.pi / 180 for i in range(0,angles_lb)] + \
				[i * np.pi / 180 for i in range(angles_hb+1,180+1)]
		else:
			angles_xy_range = [i * np.pi / 180 for i in range(angles_hb,angles_lb)]

		shift_x = [i for i in range(( big_cubic_len - 1 ) // 2 - shift_range, 
									( big_cubic_len - 1 ) // 2 - width_r)]\
				+ [i for i in range(( big_cubic_len - 1 ) // 2 + width_r + 1, 
									( big_cubic_len - 1 ) // 2 + shift_range + 1)]
		shift_y = [i for i in range(( big_cubic_len - 1 ) // 2 - shift_range, 
									( big_cubic_len - 1 ) // 2 - width_r)]\
				+ [i for i in range(( big_cubic_len - 1 ) // 2 + width_r + 1, 
									( big_cubic_len - 1 ) // 2 + shift_range + 1)]
		line_coords = generate_line_coords(big_cubic_len)
	return BatchGenConfig(angles_xy_range, shift_x, shift_y, line_coords)

def determin_cands(pos_percentage,pos_config,neg_config,line_coords_list,angles_xy_list,shift_x_list,shift_y_list,label):
	for idx in range(len(line_coords_list)):
		if np.random.random() > pos_percentage:
			angles_xy_list[idx] = np.random.choice(neg_config.angles_xy_range)
			shift_x_list[idx] = np.random.choice(neg_config.shift_x)
			shift_y_list[idx] = np.random.choice(neg_config.shift_y)
			line_coords_list[idx] = np.random.choice(neg_config.line_coords)
			label[idx] = 0
		else:
			angles_xy_list[idx] = np.random.choice(pos_config.angles_xy_range)
			shift_x_list[idx] = np.random.choice(pos_config.shift_x)
			shift_y_list[idx] = np.random.choice(pos_config.shift_y)
			line_coords_list[idx] = np.random.choice(pos_config.line_coords)
			label[idx] = 1

def gen_batch_img(out2D,line_coords_list,shift_x_list,shift_y_list,tmp_rot):
	for idx in range(len(angles_xy_list)):
		rot_line_coords = np.dot(line_coords_list[idx],tmp_rot[idx])
		rot_line_coords[:,0] = rot_line_coords[:,0] + shift_x_list[idx]
		rot_line_coords[:,1] = rot_line_coords[:,1] + shift_y_list[idx]
		out2D[rot_line_coords] = 1.0
def generate_batch(batch_size = 128,big_cubic_len = 19, sml_cubi_len = 11,
				   pos_percentage = 0.2, target_angles_xy = 30,
				   noise_mean = 0, noise_std = 0.1,
				   brightness_min = 0.2, brightness_max = 0.8,
				   non_laplace_shift_percentage = 0.75,
				   centerline_flag = False):
	line_coords = generate_line_coords(big_cubic_len)

	out2D = np.zeros((batch_size,big_cubic_len,big_cubic_len))
	label = np.zeros((batch_size,2))

	pos_config = get_config(1,target_angles_xy, big_cubic_len, sml_cubi_len)
	neg_config = get_config(0,target_angles_xy, big_cubic_len, sml_cubi_len)


	line_coords_list = [None] * batch_size
	angles_xy_list = [None] * batch_size
	shift_x_list = [None] * batch_size
	shift_y_list = [None] * batch_size
	determin_cands(pos_percentage,pos_config,neg_config,line_coords_list,angles_xy_list,shift_x_list,shift_y_list)

	tmp_rot = [np.array(
			   [[np.cos(angles_xy[idx]),-np.sin(angles_xy[idx])],
			   	[np.sin(angles_xy[idx]),np.cos(angles_xy[idx])]
			   ]) for idx in range(batch_size)]
	gen_batch_img(out2D,line_coords_list,shift_x_list,shift_y_list,tmp_rot)
	return out2D


if __name__ == '__main__':
