import numpy as np
from scipy import misc
from skimage import morphology
from skimage import util
from skimage.draw import line
import random
import pickle
import time
ROOT = '../../'
def drawline_width(idx,	length_low = 12, length_high = 24,
						center_xy = 30, center_xz = 30,
						angle_deviance = 20,
						size = 64,
						gap_range_low = 1,
						gap_range_high = 2,
						sample_range = 11,
						noise_type = 'gaussian',
						noise_var = 0.05,
						intensity_low = 0.2, intensity_high = 0.8):
	#ts = time.time()
	img = np.zeros((size,size,size),dtype = np.uint8)
	mask = np.zeros((size,size,size),dtype = np.uint8)
	sample = np.zeros((size,size,size),dtype = np.uint8)
	#random length
	length = np.random.randint(length_low,length_high)
	#random angle for plane xy and xz
	angle_xy = np.random.randint(center_xy - angle_deviance,
								 center_xy + angle_deviance + 1)
	angle_xz = np.random.randint(center_xz - angle_deviance,
								 center_xz + angle_deviance + 1)
	p1 = np.array([size/2,size/2,size/2])
	p2 = np.array([size/2,size/2,size/2])
	p1[0] = size/2 + np.sin(np.pi * angle_xy / 180) * length
	p1[1] = size/2 + np.cos(np.pi * angle_xy / 180) * length
	p1[2] = size/2 + np.cos(np.pi * angle_xz / 180) * length

	p2[0] = size/2 + np.sin(np.pi * (angle_xy+180) / 180) * length
	p2[1] = size/2 + np.cos(np.pi * (angle_xy+180) / 180) * length
	p2[2] = size/2 + np.cos(np.pi * (angle_xz+180) / 180) * length

	p1 = np.around(p1)
	p2 = np.around(p2)
	img = drawline(p1,p2,img)
	width = np.random.randint(2)
	img = morphology.dilation(img,morphology.ball(width))
	gold = img
	mid_x,mid_y,mid_z = np.around((p1 + p2)/2)
	mask[mid_x,mid_y,mid_z] = 1.0
	gap = width + np.random.randint(gap_range_low, gap_range_high + 1)
	mask = morphology.dilation(mask,morphology.ball(gap))
	sample = np.maximum(morphology.binary_dilation(gold,morphology.ball(sample_range)),
						gold * 2)
	sample = np.maximum(sample,mask*3)
	img = img * (1.0 - mask)
	if img[mid_x,mid_y,mid_z] != 0: print('idx',idx)
	#random intensity
	intensity = ((intensity_high - intensity_low) * np.random.random() + intensity_low)
	print(intensity)
	img = img * intensity
	img = util.random_noise(img,mode=noise_type,mean=0,var=(intensity ** 1.5) * noise_var)
	img = np.uint8(img * 255)
	#te = time.time()
	#print("drawline_width %f\n", te-ts)
	#print(gold.dtype,np.amax(gold))
	gold = gold | (sample << 1)
	return gold,img	
def drawline3D(p1,p2,mat):
	x1,y = line(p1[0],p1[1],p2[0],p2[1])
	x2,z = line(p1[0],p1[2],p2[0],p2[2])
	print(x2)
	print(x2)
def drawline(p1,p2,mat):
	#ts = time.time()
	p1 = np.around(p1)
	p2 = np.around(p2)
	if all(p1 == p2):
		mat[p1[0],p1[1]] = 1
		return mat
	unit_v = (p2-p1)/(np.linalg.norm(p2-p1))
	try:
		for x in np.linspace(p1[0],p2[0],np.abs(p2[0] - p1[0]) + 1):
			if unit_v[0] == 0: break
			diff = x - p1[0]
			y = diff * unit_v[1] / unit_v[0] + p1[1]
			z = diff * unit_v[2] / unit_v[0] + p1[2]
			mat[x,y,z] = 1
		for y in np.linspace(p1[1],p2[1],np.abs(p2[1] - p1[1]) + 1):
			if unit_v[1] == 0 : break
			diff = y - p1[1]
			x = diff * unit_v[0] / unit_v[1] + p1[0]
			z = diff * unit_v[2] / unit_v[1] + p1[2]
			mat[x,y,z] = 1
		for z in np.linspace(p1[2],p2[2],np.abs(p2[2] - p1[2]) + 1):
			if unit_v[2] == 0 : break
			diff = z - p1[2]
			x = diff * unit_v[0] / unit_v[2] + p1[0]
			y = diff * unit_v[1] / unit_v[2] + p1[1]
			mat[x,y,z] = 1
	except:
		print(x,y,z)
	#te = time.time()
	#print("drawline_width %f\n", te-ts)
	return mat

if __name__ == '__main__':
	from sys import argv
	from sys import path
	path.append('../../')
	from NTlib.image_data import ImageData
	import time
	if argv[1] == '0':
		data = [None for i in range(10)]
		for idx in range(10):
			print(idx)
			gold,sample,train = drawline_width(idx)
			data[idx] = [gold,sample,train]
			misc.imsave(ROOT+'data/Sample_3Dx_lines/gold_'+str(idx)+'.png',np.max(gold,2))
			misc.imsave(ROOT+'data/Sample_3D_lines/sample_'+str(idx)+'.png',np.max(sample,2))
			misc.imsave(ROOT+'data/Sample_3D_lines/train_'+str(idx)+'_x.png',np.max(train,0))
			misc.imsave(ROOT+'data/Sample_3D_lines/train_'+str(idx)+'_y.png',np.max(train,1))
			misc.imsave(ROOT+'data/Sample_3D_lines/train_'+str(idx)+'_z.png',np.max(train,2))
	else:
		t1 = time.time()
		number_of_each_sample = int(argv[1])
		data = [None for i in range(number_of_each_sample)]
		for angle_xy in range(30,181,30):
			for angle_xz in range(30,181,30):
				for idx in range(number_of_each_sample):
					gold,sample,train = drawline_width(idx)
					data[idx] = ImageData(gold,sample,train)
		version = argv[2]
		t2 = time.time()
		with open(ROOT+'data/images/image_set_'+version+'.bin','wb') as fout:
			pickle.dump(data,fout)
		t3 = time.time()
		print("Cal %f IO %f\n" % (t2-t1,t3-t2))
