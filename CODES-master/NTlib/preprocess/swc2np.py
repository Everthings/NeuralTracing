import numpy as np
import glob
from scipy import ndimage
from skimage import morphology
class SWCNode():
	def __init__(self, id,x,y,z,p):
		self.id = id
		self.x = x
		self.y = y
		self.z = z
		self.parent = p
		
class SWCReader():
	def swc2mat(self, size, swcfile):
		mat = np.zeros(size,dtype = np.uint8)
		"""
		with open(swcfile,'r') as fin:
			for line in fin:
				if line[0] == '#':
					continue
				parts = line.split()
				x = int(round(float(parts[2])))
				y = int(round(float(parts[3])))
				mat[x,y] = 1
		"""
		parent_dict,node_dict = self.swc2tree(size,swcfile)
		self.draw_tree(parent_dict,node_dict,mat)
		
		return mat
	def swc2tree(self,size,swcfile):
		parent_dict = dict()
		node_dict = dict()
		with open(swcfile,'r') as fin:
			for line in fin:
				if line[0] == "#":
					continue
				parts = line.split()
				id = int(parts[0])
				x = int(round(float(parts[2])))
				y = int(round(float(parts[3])))
				z = int(round(float(parts[4])))
				parent = int(parts[6])
				node = SWCNode(id,x,y,z,parent)
				node_dict[id] = node
				if parent in parent_dict:
					parent_dict[parent].append(node)
				else:
					parent_dict[parent] = [node]
		return parent_dict,node_dict

	def draw_tree(self,parent_dict,node_dict,mat):
		for key in parent_dict.keys():
			if (key == -1): continue
			parent = node_dict[key]

			for child in parent_dict[key]:
				self.drawline([parent.x,parent.y,parent.z],[child.x,child.y,child.z],mat)
	def drawline(self,p1,p2,mat):
		#ts = time.time()
		fp1 = p1
		fp2 = p2
		p1 = np.around(p1)
		p2 = np.around(p2)
		if all(p1 == p2):
			mat[p1[0],p1[1],p1[2]] = 1
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
	def generate_sample(self,mat,disk_size = 11):
		sample = np.maximum(morphology.binary_dilation(mat,morphology.ball(disk_size)),
						mat * 2)
		return sample

if __name__ == '__main__':
	from scipy import misc
	#mat = SWCReader().swc2mat((864,840),'/Users/zheng/documents/projects/tensor_tut/neurons/checked7_janelia_flylight_part1/err_GMR_57C10_AD_01-1xLwt_attp40_4stop1-f-A01-20110325_3_A1-right_optic_lobe.v3draw.extract_5/GMR_57C10_AD_01-1xLwt_attp40_4stop1-f-A01-20110325_3_A1-right_optic_lobe.v3draw.extract_5.v3dpbd_stamp_2015_06_15_18_45.swc')
	mat = SWCReader().swc2mat((511,511,401),'../../data/neuron/train/GMR_57C10_AD_01-1xLwt_attp40_4stop1-f-A01-20110325_3_A6-left_optic_lobe.v3draw.extract_2.v3dpbd.ano_stamp_2015_06_16_18_18.swc')
	print(np.sum(mat))
	#sample = SWCReader().generate_sample(mat)
	misc.imsave('tt.png',np.max(mat.T,axis = 0))
	misc.imsave('tt1.png',np.max(mat.T,axis = 1))
	misc.imsave('tt2.png',np.max(mat.T,axis = 2))
	"""
	misc.imsave('tts.png',np.max(sample.T,axis = 0))
	misc.imsave('tts1.png',np.max(sample.T,axis = 1))
	misc.imsave('tts2.png',np.max(sample.T,axis = 2))
	"""
	"""
	mat = SWCReader().swc2mat((2715,4011),'/Users/zheng/documents/projects/tensor_tut/neurons/1201_01_s06b_L36_Sum_ch2/1201_01_s06b_L36_Sum_ch2.tif.v3dpbd_stamp_2015_06_16_13_27.swc')
	misc.imsave('tt1.png',mat.T)
	swcfiles = glob.glob('./neuron_data/*.swc')
	for swcfile in swcfiles:
		img_name = swcfile.split('v3dpbd')[0] + 'v3dpbd.png'
		origin_img = misc.imread(img_name)
		mat = SWCReader().swc2mat(origin_img.shape,swcfile)
		sample = SWCReader().generate_sample(mat)
		misc.imsave(img_name[:-4]+'_gold.png',mat.T)
		misc.imsave(img_name[:-4]+'_sample.png',sample.T)
	"""
