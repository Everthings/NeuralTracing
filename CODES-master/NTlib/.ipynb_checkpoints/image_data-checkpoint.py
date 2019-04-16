import numpy as np
class ImageData(object):
	def __init__(self,gold_sample, image):
		self.image = image
		self.gold_sample = gold_sample & np.array(1,dtype=np.uint8)
		#self.sample = sample
	def get_gold(self):
		return np.bitwise_and(self.gold_sample, np.array(254) )

class AngleImageDataSet(object):
	def __init__(self):
		self.data = dict()

class PatchData(object):
	def __init__(self,patch,label):
		self.patch = patch
		self.label = label