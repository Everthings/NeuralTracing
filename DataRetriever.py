# -*- coding: utf-8 -*-
"""
Created on Fri May 10 22:44:38 2019

@author: andyx
"""

from PIL import Image
import numpy as np

class DataRetriever():
    
    def getData(self, folder_path, image_num):

        swc_mat = np.asarray(Image.open(folder_path + "image_" + str(image_num) + "_swc.png")).astype(np.bool)
        swc_mat_2 =  ~(swc_mat).astype(np.bool)

        #swc_mat = np.expand_dims(swc_mat, axis = 2)
        swc_mat_2 = np.expand_dims(swc_mat_2, axis = 2)
        
        #swc_mat = np.concatenate((swc_mat, swc_mat_2), axis = 2)
        
        # 3 images need to be combined to form tif_mat
        tif_mat = np.asarray(Image.open(folder_path + "image_" + str(image_num) + "_tif_1.png"))
        tif_mat2 = np.asarray(Image.open(folder_path + "image_" + str(image_num) + "_tif_2.png"))
        tif_mat3 = np.asarray(Image.open(folder_path + "image_" + str(image_num) + "_tif_3.png"))
        
        tif_mat = np.concatenate((tif_mat, tif_mat2), axis = 2)
        tif_mat = np.concatenate((tif_mat, tif_mat3), axis = 2)
        
        return swc_mat, tif_mat
        