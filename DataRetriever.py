# -*- coding: utf-8 -*-
"""
Created on Fri May 10 22:44:38 2019

@author: andyx
"""

from PIL import Image
import numpy as np



class DataRetriever():
    
    def getData(self, folder_path, image_num):
        
        swc_mat = np.asarray(Image.open(folder_path + "image_" + str(image_num) + "_swc.png"))
        swc_mat = np.expand_dims(swc_mat, axis = 2)
        
        #swc_mat = np.concatenate((swc_mat, swc_mat_2), axis = 2)
        
        # 3 images need to be combined to form tif_mat
        tif_mat = np.asarray(Image.open(folder_path + "image_" + str(image_num) + "_tif.png"))
        im_size = 128
        tif_mat1 = tif_mat[0:im_size, :, :]
        tif_mat2 = tif_mat[im_size:2*im_size, :, :]
        tif_mat3 = tif_mat[2*im_size:3*im_size, :, :]
        
        tif_mat = np.concatenate((tif_mat1, tif_mat2, tif_mat3), axis = 2)

        return tif_mat, swc_mat