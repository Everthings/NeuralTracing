# -*- coding: utf-8 -*-
"""
Created on Thu May  9 13:24:42 2019

@author: andyx
"""

import numpy as np
import random as rand
from SWCExtractor import SWCExtractor
from TIFFExtractor import TIFFExtractor

class SWCNode():
    def __init__(self, index, x, y, z, parent):
        self.index = index
        self.x = x
        self.y = y
        self.z = z
        self.parent = parent

class Point():
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def add_z(self, z):
        self.z = z
        
class Data():
    def __init__(self, tiff, swc):
        self.input = tiff
        self.output = swc
        

class PreprocessSkeleton():
    box_size = 128
    n = 50
    size = (1024, 1024)
    
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3
    
    swc = None
    tiff = None
    
    ds = 1.612
    
    def generateBoxes(self, swc_filepath, tiff_filepath):
        
        mat = np.zeros((30, 1024, 1024), dtype = np.uint8)
        self.tiff = TIFFExtractor().extract(tiff_filepath)  
        parent_dict, node_dict = SWCExtractor().generateTree(swc_filepath)
        delta = SWCExtractor().gridSearch(swc_filepath, tiff_filepath)
        SWCExtractor().drawTree(parent_dict, node_dict, mat, delta)
        self.swc = np.max(mat, axis = 0)
        data = self._getBoxes(parent_dict, node_dict, delta)
        
        return data


    def _getBoxes(self, parent_dict, node_dict, delta):

        datas = []
        counter = 0

        for key in parent_dict.keys():

            if (key == -1): continue

            parent = node_dict[key]

            for child in parent_dict[key]:
                subpoints = self._generateSubpoints(Point((delta[4] - parent.x) * self.ds + delta[0], (parent.y - delta[5])*self.ds + delta[1]), Point((delta[4] - child.x) * self.ds + delta[0], (child.y - delta[5])*self.ds + delta[1]))
                for point in subpoints:
                    
                    counter += 1
                    
                    if counter % self.n == 0:
                        
                        rand_offset_x = int((rand.random()-0.5)*self.box_size)
                        rand_offset_y = int((rand.random()-0.5)*self.box_size)
                        point = Point(point.x + rand_offset_x, point.y + rand_offset_y)
                        
                        box_swc = self._getBox(point, self.swc)
                        box_tiff = []
                        for layer in self.tiff:
                            box_tiff.append(self._getBox(point, layer))
    
                        box_tiff = np.array(box_tiff)
                        box_tiff = box_tiff.astype('uint8')
    
                        datas.append(Data(box_tiff, box_swc))

        return datas


    def _getBox(self, p, mat):
        # rand offset to vary training data so not all training data is centered on a line
        lower_bound_x = int(max(0, p.x - self.box_size/2))
        lower_bound_y = int(max(0, p.y - self.box_size/2))
        upper_bound_x = int(min(self.size[0], p.x + self.box_size/2))
        upper_bound_y = int(min(self.size[1], p.y + self.box_size/2))
        
        curr_box = mat[lower_bound_x:upper_bound_x, lower_bound_y:upper_bound_y]
        
        box = self._create_valid_box(curr_box, p)
        
        return box
        
    
    def _create_valid_box(self, curr_box, p):
        
        if p.x - self.box_size/2 < 0:
            curr_box = self._padd(curr_box, int(self.box_size/2 - p.x), self.LEFT)
        if p.x + self.box_size/2 > self.size[0]:
            curr_box = self._padd(curr_box, int(p.x + self.box_size/2 - self.size[0]), self.RIGHT)
        if p.y - self.box_size/2 < 0:
            curr_box = self._padd(curr_box, int(self.box_size/2 - p.y), self.UP)
        if p.y + self.box_size/2 > self.size[1]:
            curr_box = self._padd(curr_box, int(p.y + self.box_size/2 - self.size[1]), self.DOWN)
            
        return curr_box
          
    
    def _padd(self, curr_box, padd_size, direction):
        if direction == self.LEFT:
            curr_box = np.pad(curr_box, [(padd_size, 0), (0, 0)], "constant")
        elif direction == self.RIGHT:
            curr_box = np.pad(curr_box, [(0, padd_size), (0, 0)], "constant")
        elif direction == self.UP:
            curr_box = np.pad(curr_box, [(0, 0), (padd_size, 0)], "constant")
        elif direction == self.DOWN:
            curr_box = np.pad(curr_box, [(0, 0), (0, padd_size)], "constant")
            
        return curr_box
        
    def _generateSubpoints(self, point1, point2):
        
        p1 = [point1.x, point1.y]
        p2 = [point2.x, point2.y]
        
        p1 = np.around(p1)
        p2 = np.around(p2)

        # stores data at (z, y, x, 0) so that matrix is (cross-section, X, Y, classes)
        # x and -y are swappped to rotate image so it aligns with the .tif (rotates 90 counterclockwise)

        if all(p1 == p2):
            return []

        point_array = []
        unit_v = (p2 - p1) / (np.linalg.norm(p2 - p1))
        try:
            for x in np.linspace(int(p1[0]), int(p2[0]), int(np.abs(p2[0] - p1[0]) + 1)):
                if unit_v[0] == 0: break
                diff = x - p1[0]
                y = diff * unit_v[1] / unit_v[0] + p1[1]
                point_array.append(Point(int(round(x)), int(round(y))))
            for y in np.linspace(int(p1[1]), int(p2[1]), int(np.abs(p2[1] - p1[1]) + 1)):
                if unit_v[1] == 0: break
                diff = y - p1[1]
                x = diff * unit_v[0] / unit_v[1] + p1[0]
                point_array.append(Point(int(round(x)), int(round(y))))
        except Exception as e:
            print(str(e))

        return point_array
    

def main():
    from PIL import Image
    import imageio
    
    counter = 1
    
    processor = PreprocessSkeleton()
    
    for j in range(0, 30):
        data = processor.generateBoxes("neuron-data/data" + str(j + 1) + "_label.swc", "neuron-data/data" + str(j + 1) + "_input.tif")
        for i in range(len(data)):
            dPoint = data[i]
            imageio.imwrite("subimages/image_" + str(counter) + "_swc.png", dPoint.output)
            
            depth = dPoint.input.shape[0]
            dPoint.input = np.transpose(dPoint.input, (1, 2, 0))
            
            
            """
            #all images will have a depth of 9 to keep input through UNet constant
            if depth == 6:
                im1 = Image.fromarray(dPoint.input[:, :, 0:3])
                im1.save("subimages/image_" + str(counter) + "_tif_1.png")
                im2 = Image.fromarray(dPoint.input[:, :, 3:6])
                im2.save("subimages/image_" + str(counter) + "_tif_2.png")
                im3 = Image.fromarray(np.zeros((processor.box_size, processor.box_size, 3), np.uint8))
                im3.save("subimages/image_" + str(counter) + "_tif_3.png")
            elif depth == 7:
                im1 = Image.fromarray(dPoint.input[:, :, 0:3])
                im1.save("subimages/image_" + str(counter) + "_tif_1.png")
                im2 = Image.fromarray(dPoint.input[:, :, 3:6])
                im2.save("subimages/image_" + str(counter) + "_tif_2.png")
                new_layer_3 = np.concatenate((dPoint.input[:, :, 6:7], np.zeros((processor.box_size, processor.box_size, 2), np.uint8)), axis = 2)
                im3 = Image.fromarray(new_layer_3)
                im3.save("subimages/image_" + str(counter) + "_tif_3.png")
            elif depth == 8:
                im1 = Image.fromarray(dPoint.input[:, :, 0:3])
                im1.save("subimages/image_" + str(counter) + "_tif_1.png")
                im2 = Image.fromarray(dPoint.input[:, :, 3:6])
                im2.save("subimages/image_" + str(counter) + "_tif_2.png")
                new_layer_3 = np.concatenate((dPoint.input[:, :, 6:8], np.zeros((processor.box_size, processor.box_size, 1), np.uint8)), axis = 2)
                im3 = Image.fromarray(new_layer_3)
                im3.save("subimages/image_" + str(counter) + "_tif_3.png")
            elif depth == 9:
                im1 = Image.fromarray(dPoint.input[:, :, 0:3])
                im1.save("subimages/image_" + str(counter) + "_tif_1.png")
                im2 = Image.fromarray(dPoint.input[:, :, 3:6])
                im2.save("subimages/image_" + str(counter) + "_tif_2.png")
                im3 = Image.fromarray(dPoint.input[:, :, 6:9])
                im3.save("subimages/image_" + str(counter) + "_tif_3.png")
            else:
                print("Extra Depth in ProcessSkeleton: " + str(depth))
            """
            
            mat1 = None
            mat2 = None
            mat3 = None
            
            #all images will have a depth of 9 to keep input through UNet constant
            if depth == 6:
                mat1 = dPoint.input[:, :, 0:3]
                mat2 = dPoint.input[:, :, 3:6]
                mat3 = np.zeros((processor.box_size, processor.box_size, 3), np.uint8)
            elif depth == 7:
                mat1 = dPoint.input[:, :, 0:3]
                mat2 = dPoint.input[:, :, 3:6]
                mat3 = np.concatenate((dPoint.input[:, :, 6:7], np.zeros((processor.box_size, processor.box_size, 2), np.uint8)), axis = 2)
            elif depth == 8:
                mat1 = dPoint.input[:, :, 0:3]
                mat2 = dPoint.input[:, :, 3:6]
                mat3 = np.concatenate((dPoint.input[:, :, 6:8], np.zeros((processor.box_size, processor.box_size, 1), np.uint8)), axis = 2)
            elif depth == 9:
                mat1 = dPoint.input[:, :, 0:3]
                mat2 = dPoint.input[:, :, 3:6]
                mat3 = dPoint.input[:, :, 6:9]
            else:
                print("Extra Depth in ProcessSkeleton: " + str(depth))
            
            mat = np.concatenate((mat1, mat2, mat3), axis = 0)
            im = Image.fromarray(mat)
            im.save("subimages/image_" + str(counter) + "_tif.png")
            
            counter += 1
            
        print("Done: " + str(j))
        
    

main()

