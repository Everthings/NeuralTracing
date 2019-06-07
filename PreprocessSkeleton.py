# -*- coding: utf-8 -*-
"""
Created on Thu May  9 13:24:42 2019

@author: andyx
"""

import numpy as np
import random as rand
from SWCExtractor import SWCExtractor
from TIFFExtractor import TIFFExtractor
import random as rand

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
    
    def sampleBoxesFromArea(self, swc_filepath, tiff_filepath, area_bounds, num_boxes):
        # area bounds must be stored as [x_min, y_min, x_max, y_max]
        
        boxes = []
        
        self.tiff = TIFFExtractor().extract(tiff_filepath)  
        
        mat = np.zeros((30, 1024, 1024), dtype = np.uint8)
        parent_dict, node_dict = SWCExtractor().generateTree(swc_filepath)
        delta = SWCExtractor().gridSearch(swc_filepath, tiff_filepath)
        SWCExtractor().drawTree(parent_dict, node_dict, mat, delta)
        self.swc = np.max(mat, axis = 0)
        
        for i in range(num_boxes):
            rand_point = Point(int(rand.uniform(area_bounds[0], area_bounds[2])), int(rand.uniform(area_bounds[1], area_bounds[3])))
            box_swc = self._getBox(rand_point, self.swc)
            box_tiff = []
            for layer in self.tiff:
                box_tiff.append(self._getBox(rand_point, layer))

            box_tiff = np.array(box_tiff)
            box_tiff = box_tiff.astype('uint8')

            boxes.append(Data(box_tiff, box_swc))
        
        return boxes
        
    
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
    
    #x1, y1, x2, y2
    artifact_bounds = [[850, 185, 990, 750],
                       [70, 260, 290, 970],
                       [135, 45, 500, 285],
                       [115, 120, 260, 1015],
                       [110, 300, 325, 900],
                       [380, 765, 935, 915],
                       [180, 815, 620, 970],
                       [831, 165, 990, 680],
                       [10, 580, 560, 780],
                       [730, 675, 950, 950],
                       [225, 430, 400, 800],
                       [480, 750, 912, 860],
                       [170, 680, 450, 970],
                       [180, 180, 385, 815],
                       [200, 115, 430, 535],
                       [435, 700, 700, 970],
                       [150, 110, 700, 380],
                       [240, 115, 980, 320],
                       [80, 740, 1000, 960],
                       [725, 665, 1000, 1000],
                       [250, 730, 915, 900],
                       [170, 150, 360, 600],
                       [630, 40, 870, 670],
                       [200, 370, 385, 1025],
                       [250, 225, 400, 700],
                       [215, 205, 375, 760],
                       [180, 775, 900, 965],
                       [175, 170, 460, 1025],
                       [690, 230, 860, 815],
                       [125, 175, 280, 640]]
    
    for j in range(0, 30):
        data1 = processor.generateBoxes("neuron-data/data" + str(j + 1) + "_label.swc", "neuron-data/data" + str(j + 1) + "_input.tif")
        print(len(data1))
        data2 = processor.sampleBoxesFromArea("neuron-data/data" + str(j + 1) + "_label.swc", "neuron-data/data" + str(j + 1) + "_input.tif", artifact_bounds[j], 150)
        print(len(data2))
        data = data1 + data2
        print(len(data))
        for i in range(len(data)):
            dPoint = data[i]
            imageio.imwrite("subimages/image_" + str(counter) + "_swc.png", dPoint.output)
            
            depth = dPoint.input.shape[0]
            dPoint.input = np.transpose(dPoint.input, (1, 2, 0))
            
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
            
        print("Done: " + str(j) + " " + str(counter))
    

main()

