# -*- coding: utf-8 -*-
"""
Created on Thu May  9 13:24:42 2019

@author: andyx
"""

import numpy as np
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
    box_size = 100
    n = 50
    size = (30, 1024, 1024)
    
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3
    
    swc = None
    tiff = None
    
    ds = 1.612
    
    def generateBoxes(self, swc_filepath, tiff_filepath):
        self.swc = SWCExtractor().extract((self.size), swc_filepath)
        self.size = self.swc.shape
        self.swc = np.max(self.swc, axis = 0)
        self.tiff = TIFFExtractor().extract(tiff_filepath)
        parent_dict, node_dict = self._generateTree(self.size, swc_filepath)
        delta = self._gridSearch(swc_filepath, tiff_filepath)
        data = self._getBoxes(parent_dict, node_dict, delta)
        
        return data


    def _generateTree(self, size, swcfile):
        parent_dict = dict()
        node_dict = dict()

        with open(swcfile, 'r') as fin:
            for line in fin:
                if line[0] == "#": continue

                parts = line.split()
                index = int(parts[0])
                x = int(round(float(parts[3])))
                y = int(round(float(parts[2])))
                z = int(round(float(parts[4])))
                parent = int(parts[6])
                node = SWCNode(index, x, y, z, parent)
                node_dict[index] = node
                if parent in parent_dict:
                    parent_dict[parent].append(node)
                else:
                    parent_dict[parent] = [node]

        return parent_dict, node_dict


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
                        box_swc = self._getBox(point, self.swc)
                        box_tiff = []
                        for layer in self.tiff:
                            box_tiff.append(self._getBox(point, layer))
    
                        box_tiff = np.array(box_tiff)
                        box_tiff = box_tiff.astype('uint8')
    
                        datas.append(Data(box_tiff, box_swc))

        return datas


    def _getBox(self, p, mat):
        lower_bound_x = int(max(0, p.x - self.box_size/2))
        lower_bound_y = int(max(0, p.y - self.box_size/2))
        upper_bound_x = int(min(self.size[1], p.x + self.box_size/2))
        upper_bound_y = int(min(self.size[2], p.y + self.box_size/2))
        
        curr_box = mat[lower_bound_x:upper_bound_x, lower_bound_y:upper_bound_y]
        
        box = self._create_valid_box(curr_box, p)
        
        return box
        
    
    def _create_valid_box(self, curr_box, p):
        
        if p.x - self.box_size/2 < 0:
            curr_box = self._padd(curr_box, int(self.box_size/2 - p.x), self.LEFT)
        if p.x + self.box_size/2 > self.size[1]:
            curr_box = self._padd(curr_box, int(p.x + self.box_size/2 - self.size[1]), self.RIGHT)
        if p.y - self.box_size/2 < 0:
            curr_box = self._padd(curr_box, int(self.box_size/2 - p.y), self.UP)
        if p.y + self.box_size/2 > self.size[2]:
            curr_box = self._padd(curr_box, int(p.y + self.box_size/2 - self.size[2]), self.DOWN)
            
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


    def _distributIntensitiesToMat(self, points):
        mapped_intensity = 255
        mat = np.zeros((len(self.TIFF), 1024, 1024), dtype = np.uint8)
        for p in points:
            if p.z == -1:
                continue
            
            lower_bound = (int)(p.z)
            res = p.z - lower_bound
            
            if (res < 0.001):
                mat[lower_bound][p.x][p.y] = mapped_intensity
            elif (res > 0.999):
                mat[lower_bound + 1][p.x][p.y] = mapped_intensity
            else:
                inverse_down = 1 / res
                inverse_up = 1 / (1 - res)
                normalizer = mapped_intensity / (inverse_up + inverse_down)
                mat[lower_bound][p.x][p.y] = (int)(inverse_down * normalizer)
                mat[lower_bound+1][p.x][p.y] = (int)(inverse_up * normalizer)
                
        return mat
    
    
    def _gridSearch(self, swc_filepath, tiff_filepath):
        from skimage import io
        swc = np.loadtxt(swc_filepath, dtype=np.int16)
        tif = io.imread(tiff_filepath)
        #ls = list(tif.iter_images())
        merge = np.max(tif, axis=0)
        result = [0, 0, 0, 0, 0, 0]
        Smax = 0
        Xmax = np.max(swc[:,3])
        Xmin = np.min(swc[:,3])
        Ymax = np.max(swc[:,2])
        Ymin = np.min(swc[:,2])
        for dx in range(0, self.size[1]-int((Xmax-Xmin)*self.ds)):
            for dy in range(0, self.size[2]-int((Ymax-Ymin)*self.ds)):
                Sum = np.sum(merge[((Xmax-swc[:,3])*self.ds+dx).astype(np.int), ((swc[:,2]-Ymin)*self.ds+dy).astype(np.int)])
                if Sum > Smax:
                    result[0] = dx
                    result[1] = dy
                    Smax = Sum
        result[2] = int((Xmax-Xmin)*self.ds+result[0])
        result[3] = int((Ymax-Ymin)*self.ds+result[1])
        result[4] = Xmax
        result[5] = Ymin
        return result


def main():
    from PIL import Image
    import imageio
    
    alphabet = "abcdefghijklmnopqrstuvwxyz!@#$%^&*()"
    
    for j in range(0, 30):
        data = PreprocessSkeleton().generateBoxes("neuron-data/data" + str(j + 1) + "_label.swc", "neuron-data/data" + str(j + 1) + "_input.tif")
        for i in range(len(data)):
            dPoint = data[i]
            imageio.imwrite("subimages/image_" + alphabet[j] + str(i) + "_swc.png", dPoint.output)
            
            depth = dPoint.input.shape[0]
            dPoint.input = np.transpose(dPoint.input, (1, 2, 0))

            if depth == 6:
                im1 = Image.fromarray(dPoint.input[:, :, 0:2])
                im1.save("subimages/image_" + alphabet[j] + str(i) + "_tif_1.png")
                im2 = Image.fromarray(dPoint.input[:, :, 2:4])
                im2.save("subimages/image_" + alphabet[j] + str(i) + "_tif_2.png")
                im3 = Image.fromarray(dPoint.input[:, :, 4:6])
                im3.save("subimages/image_" + alphabet[j] + str(i) + "_tif_3.png")
            elif depth == 7:
                im1 = Image.fromarray(dPoint.input[:, :, 0:3])
                im1.save("subimages/image_" + alphabet[j] + str(i) + "_tif_1.png")
                im2 = Image.fromarray(dPoint.input[:, :, 3:5])
                im2.save("subimages/image_" + alphabet[j] + str(i) + "_tif_2.png")
                im3 = Image.fromarray(dPoint.input[:, :, 5:7])
                im3.save("subimages/image_" + alphabet[j] + str(i) + "_tif_3.png")
            elif depth == 8:
                im1 = Image.fromarray(dPoint.input[:, :, 0:3])
                im1.save("subimages/image_" + alphabet[j] + str(i) + "_tif_1.png")
                im2 = Image.fromarray(dPoint.input[:, :, 3:6])
                im2.save("subimages/image_" + alphabet[j] + str(i) + "_tif_2.png")
                im3 = Image.fromarray(dPoint.input[:, :, 6:8])
                im3.save("subimages/image_" + alphabet[j] + str(i) + "_tif_3.png")
            elif depth == 9:
                im1 = Image.fromarray(dPoint.input[:, :, 0:3])
                im1.save("subimages/image_" + alphabet[j] + str(i) + "_tif_1.png")
                im2 = Image.fromarray(dPoint.input[:, :, 3:6])
                im2.save("subimages/image_" + alphabet[j] + str(i) + "_tif_2.png")
                im3 = Image.fromarray(dPoint.input[:, :, 6:9])
                im3.save("subimages/image_" + alphabet[j] + str(i) + "_tif_3.png")
            else:
                print("Extra Depth in ProcessSkeleton: " + str(depth))           
            
        print("Done: " + str(j))
    

main()

