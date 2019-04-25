# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 13:53:48 2019

@author: andyx
"""



#from libtiff import TIFF
import os
import numpy as np
from skimage import io

class SWCNode():
    def __init__(self, index, x, y, z, parent):
        self.index = index
        self.x = x
        self.y = y
        self.z = z
        self.parent = parent


class SWCExtractor():
    
    side = 1024
    ds = 1.612
    
    def extract(self, size, filePath):
        mat = np.zeros((size), dtype = np.uint8)
        #mat = np.expand_dims(mat, axis = 3)

        parent_dict, node_dict = self._generateTree(size, filePath)
        
        delta = self._gridSearch(filePath)
        
        self._drawTree(parent_dict, node_dict, mat, delta)
        return mat

    def _generateTree(self, size, swcfile):
        parent_dict = dict()
        node_dict = dict()
        
        with open(swcfile,'r') as fin:
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

    def _drawTree(self, parent_dict, node_dict, mat, delta):
        
        for key in parent_dict.keys():
            
            if (key == -1): continue
        
            parent = node_dict[key]

            for child in parent_dict[key]:
                self._drawBranch([(delta[4] - parent.x)*self.ds + delta[0], (parent.y - delta[5])*self.ds + delta[1], parent.z], [(delta[4] - child.x)*self.ds + delta[0], (child.y - delta[5])*self.ds + delta[1], child.z], mat)
                #self._drawBranch([(parent.x - delta[5])*self.ds + delta[0], (delta[4] - parent.y)*self.ds + delta[1], parent.z], [(child.x - delta[5])*self.ds + delta[0], (delta[4] - child.y)*self.ds + delta[1], child.z], mat)

    def _drawBranch(self, p1, p2, mat):

        p1 = np.around(p1)
        p2 = np.around(p2)
        
        # stores data at (z, y, x, 0) so that matrix is (cross-section, X, Y, classes)
        # x and -y are swappped to rotate image so it aligns with the .tif (rotates 90 counterclockwise)
        
        if all(p1 == p2):
            mat[int(p1[2]), int(p1[0]), int(p1[1])] = 255
            return mat
        
        unit_v = (p2-p1)/(np.linalg.norm(p2-p1))
        try:
            for x in np.linspace(int(p1[0]), int(p2[0]), int(np.abs(p2[0] - p1[0]) + 1)):
                if unit_v[0] == 0: break
                diff = x - p1[0]
                y = diff * unit_v[1] / unit_v[0] + p1[1]
                z = diff * unit_v[2] / unit_v[0] + p1[2]
                mat[int(round(z)), int(round(x)), int(round(y))] = 255
            
            for y in np.linspace(int(p1[1]), int(p2[1]), int(np.abs(p2[1] - p1[1]) + 1)):
                if unit_v[1] == 0 : break
                diff = y - p1[1]
                x = diff * unit_v[0] / unit_v[1] + p1[0]
                z = diff * unit_v[2] / unit_v[1] + p1[2]
                mat[int(round(z)), int(round(x)), int(round(y))] = 255

            for z in np.linspace(int(p1[2]), int(p2[2]), int(np.abs(p2[2] - p1[2]) + 1)):
                if unit_v[2] == 0 : break
                diff = z - p1[2]
                x = diff * unit_v[0] / unit_v[2] + p1[0]
                y = diff * unit_v[1] / unit_v[2] + p1[1]
                mat[int(round(z)), int(round(x)), int(round(y))] = 255

        except Exception as e:
            print(str(e))

        return mat
    
    def _gridSearch(self, fname):
        swc = np.loadtxt(fname, dtype=np.int16)
        tif = io.imread(os.path.splitext(fname)[0].split("_")[0] + '_input.tif')
        #ls = list(tif.iter_images())
        merge = np.max(tif, axis=0)
        result = [0, 0, 0, 0, 0, 0]
        Smax = 0
        Xmax = np.max(swc[:,3])
        Xmin = np.min(swc[:,3])
        Ymax = np.max(swc[:,2])
        Ymin = np.min(swc[:,2])
        for dx in range(0, self.side-int((Xmax-Xmin)*self.ds)):
            for dy in range(0, self.side-int((Ymax-Ymin)*self.ds)):
                Sum = np.sum(merge[((Xmax-swc[:,3])*self.ds+dx).astype(np.int),((swc[:,2]-Ymin)*self.ds+dy).astype(np.int)])
                if Sum > Smax:
                    result[0] = dx
                    result[1] = dy
                    Smax = Sum
        result[2] = int((Xmax-Xmin)*self.ds+result[0])
        result[3] = int((Ymax-Ymin)*self.ds+result[1])
        result[4] = Xmax
        result[5] = Ymin
        return result


if __name__ == '__main__':
    import imageio
    mat = SWCExtractor().extract((10, 1024, 1024), "neuron-data/data1_label.swc")
    print("SWC Extracted." + str(mat.shape))
    
    #imageio.imwrite("test.png", np.max(mat.T, axis = 2))
    
    #for i in range(0, 10):
    #    print((mat.T[:, :, i].shape))
    #    imageio.imwrite("test" + str(i) + ".png", mat.T[:, :, i])
    
    # imageio.imwrite('test.png', np.max(mat.T, axis = 2))
    # print("SWC Image Drawn!")
    # for i in range(0, 401, 10):
    #    imageio.imwrite("test" + str(i) + ".png", mat.T[i, :, :])
