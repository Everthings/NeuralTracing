import numpy as np
from skimage import io
import os
import math

from TIFFExtractor import TIFFExtractor
from SWCExtractor import SWCExtractor

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


class Intensity_Depth:
    # use l2, return float
    TIFF = None
    ds = 1.612
    side = 1024

    def map(self, size, swc_filepath, tiff_filepath):
        self.TIFF = TIFFExtractor().extract(tiff_filepath)
        parent_dict, node_dict = self._generateTree(size, swc_filepath)
        delta = self._gridSearch(swc_filepath)
        points = self._getNewPoints(parent_dict, node_dict, delta)
        final_matrix = self._distributIntensitiesToMat(points)

        return final_matrix


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


    def _getNewPoints(self, parent_dict, node_dict, delta):

        points = []

        for key in parent_dict.keys():

            if (key == -1): continue

            parent = node_dict[key]

            for child in parent_dict[key]:
                subpoints = self._generateSubpoints(Point((delta[4] - parent.x) * self.ds + delta[0], (parent.y - delta[5])*self.ds + delta[1]), Point((delta[4] - child.x) * self.ds + delta[0], (child.y - delta[5])*self.ds + delta[1]))
                for point in subpoints:
                    intensities = self._getIntensities(point)
                    new_z = self._calcZ(intensities)
                    #points.append(Point(point.x, point.z, new_z))
                    point.add_z(new_z)
                    points.append(point)

        return points


    def _getIntensities(self, p):
        return self.TIFF[:, p.x, p.y]


    def _calcZ(self, intensities):
        z = np.dot([0, 1, 2, 3, 4, 5, 6, 7], intensities)/sum(intensities)
        
        if math.isnan(z):
            return -1
        else:
            return z


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
        mat = np.zeros((8, 1024, 1024), dtype = np.uint8)
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
    import imageio
    
    mat = Intensity_Depth().map((10, 1024, 1024), "neuron-data/data1_label.swc", "neuron-data/data1_input.tif")

    for i in range(8):
        imageio.imwrite("test" + str(i) + ".png", mat[i])


main()