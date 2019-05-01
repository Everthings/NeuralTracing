import numpy as np
from scipy.optimize import minimize


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

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


class Intensity_Depth:
    # use l2, return float
    TIFF = None

    def map(self, size, swc_filepath, tiff_filepath):
        self.TIFF = TIFFExtractor().extract(tiff_filepath)
        parent_dict, node_dict = self._generateTree(size, swc_filepath)
        points = self._getNewPoints(parent_dict, node_dict)
        final_matrix = self.distributed_intensity_mat(points)

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


    def _getNewPoints(self, parent_dict, node_dict):

        points = []

        for key in parent_dict.keys():

            if (key == -1): continue

            parent = node_dict[key]

            for child in parent_dict[key]:
                subpoints = self._generate_subpoints(Point(parent.x, parent.y), Point(child.x, child.y))
                for point in subpoints:
                    intensities = self._get_intensities(point)
                    new_z = self._calc_z(intensities)
                    points.append(Point(point.x, point.z, new_z))

        return points


    def _get_intensities(self, p):
        return self.TIFF[:][p.x][p.y]


    def _calc_z(self, intensities):
        return np.dot([0, 1, 2, 3, 4, 5, 6, 7], intensities)/sum(intensities)


    def _generate_subpoints(self, p1, p2):
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



    def distributed_intensity_mat(self, points):
        mapped_intensity = 255
        mat = np.zeros(8, 1024, 1024)
        for p in points:
            lower_bound = p.z//1
            res = p.z - lower_bound
            if (res == 0):
                mat[lower_bound][p.x][p.y] = mapped_intensity
            else:
                inverse_down = 1 / res
                inverse_up = 1 / (1 - res)
                normalizer = mapped_intensity / (inverse_up + inverse_down)
                mat[lower_bound][p.x][p.y] = inverse_down * normalizer
                mat[lower_bound+1][p.x][p.y] = inverse_up * normalizer
        return mat


def main():
    import imageio
    import os
    dirpath = os.getcwd()
    print("current directory is:" + dirpath)

    mat = Intensity_Depth().map((10, 1024, 1024), "neuron-data/data1_label.swc", "neuron-data/data1_input.tif")

    for i in range(8):
        imageio.imwrite("test" + str(i) + ".png", mat[i, :, :])



main()