import numpy as np
from skimage import morphology


class SWCNode():
    def __init__(self, index, x, y, z, parent):
        self.index = index
        self.x = x
        self.y = y
        self.z = z
        self.parent = parent


class SWCExtractor():
    def extract(self, size, filePath):
        mat = np.zeros((size), dtype = np.uint8)
        mat = np.expand_dims(mat, axis = 3)

        parent_dict, node_dict = self._generateTree(size, filePath)
        self._drawTree(parent_dict, node_dict, mat)
        return mat

    def _generateTree(self, size, swcfile):
        parent_dict = dict()
        node_dict = dict()
        
        with open(swcfile,'r') as fin:
            for line in fin:
                if line[0] == "#": continue
                
                parts = line.split()
                index = int(parts[0])
                x = int(round(float(parts[2])))
                y = int(round(float(parts[3])))
                z = int(round(float(parts[4])))
                parent = int(parts[6])
                node = SWCNode(index,x,y,z,parent)
                node_dict[index] = node
                if parent in parent_dict:
                    parent_dict[parent].append(node)
                else:
                    parent_dict[parent] = [node]
                    
        return parent_dict, node_dict

    def _drawTree(self, parent_dict, node_dict, mat):
        for key in parent_dict.keys():
            
            if (key == -1): continue
        
            parent = node_dict[key]

            for child in parent_dict[key]:
                self._drawBranch([parent.x,parent.y,parent.z], [child.x,child.y,child.z], mat)

    def _drawBranch(self, p1, p2, mat):

        p1 = np.around(p1)
        p2 = np.around(p2)
        
        # stores data at (z, y, x, 0) so that matrix is (cross-section, X, Y, classes)
        # x and -y are swappped to rotate image so it aligns with the .tif (rotates 90 counterclockwise)
        
        if all(p1 == p2):
            mat[p1[2], -p1[1], p1[0], 0] = 1
            return mat
        
        unit_v = (p2-p1)/(np.linalg.norm(p2-p1))
        try:
            for x in np.linspace(p1[0], p2[0], np.abs(p2[0] - p1[0]) + 1):
                if unit_v[0] == 0: break
                diff = x - p1[0]
                y = diff * unit_v[1] / unit_v[0] + p1[1]
                z = diff * unit_v[2] / unit_v[0] + p1[2]
                mat[int(round(z)), -int(round(y)), int(round(x)), 0] = 1
            
            for y in np.linspace(p1[1],p2[1],np.abs(p2[1] - p1[1]) + 1):
                if unit_v[1] == 0 : break
                diff = y - p1[1]
                x = diff * unit_v[0] / unit_v[1] + p1[0]
                z = diff * unit_v[2] / unit_v[1] + p1[2]
                mat[int(round(z)), -int(round(y)), int(round(x)), 0] = 1

            for z in np.linspace(p1[2],p2[2],np.abs(p2[2] - p1[2]) + 1):
                if unit_v[2] == 0 : break
                diff = z - p1[2]
                x = diff * unit_v[0] / unit_v[2] + p1[0]
                y = diff * unit_v[1] / unit_v[2] + p1[1]
                mat[int(round(z)), -int(round(y)), int(round(x)), 0] = 1

        except Exception as e:
            print(str(e))

        return mat


if __name__ == '__main__':
    mat = SWCExtractor().extract((401, 511, 511), "neuron_data/data1_label.swc")
    print(mat.shape)
    
    # imageio.imwrite('test.png', np.max(mat.T, axis = 2))
    # print("SWC Image Drawn!")
    # for i in range(0, 401, 10):
    #    imageio.imwrite("test" + str(i) + ".png", mat.T[i, :, :])

