import numpy as np
from scipy.optimize import minimize


from TIFFExtractor import TIFFExtractor
from SWCExtractor import SWCExtractor


class LayerMapping():

    # use l2, return float
    TIFF = None
    SWC = None

    def map(self, tiff_filepath, swc_filepath):
        import imageio
        # Main function that calls things
        #x0 = [8/self.SWC.shape[0], 0]
        x0 = [1, 0]
        bound_k = (0.1, 2)
        bound_b = (-5, 5)
        bounds = (bound_k, bound_b)
        self.TIFF = TIFFExtractor().extract(tiff_filepath)
        self.SWC = SWCExtractor().extract((10, 1024, 1024), swc_filepath)

        print(minimize(self.objective, x0, method='BFGS', bounds=bounds))


    def map_to_8(self, mat, k, b):
        # creates 8 layers and distributes intensity
        def good_round(height):
            lower_bound = height // 1
            res = height - lower_bound
            if (lower_bound > 6): # 7 or above
                return 6, 255, 0
            if (lower_bound < 0):
                return 0, 0, 255
            if (res==0):
                return lower_bound, 0, 255
            else:
                inverse_down = 1/res
                inverse_up = 1/(1-res)
                normalizer = 255/(inverse_up + inverse_down)
                up_intensity = inverse_up * normalizer
                down_intensity = inverse_down * normalizer
                return lower_bound, up_intensity, down_intensity

        mapped = np.zeros((8, 1024, 1024), dtype=np.uint8)
        for z in range(mat.shape[0]):
            new_height = (z * k) + b
            intensity = good_round(new_height)
            index_lower = intensity[0]
            index_upper = index_lower + 1
            for x in range(mat.shape[1]):
                for y in range(mat.shape[2]):
                    if mat[z][x][y] > 0:
                        intensity1 = min(mapped[int(index_lower)][x][y] + intensity[1], 255)
                        mapped[int(index_lower)][x][y] = intensity1
                        intensity2 = min(mapped[int(index_upper)][x][y] + intensity[2], 255)
                        mapped[int(index_upper)][x][y] = intensity2
                        
        return mapped

    def objective(self, args):
        k = args[0]
        b = args[1]

        print("In objective. k = " + str(k), "b = " + str(b))
        mapped_SWC = self.map_to_8(self.SWC, k, b)
        print("Finished mapping")

        """
        error = 0
        for z in range(8):
            output = mapped_SWC[z]
            label = self.TIFF[z]
            for x in range(1024):
                for y in range(1024):
                    error += (output[x][y] - label[x][y])**2
        """
        loss = np.sum((mapped_SWC - self.TIFF) ** 2)
        print("Finished error calculation: " + str(loss))
        return loss


def main():
    LayerMapping().map("neuron-data/data1_input.tif", "neuron-data/data1_label.swc")


main()