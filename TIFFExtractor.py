from skimage import io
import skimage.measure
import numpy as np
from numpy.lib.stride_tricks import as_strided
import png

class TIFFExtractor():
    
    def extract(self, filePath):
        imageMat = io.imread(filePath)
        imageMat = imageMat[:, 0:-2, 0:-2]
        imageMat = imageMat * 16
        pooledMat = np.zeros(shape=(8, 511, 511), dtype=np.uint16)
        
        for i in range(imageMat.shape[0]):
            pooledMat[i] = self._maxPool(imageMat[i], kernel_size=2, stride=2, padding=0)
        #adds extra channel to matrix for DataProvider    
        pooledMat = np.expand_dims(pooledMat, axis=3)
        return pooledMat;

    def _maxPool(self, A, kernel_size, stride, padding):
        """
        2D Pooling
    
        Parameters:
            A: input 2D array
            kernel_size: int, the size of the window
            stride: int, the stride of the window
            padding: int, implicit zero paddings on both sides of the input
            pool_mode: string, 'max' or 'avg'
        """
        # Padding
        A = np.pad(A, padding, mode='constant')
    
        # Window view of A
        output_shape = ((A.shape[0] - kernel_size)//stride + 1,
                        (A.shape[1] - kernel_size)//stride + 1)
        kernel_size = (kernel_size, kernel_size)
        A_w = as_strided(A, shape=output_shape + kernel_size,
                         strides=(stride*A.strides[0],
                                  stride*A.strides[1]) + A.strides)
        A_w = A_w.reshape(-1, *kernel_size)
    
        return A_w.max(axis=(1, 2)).reshape(output_shape)
    
        
def main():
    extract = TIFFExtractor()
    mat = extract.extract("neuron_data/data1_input.tif")
    print(mat.shape)
    
    #for i in range(mat.shape[0]):
    #    png.from_array(mat[i], 'L').save("TIFF" + str(i) + ".png")
     
    #print("TIFF Image Drawn!")


main()

