from skimage import io
import png

class TIFFExtractor():
    
    def extractMatrix(self, filePath):
        imageMat = io.imread(filePath)
        imageMat = imageMat * 16
        return imageMat;
        
def main():
    extract = TIFFExtractor()
    mat = extract.extractMatrix("neuron_data/1xppk+Dcr_01-AlstR_TRiP27280_007 ch 1.tif")
    
    for i in range(mat.shape[0]):
        png.from_array(mat[i], 'L').save("TIFF" + str(i) + ".png")
    
    print("TIFF Image Drawn!")
    
main()

