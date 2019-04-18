import numpy as np
from PIL import Image

class TIFExtractor():
    
    def extractMatrix():
        im = Image.open('neuron_data/1xppk+Dcr_01-AlstR_TRiP27280_001 ch 1.tif')
        im.show()
        

def main():
    extract = TIFExtractor()
    extract.extractMatrix()
    
main()