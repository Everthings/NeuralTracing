from __future__ import print_function, division, absolute_import, unicode_literals
import numpy as np


from PIL import Image
from tf_unet import unet
from tf_unet.image_util import BaseDataProvider
from DataRetriever import DataRetriever
from random import shuffle

from TIFFExtractor import TIFFExtractor

import tensorflow as tf
import math

def main():
    #train("adam", 15000)
    for i in range(0, 30):  
        generatePredictions("neuron-data/data" + str(i + 1 ) + "_input.tif", i + 1)
    
    '''
    import imageio
    
    net = unet.Unet(channels=9,
                    n_class=2,
                    layers=3,
                    features_root=64,
                    cost_kwargs=dict(regularizer=0.001))
    
    tif, swc = DataRetriever().getData("subimages/", 219)
    
    predictions = net.predict("saved_model/model.ckpt", np.expand_dims(tif, axis = 0))
    
    predictions = convertBoolToIntMat(predictions[..., 0])
    
    imageio.imwrite("unet_output.png", predictions[0])
    '''
    
def train(optimize, num_data):
    data_provider = CustomDataProvider("subimages/", num_data)
    net = unet.Unet(channels=data_provider.channels,
                    n_class=data_provider.n_class,
                    layers=4,
                    features_root=64,
                    filter_size=3,
                    cost_kwargs=dict(regularizer=0.001))

    trainer = unet.Trainer(net, optimizer=optimize, batch_size=20, verification_batch_size = 20, opt_kwargs=dict())    
    path = trainer.train(data_provider, "./saved_model",
                         training_iters=10,
                         epochs=100,
                         dropout=0.5,
                         display_step=1)


def generatePredictions(file, save_num):
    import imageio

    mat = predict(128, 36, file)
    tif_mat = TIFFExtractor().extract(file)
    
    mat[mat > 0.0005] = 1
    imageio.imwrite("predicted" + str(save_num) + ".png", mat)
    imageio.imwrite("tif.png", np.max(tif_mat, axis = 0))
    
    print("Done")
    
    
def predict(square_size, prediction_size, tif_file_path):
    #Note: prediction size will be smaller than input tif size
    
    tif = TIFFExtractor().extract(tif_file_path)    
    tif = processTif(tif)
    
    squares = getSubSquares(square_size, prediction_size, tif)
    inputs = np.array(squares) 

    net = unet.Unet(channels=9,
                    n_class=2,
                    layers=4,
                    features_root=64,
                    cost_kwargs=dict(regularizer=0.001))
    
    predictions = net.predict("saved_model/model.ckpt", inputs)
    
    predictions = predictions[..., 1]

    return reconstructImage(predictions)
    
def processTif(tif):
    tif = np.transpose(tif, axes = (1, 2, 0))
    tif = tif.astype(np.float)
    if tif.shape[2] < 9:
        tif = np.pad(tif, [(0, 0), (0, 0), (0, 9 - tif.shape[2])], "constant")
        
    tif -= np.amin(tif)

    if np.amax(tif) != 0:
        tif /= np.amax(tif)
        
    return tif

def reconstructImage(squares):
    squares_x = int(math.sqrt(squares.shape[0]))
    
    im = []
    
    for row in range(squares_x):
        row_im = np.array(squares[row * squares_x])
        for column in range(1, squares_x):
            row_im = np.concatenate((row_im, squares[row * squares_x + column]), axis = 0)
    
        im.append(row_im)
       
        
    im_mat = im[0]
    for i in range(1, len(im)):
        im_mat = np.concatenate((im_mat, im[i]), axis = 1)
    
    return im_mat
    
def getSubSquares(square_size, prediction_size, image_mat):
    
    #only works if squares perfectly fit image_mat dimensiona
    
    squares = []
    x = 0
    y = 0
    while y + square_size <= image_mat.shape[1]:
        s = image_mat[x : x + square_size, y : y + square_size, :]
        squares.append(s)
        
        x += prediction_size
        if x + square_size > image_mat.shape[0]:
            y += prediction_size
            x = 0
            
    return squares


      


class CustomDataProvider(BaseDataProvider):
    
    retriever = DataRetriever()
    num_data = -1
    indexes = None
    search_path = ""

    def __init__(self, search_path, num_data):
        super(CustomDataProvider, self).__init__(None, None)
        self.file_idx = -1

        self.search_path =  search_path

        self.num_data = num_data
        self.indexes = self.getShuffledIndexes(num_data)
        
        self.channels = 9
        self.n_class = 2

        print("Number of channels: %s" % self.channels)
        print("Number of classes: %s" % self.n_class)

    def _cylce_file(self):
        self.file_idx += 1
        if self.file_idx >= self.num_data:
            self.file_idx = 0
            self.indexes = self.getShuffledIndexes(self.num_data)

    def _next_data(self):
        self._cylce_file()
        
        data = self.retriever.getDataUnet(self.search_path, self.indexes[self.file_idx])
        
        label = data[0]
        img = data[1]
        
        return img, label

    def getShuffledIndexes(self, max_index):
        # generates numbers 1 - max_index (inclusive) and then shuffles them
        
        indexes = [x for x in range(1, max_index + 1)]
        shuffle(indexes)
        
        return indexes

        

main()