# NeuralTracing

## Dataset
We started with an intial dataset of 30 1024x201024xN input(tif) - output(swc) pairs. To aument the dataset to increase the number of datapoints and prevent the model from overfitting to specific global features of the 30 full images, we spliced each image at various points to create roughly 500 100x100 subimages. The resulting datapoints are stored under "/subimages" where images with suffix "_swc" are the labeled outputs and the "_tif_[number]" are the inputs. The three .tiffs for each label correspond to different layer grouping as PIL doesn't allow for the saving of more than 4 layer channels. Thus, a tif with 9 layers will be stores as group 1 (layers 0 - 2), group 2 (layers 3 - 5), and group 3 (layers 6 - 8).
