# NeuralTracing
This repo contains everything used to trace semantically segment neurons through neural cross-sectional images. We experimented with two network architechures: Unet and Pix2Pix. We found that the Unet converged quicker, and produced better results when trained in Google Colab.

### Prerequisites
Libraries you'll need to run the code:
```
Tensorflow
PIL
numpy
imageio
skimage
scipy
pypng
```

### Installing
To install, simply run the following command in the command line:
```
git clone https://github.com/Everthings/NeuralTracing
```

## Dataset
We started with an intial dataset of 30 1024x201024xN input(tif) - output(swc) pairs. To aument the dataset to increase the number of datapoints and prevent the model from overfitting to specific global features of the 30 full images, we spliced each image at various points to create roughly 500 128x128 subimages. The resulting datapoints are stored under ``/subimages`` where images with suffix ``_swc`` are the labeled outputs and the ``_tif_[number]`` are the inputs. The three .tiffs for each label correspond to different layer grouping as PIL doesn't allow for the saving of more than 4 layer channels. Thus, a tif with 9 layers will be stores as group 1 (layers 0 - 2), group 2 (layers 3 - 5), and group 3 (layers 6 - 8). We also defined a problem area in each full image from which we sampled an additional 150 images to help the network with difficult features in the images.

### Examples!
#### SWC
![alt text](https://user-images.githubusercontent.com/16503485/57531125-c4c8c200-7306-11e9-8593-43c9812788bf.png)

#### Corresponding TIFFs
![alt text](https://user-images.githubusercontent.com/16503485/57531129-c6928580-7306-11e9-9289-0d0b10af4a19.png)
![alt text](https://user-images.githubusercontent.com/16503485/57531133-c85c4900-7306-11e9-8675-7cc8cebf1bac.png)
![alt text](https://user-images.githubusercontent.com/16503485/57531135-c98d7600-7306-11e9-9d2f-0bacb29e4524.png)

## Training/Generating Images using Unet
Run ```RunUnet.py``` and comment out either ```train``` or ```generatePredictions```. Additionally, unzip ```saved_model.zip``` for fully trained weights.

## Training/Generating Images using Pix2Pix
Run ```RunPix2Pix.py``` and make appropriate changes to the ArgumentParser. Additionally, unzip ```checkpoint.zip``` for fully trained weights.

## Authors
Andy Xu and Spencer Solit

## Acknowledgments
We adapted the following to fit our needs to expedite the coding process:
1. tf-unet from https://github.com/jakeret/tf_unet
2. pix2pix_master from https://github.com/yenchenlin/pix2pix-tensorflow
3. SWCExtractor.py from https://github.com/zhihaozhengutd/CODES

