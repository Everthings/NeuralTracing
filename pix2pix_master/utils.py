"""
Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division
import math
import json
import random
import pprint
import scipy.misc
import numpy as np
from time import gmtime, strftime

from DataRetriever import DataRetriever

pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])

# -----------------------------
# new added functions for pix2pix

def load_data(folder, image_index, flip=True, is_test=False):
    img_A, img_B = DataRetriever().getData(folder, image_index)
    #img_A, img_B = preprocess_A_and_B(img_A, img_B, flip=flip, is_test=is_test)

    img_A = img_A/127.5 - 1.
    img_B = img_B/127.5 - 1.

    img_AB = np.concatenate((img_A, img_B), axis=2)
    # img_AB shape: (fine_size, fine_size, input_c_dim + output_c_dim)
    return img_AB

def preprocess_A_and_B(img_A, img_B, load_size=128, fine_size=128, flip=True, is_test=False):
    if is_test:
        img_A_1 = scipy.misc.imresize(img_A[:, :, 0:3], [load_size, load_size])
        img_A_2 = scipy.misc.imresize(img_A[:, :, 3:6], [load_size, load_size])
        img_A_3 = scipy.misc.imresize(img_A[:, :, 6:9], [load_size, load_size])
        img_A = np.concatenate((img_A_1, img_A_2, img_A_3), axis = 2)
    else:
        img_A_1 = scipy.misc.imresize(img_A[:, :, 0:3], [load_size, load_size])
        img_A_2 = scipy.misc.imresize(img_A[:, :, 3:6], [load_size, load_size])
        img_A_3 = scipy.misc.imresize(img_A[:, :, 6:9], [load_size, load_size])
        img_A = np.concatenate((img_A_1, img_A_2, img_A_3), axis = 2)
        img_B = np.expand_dims(scipy.misc.imresize(img_B[:, :, 0], [load_size, load_size]), axis = 2)

        h1 = int(np.ceil(np.random.uniform(1e-2, load_size-fine_size)))
        w1 = int(np.ceil(np.random.uniform(1e-2, load_size-fine_size)))
        img_A = img_A[h1:h1+fine_size, w1:w1+fine_size]
        img_B = img_B[h1:h1+fine_size, w1:w1+fine_size]

        if flip and np.random.random() > 0.5:
            img_A = np.fliplr(img_A)
            img_B = np.fliplr(img_B)

    return img_A, img_B

# -----------------------------

def get_image(image_path, image_size, is_crop=True, resize_w=64, is_grayscale = False):
    return transform(imread(image_path, is_grayscale), image_size, is_crop, resize_w)

def save_images(images, size, image_path):
    transform_im = inverse_transform(images)
    return imsave(transform_im, size, image_path)

def imread(path, is_grayscale = False):
    if (is_grayscale):
        return scipy.misc.imread(path, flatten = True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)

def merge_images(images, size):
    return inverse_transform(images)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image

    return img

def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))

def transform(image, npx=64, is_crop=True, resize_w=64):
    # npx : # of pixels width/height of image
    if is_crop:
        cropped_image = center_crop(image, npx, resize_w=resize_w)
    else:
        cropped_image = image
    return np.array(cropped_image)/127.5 - 1.

def inverse_transform(images):
    return (images+1.)/2.


