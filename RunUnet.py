from __future__ import print_function, division, absolute_import, unicode_literals
import numpy as np


from PIL import Image
from tf_unet import unet
from tf_unet.image_util import BaseDataProvider
import TIFFExtractor
import SWCExtractor
import glob


def main():
    data_provider = ImageDataProvider()

    net = unet.Unet(channels=data_provider.channels,
                    n_class=data_provider.n_class,
                    layers=3,
                    features_root=64,
                    cost_kwargs=dict(regularizer=0.001),
                    )

    trainer = unet.Trainer(net, optimizer="momentum", opt_kwargs=dict(momentum=0.2))
    path = trainer.train(data_provider, "./unet_trained_bgs_example_data",
                         training_iters=32,
                         epochs=1,
                         dropout=0.5,
                         display_step=2)
    """
    # predict more using trained unet
    data_provider = DataProvider(10000, files)
    x_test, y_test = data_provider(1)
    prediction = net.predict(path, x_test)

    """



class ImageDataProvider(BaseDataProvider):
    """
    Generic data provider for images, supports gray scale and colored images.
    Assumes that the data images and label images are stored in the same folder
    and that the labels have a different file suffix
    e.g. 'train/fish_1.tif' and 'train/fish_1_mask.tif'
    Number of pixels in x and y of the images and masks should be even.

    Usage:
    data_provider = ImageDataProvider("..fishes/train/*.tif")

    :param search_path: a glob search pattern to find all data and label images
    :param a_min: (optional) min value used for clipping
    :param a_max: (optional) max value used for clipping
    :param data_suffix: suffix pattern for the data images. Default '.tif'
    :param mask_suffix: suffix pattern for the label images. Default '_mask.tif'
    :param shuffle_data: if the order of the loaded file path should be randomized. Default 'True'

    """

    def __init__(self, search_path, a_min=None, a_max=None, data_suffix="_input.tif", mask_suffix='_label.swc',
                 shuffle_data=True):
        super(ImageDataProvider, self).__init__(a_min, a_max)
        self.data_suffix = data_suffix
        self.mask_suffix = mask_suffix
        self.file_idx = -1
        self.shuffle_data = shuffle_data

        self.data_files = self._find_data_files(search_path)

        if self.shuffle_data:
            np.random.shuffle(self.data_files)

        assert len(self.data_files) > 0, "No training files"
        print("Number of files used: %s" % len(self.data_files))

        image_path = self.data_files[0]
        label_path = image_path.replace(self.data_suffix, self.mask_suffix)
        img = TIFFExtractor().extract(image_path)
        mask = SWCExtractor().extract(label_path)
        self.channels = 1 if len(img.shape) == 2 else img.shape[-1]
        self.n_class = 2 if len(mask.shape) == 2 else mask.shape[-1]

        print("Number of channels: %s" % self.channels)
        print("Number of classes: %s" % self.n_class)

    def _find_data_files(self, search_path):
        all_files = glob.glob(search_path)
        return [name for name in all_files if self.data_suffix in name and not self.mask_suffix in name]

    def _load_file(self, path, dtype=np.float32):
        return np.array(Image.open(path), dtype)

    def _cylce_file(self):
        self.file_idx += 1
        if self.file_idx >= len(self.data_files):
            self.file_idx = 0
            if self.shuffle_data:
                np.random.shuffle(self.data_files)

    def _next_data(self):
        self._cylce_file()
        image_name = self.data_files[self.file_idx]
        label_name = image_name.replace(self.data_suffix, self.mask_suffix)

        img = self._load_file(image_name, np.float32)
        label = self._load_file(label_name, np.bool)

        return img, label


main()