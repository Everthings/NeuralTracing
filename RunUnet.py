from __future__ import print_function, division, absolute_import, unicode_literals
import numpy as np
from tf_unet import image_gen
from tf_unet import unet
from tf_unet import util


from scripts.radio_util import DataProvider
from tf_unet import unet

def main():
    files = get_files()
    data_provider = DataProvider(nx, files)

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

    # predict more using trained unet
    data_provider = DataProvider(10000, files)
    x_test, y_test = data_provider(1)
    prediction = net.predict(path, x_test)
