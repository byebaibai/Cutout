from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from visdom import Visdom
import math
import os.path
import getpass
from sys import platform as _platform
from six.moves import urllib
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
import time

viz = Visdom()
assert viz.check_connection()

try:
    import matplotlib.pyplot as plt

except BaseException as err:
    print('Skipped matplotlib example')
    print('Error message: ', err)

"""-------Initial Setup-------"""
x, y = 0, 0

Gen_Loss = viz.line(
    Y=np.array([y]),
    X=np.array([x]),
)
Val_Loss = viz.line(
    Y=np.array([y]),
    X=np.array([x]),
)
Disc_Loss = viz.line(
    Y=np.array([y]),
    X=np.array([x]),
)
SSIM_Loss = viz.line(
    Y=np.array([y]),
    X=np.array([x]),
)
the_text = viz.text("")

Val_Image = viz.images(torch.zeros((4, 3, 768, 256)), nrow=4)

def validShow(val_loss=0.0, count=0, image=None):
    viz.line(
        X=np.array([count]),
        Y=np.array([val_loss]),
        win=Val_Loss,
        name='Validation Loss',
        update='append',
        opts=dict(
            title='Val_loss',
            linecolor=np.array([[46, 64, 83]])
        )
    )

    viz.images(
        image,
        win=Val_Image,
        nrow=8,
    )


def Visualization(gen_loss=0.0, disc_loss=0.0, ssim_loss=0.0, Count=0, text=None):
    global the_text

    viz.line(
        X=np.array([Count]),
        Y=np.array([gen_loss]),
        win=Gen_Loss,
        name='Binary',
        update='append',
        opts=dict(
            title='Bin_loss',
            linecolor=np.array([[224, 17, 93]])
        )
    )

    viz.line(
        X=np.array([Count]),
        Y=np.array([disc_loss]),
        win=Disc_Loss,
        name='Dice',
        update='append',
        opts=dict(
            title='Dice_loss',
            linecolor=np.array([[28, 191, 28]])
        )
    )

    viz.line(
        X=np.array([Count]),
        Y=np.array([ssim_loss]),
        win=SSIM_Loss,
        name='SSIM',
        update='append',
        opts=dict(
            title='SSIM_loss',
            linecolor=np.array([[165, 105, 189]])
        )
    )

    viz.text(text,
             win=the_text,
             append=True)