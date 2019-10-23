import os
import time
import logging
import coloredlogs
import operator

import numpy as np
from PIL import Image

import matplotlib.pyplot as plt

plt.switch_backend("Agg")


def get_list_name(obj):
    if type(obj) is list:
        for i in range(len(obj)):
            if callable(obj[i]):
                obj[i] = obj[i].__name__
    elif callable(obj):
        obj = obj.__name__
    return obj


def get_logger(logger_name="tensorflow"):
    if logger_name is not None:
        logger = logging.getLogger(logger_name)
        logger.propagate = 0
    else:
        logger = logging.getLogger()
    return logger


def build_logger(folder=None, args=None, logger_name="tensorflow"):
    FORMAT = "%(asctime)s;%(levelname)s|%(message)s"
    DATEF = "%H-%M-%S"
    logging.basicConfig(format=FORMAT, level=logging.INFO)
    logger = get_logger(logger_name)
    # logger.setLevel(logging.DEBUG)

    if folder is not None:
        fh = logging.FileHandler(filename=os.path.join(
            folder, "logfile{}.log".format(time.strftime("%m-%d"))))
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s;%(levelname)s|%(message)s",
                                      "%H:%M:%S")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    LEVEL_STYLES = dict(
        debug=dict(color="magenta"),
        info=dict(color="green"),
        verbose=dict(),
        warning=dict(color="blue"),
        error=dict(color="yellow"),
        critical=dict(color="red", bold=True),
    )
    coloredlogs.install(level=logging.INFO,
                        fmt=FORMAT,
                        datefmt=DATEF,
                        level_styles=LEVEL_STYLES)

    sorted_list = sorted(vars(args).items(), key=operator.itemgetter(0))
    logger.info("#" * 120)
    logger.info("----------Configurable Parameters In this Model----------")
    for name, val in sorted_list:
        logger.info("# " + ("%20s" % name) + ":\t" + str(get_list_name(val)))
    logger.info("#" * 120)
    return logger


def plot_image(images, name, shape=None, figsize=(10, 10)):
    images = np.minimum(np.maximum(images, 0.0), 1.0)
    len_list = images.shape[0]
    im_list = []
    for i in range(len_list):
        im_list.append(images[i])

    if shape is None:
        unit = int(len_list**0.5)
        shape = (unit, unit)

    imshape = im_list[0].shape
    if imshape[2] == 1:
        im_list = [np.repeat(im, 3, axis=2) for im in im_list]
        imshape = im_list[0].size
    else:
        im_list = [im for im in im_list]
        imshape = im_list[0].size

    plt.figure(figsize=figsize)
    for i in range(shape[0] * shape[1]):
        plt.subplot(shape[0], shape[1], i + 1)
        plt.axis("off")
        print(im_list[i].size)
        plt.imshow(im_list[i], aspect="equal")

    # plt.subplots_adjust(hspace=5, wspace=5)
    plt.axis("off")
    plt.savefig(name)
    plt.close("all")


def pile_image(images_or_lists, name, shape=None):
    if isinstance(images_or_lists, (list, tuple)):
        list_num = len(images_or_lists)
        len_list = images_or_lists[0].shape[0]
        im_list = []
        for i in range(len_list):
            temp_im = []
            for j in range(list_num):
                temp_im.append(images_or_lists[j][i])
            temp_im = np.concatenate(temp_im, axis=1)
            im_list.append(temp_im)
    else:
        len_list = images_or_lists.shape[0]
        im_list = []
        for i in range(len_list):
            im_list.append(images_or_lists[i])

    imshape = im_list[0].shape
    if imshape[2] == 1:
        im_list = [np.repeat(im, 3, axis=2) for im in im_list]
        imshape = im_list[0].size
    else:
        im_list = [im for im in im_list]
        imshape = im_list[0].size

    len_list = len(im_list)
    if shape is None:
        unit = int(len_list**0.5)
        shape = (unit, unit)
    size = (shape[0] * imshape[0], shape[1] * imshape[1])
    result = Image.new("RGB", size)
    for i in range(min(len_list, shape[0] * shape[1])):
        x = i // shape[0] * imshape[0]
        y = i % shape[1] * imshape[1]
        temp_im = Image.fromarray(im_list[i])
        result.paste(temp_im, (x, y))

    result.save(name)
