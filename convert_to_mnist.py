from PIL import Image, ImageFilter
from matplotlib import pyplot as plt
import numpy as np


def create_chunks(list_name, n):
    for i in range(0, len(list_name), n):
        yield list_name[i: i + n]


def image_prepare(argv):
    im = Image.open(argv).convert('L')
    width = float(im.size[0])
    height = float(im.size[1])
    newImage = Image.new('L', (28, 28), 255)

    if width > height:
        nheight = int(round((20.0 / width * height), 0))
        if nheight == 0:
            nheight = 1
        img = im.resize((20, nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wtop = int(round(((28 - nheight) / 2), 0))
        newImage.paste(img, (4, wtop))
    else:
        nwidth = int(round((20.0 / height * width), 0))
        if nwidth == 0:
            nwidth = 1
        img = im.resize((nwidth, 20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wleft = int(round(((28 - nwidth) / 2), 0))
        newImage.paste(img, (wleft, 4))

    tv = list(newImage.getdata())  # get pixel values
    tva = [(255 - x) for x in tv]  # 255 becomes 0, 0 becomes 255

    return tva


def array_to_export():
    image_to_export = image_prepare('./out.png')
    chunk_size = 28
    tva1 = list(create_chunks(image_to_export, chunk_size))
    tva2 = np.asarray(tva1)

    return tva2


def create_export_image():
    image_to_export = [image_prepare('./out.png')]  # file path here
    width = 28
    height = 28
    newArr = [[0 for i in range(width)] for j in range(height)]  # declare the matrix
    k = 0

    for i in range(width):
        for j in range(height):
            newArr[i][j] = image_to_export[0][k]
            k = k + 1

    return newArr


def plot_image():
    x = create_export_image()
    plt.imshow(x, interpolation='nearest')
    plt.savefig('out_mnist.png')
    plt.show()
