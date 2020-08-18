# -*- coding: utf-8 -*-

# use!: from utils_ol import *
# To achieve the right and the rights of the writer
# The original project is for this place https://github.com/kylemcdonald/SmileCNN/
# Last edited by by olla and mohamed
from cv2 import resize ,cvtColor ,COLOR_BGR2GRAY
import numpy as np
from skimage.measure import block_reduce
from io import BytesIO
import PIL.Image
import IPython.display
import shutil
import math
import os
import fnmatch
from matplotlib import pyplot as plt

#------------cut--face---resize---------#
def resize_face(img,h=64,w=64):
    block_size=2
    ######################################################################################
    gray = cvtColor(img, COLOR_BGR2GRAY)                                               ###
    nnm = resize(gray, (h,w))                                                          ###
    imgxy = block_reduce(nnm, block_size=(block_size, block_size), func=np.mean)       ###
    imgxy = np.asarray(imgxy)                                                          ###
    imgxy= imgxy.astype(np.float32) / 255.                                             ###
    imm = np.expand_dims(imgxy, axis=-1)                                               ###
    print (imm.dtype, imm.min(), imm.max(), imm.shape)                                 ###
    ######################################################################################
    return imm

#------crop_fun_and_resize_frame--------#

def crop_and_resize(img, target_size=32, zoom=1):
    small_side = int(np.min(img.shape) * zoom)
    reduce_factor = small_side / target_size
    crop_size = target_size * reduce_factor
    mid = np.array(img.shape) / 2
    half_crop = crop_size / 2
    center = img[mid[0]-half_crop:mid[0]+half_crop,
    	mid[1]-half_crop:mid[1]+half_crop]
    return block_reduce(center, (reduce_factor, reduce_factor), np.mean)

#----end----#

#--------list fun--all--files---------#

def list_all_files(directory, extensions=None):
    for root, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            base, ext = os.path.splitext(filename)
            joined = os.path.join(root, filename)
            if extensions is None or ext.lower() in extensions:
                yield joined

#----end----#

#----show--array--and--plot--images-#
def show_array(a, fmt='PNG', filename=None):
    a = np.squeeze(a)
    a = np.uint8(np.clip(a, 0, 255))
    image_data = BytesIO()
    PIL.Image.fromarray(a).save(image_data, fmt)
    if filename is None:
        #IPython.display.display(IPython.display.Image(data=image_data.getvalue())) #note book onlay
        plt.savefig(image_data, format='png')
        image_data.seek(0)
        im = PIL.Image.open(image_data)
        img = np.array(im)
        plt.imshow(img)
        plt.show() # plot
        im.show() # pil
        image_data.close() #del
    else:
        with open(filename, 'w') as f:
            image_data.seek(0)
            shutil.copyfileobj(image_data, f)

#----end----#

#------fun---find--rectangle-----#
def find_rectangle(n, max_ratio=2):
    sides = []
    square = int(math.sqrt(n))
    for w in range(square, max_ratio * square):
        h = n / w
        used = w * h
        leftover = n - used
        sides.append((leftover, (w, h)))
    return sorted(sides)[0][1]

#--end_s--#

#---------------fun--make--and--get--rshape--images-!!!-----------------#
# should work for 1d and 2d images, assumes images are square but can be overriden

def make_mosaic(images, n=None, nx=None, ny=None, w=None, h=None):
    if n is None and nx is None and ny is None:
        nx, ny = find_rectangle(len(images))
    else:
        nx = n if nx is None else nx
        ny = n if ny is None else ny
    images = np.array(images)
    if images.ndim == 2:
        side = int(np.sqrt(len(images[0])))
        h = side if h is None else h
        w = side if w is None else w
        images = images.reshape(-1, h, w)
    else:
        h = images.shape[1]
        w = images.shape[2]
    image_gen = iter(images)
    mosaic = np.empty((h*ny, w*nx))
    for i in range(ny):
        ia = (i)*h
        ib = (i+1)*h
        for j in range(nx):
            ja = j*w
            jb = (j+1)*w
            mosaic[ia:ib, ja:jb] = next(image_gen)
    return mosaic

#----end_s-A---#


#----------------------------end-All-----------------------------#
#---end--end---#
