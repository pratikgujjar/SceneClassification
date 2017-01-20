from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,adam
from keras.utils import np_utils

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import theano
from PIL import Image
from numpy import *
# SKLEARN
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split

# input image dimensions
img_rows, img_cols = 200, 200

# number of channels
img_channels = 1

#  Data

path1 = '/home/pratik/git_projects/ContextAwareness/dataset/envelope'           #path of folder of images
path2 = '/home/pratik/git_projects/ContextAwareness/dataset/envelope_train'     #path of folder to save training images
path3 = '/home/pratik/git_projects/ContextAwareness/dataset/envelope_test'      #path of folder to save test images

listing = os.listdir(path1)
num_samples=size(listing)
print num_samples

for file in listing:
    im = Image.open(path1 + '/' + file)
    img = im.resize((img_rows,img_cols))
    gray = img.convert('L')
                #need to do some more processing here
    gray.save(path2 +'\\' +  file, "JPEG")

imlist = os.listdir(path2)

im1 = array(Image.open('input_data_resized' + '\\'+ imlist[0])) # open one image to get size
m,n = im1.shape[0:2] # get the size of the images
imnbr = len(imlist) # get the number of images

import os
import glob
from shutil import copyfile

train_dir = '/home/pratik/voxnet1/train/'
dest_dir = '/home/pratik/voxnet1/organisedtrain/'
validation_dir = '/home/pratik/voxnet1/validation/'
test_dir = '/home/pratik/voxnet1/organisedtest/'

for s in range(1, 13):
    count = 1
    s_dir = '{:02d}'.format(s)
    train_path = validation_dir + 'S' + s_dir
    for fname in glob.glob(os.path.join(train_path, '*.pcd')):
        if (".003.pcd") in fname:
            #print fname
            filename = os.path.splitext(fname)[0].split('/')[-1].encode('utf8')
            filename = filename[0:len(filename) - 8]  # r Remove rotation number
            filename = filename + '{:04d}'.format(count)
            count += 1
            filename = filename + '.002.pcd'
            # print count
            copyfile(fname, test_dir + 'S' + s_dir + '/' + filename)

print "Done!"