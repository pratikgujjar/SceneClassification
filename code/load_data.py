import numpy as np
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')
import cv2
import os

train_path = '/home/pratik/git_projects/ContextAwareness/dataset/envelope_train'    #path of folder to save training images
test_path = '/home/pratik/git_projects/ContextAwareness/dataset/envelope_test'      #path of folder to save test images

# load training data
x_train = []
y_train = []

listing = os.listdir(train_path)
for file in sorted(listing):
    img = cv2.imread(train_path + "/" + file, cv2.IMREAD_COLOR)
    x_train.append(img.T)
    class_num = int(os.path.splitext(file)[0].split('_')[0].encode('utf8'))
    y_train.append(class_num)

y_train[:] = [y - 1 for y in y_train] # Readjusting class numbers to begin from 0
x_train = np.asarray(x_train) # Change list to numpy array
y_train = np.asarray(y_train) # Change list to numpy array

# load test data
x_test = []
y_test = []

listing = os.listdir(test_path)
for file in sorted(listing):
    img = cv2.imread(test_path + "/" + file, cv2.IMREAD_COLOR)
    x_test.append(img.T)
    class_num = int(os.path.splitext(file)[0].split('_')[0].encode('utf8'))
    y_test.append(class_num)

y_test[:] = [y - 1 for y in y_test] # Readjusting class numbers to begin from 0
x_test = np.asarray(x_test) # Change list to numpy array
y_test = np.asarray(y_test) # Change list to numpy array

# normalize inputs from 0-255 to 0.0-1.0
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = x_train / 255.0
x_test = x_test / 255.0

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

print "Image vectors loaded into x_train, y_train, x_test and y_test"