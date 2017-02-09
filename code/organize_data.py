from numpy import *
import os
from shutil import copyfile, move

# input image dimensions
img_rows, img_cols = 200, 200

# number of channels
img_channels = 1

#  Data

source_path = '/home/pratik/git_projects/ContextAwareness/dataset/envelope'           #path of folder of images
train_path = '/home/pratik/git_projects/ContextAwareness/dataset/envelope_train'     #path of folder to save training images
test_path = '/home/pratik/git_projects/ContextAwareness/dataset/envelope_test'      #path of folder to save test images

listing = os.listdir(test_path)
num_samples=size(listing)
print num_samples

s=8         # Class_number
count = 1   # Image_number

for file in listing:

    # Code to rename files according to spec: classnumber_classname_imagenumber

    # class_number = '{:2d}'.format(s)
    # if ("tallbuilding_") in file:
    #     print file
    #     # print fname
    #     # filename = os.path.splitext(file)[0].split('_')[0].encode('utf8')
    #     filename = class_number + "_" + "buildings" + "_" + str(count) + ".jpg"
    #     print filename
    #     count += 1
    #     copyfile(source_path + "/" + file, train_path + '/' + filename)

    # Code to split data into training and testing set.
    # 250 train images, rest is test

    # img_num = os.path.splitext(file)[0].split('_')[-1].encode('utf8')
    # if int(img_num) > 250:
    #     print file
    #     move(train_path + "/" + file, test_path + '/' + file)

    os.rename(test_path + "/" + file, test_path + "/" + file.replace(" ", ""))

print "Done!"