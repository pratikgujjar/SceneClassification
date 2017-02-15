import os
import cv2
import numpy as np
import inception_v3 as inception
from keras.models import Sequential, Model
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import Embedding
from keras.layers import TimeDistributedDense
from keras.layers import Merge
from keras.layers import Activation
from keras.optimizers import RMSprop
from keras.preprocessing import image
from keras.preprocessing import sequence
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras import backend as K

K.set_image_dim_ordering('tf')

# path of folder with training images
train_path = '/home/pratik/git_projects/ContextAwareness/dataset/envelope'
# path of folder with test images
test_path = '/home/pratik/git_projects/ContextAwareness/dataset/envelope_test'

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# Configuration constants
MAX_LABELS = 2
NUM_LABELS = 8 # Number of classes
OUTPUT_DIM = 128
IM_SIZE = (256,256)
DIM_WORD = 300

# Create the model

# Import Inception v3 as the standard CNN feature extractor
# Start with an Inception V3 model, not including the final softmax layer.
print "Loading Inception V3 as image model"
image_model = inception.InceptionV3(weights='imagenet')
print 'Loaded Inception model'

# Turn off training on base model layers
for layer in image_model.layers:
    layer.trainable = False

# Add on a dense layer to non-linearize features; Feature size is 2048
x = Dense(2048, activation='relu')(image_model.get_layer('flatten').output)
x = Dropout(0.5)(x)

# Build the label LSTM model
print "Loading label model"
label_model = Sequential()
label_model.add(Embedding(NUM_LABELS, 256, input_length=MAX_LABELS))
label_model.add(LSTM(output_dim=128, return_sequences=True))
label_model.add(TimeDistributedDense(128))
print "Label model loaded"

# Repeat image feature vector to turn it into a sequence
print "Repeat model loading"
x = RepeatVector(MAX_LABELS)(x)
# image_model.add(RepeatVector(MAX_LABELS))
print "Repeat model loaded"

img_model = Model(input=image_model.input, output=x)

# Merge image features and label features by concatenation
print "Merging image and label features"
model = Sequential()
model.add(Merge([img_model, label_model], mode='concat', concat_axis=-1))

# Encode this vector into a single representation vector
model.add(LSTM(256, return_sequences=False))
model.add(Dense(NUM_LABELS))
model.add(Activation('softmax'))

print "Merged"

# Compile model
optimizer = RMSprop()
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Show some debug output
print (model.summary())
print 'Trainable weights'
print model.trainable_weights

#Save the model
model_json = model.to_json()
with open("scene8_model.json", "w") as json_file:
    json_file.write(model_json)

print "Model saved to disk"

# Ready the data

print "Preprocessing data"
label_map=['coast', 'forest', 'city', 'buildings', 'highway', 'street', 'country', 'mountain']

# Load image data
images = []

listing = os.listdir(train_path)
for file in (listing):
    path = train_path + "/" + file
    print path
    img = cv2.imread(train_path + "/" + file, cv2.IMREAD_COLOR)
    img.resize((299,299,3))
    images.append(img)

images = np.asarray(images) # Change list to numpy array

# Load label data
labels = []
words = [txt.split() for txt in label_map]
unique = []
for word in words:
    unique.extend(word) # Integrate separate list elements into a single list

unique = list(set(unique)) # Finds unique words
word_index = {}
index_word = {}
for i, word in enumerate(unique):
    word_index[word] = i
    index_word[i] = word

partial_captions = []
for label in label_map:
    one = [word_index[lab] for lab in label.split()]
    partial_captions.append(one)

partial_captions = sequence.pad_sequences(partial_captions, maxlen=MAX_LABELS, padding='post')

next_words = np.zeros((8, NUM_LABELS))
for i, label in enumerate(label_map):
    label = label.split()
    x = [word_index[lab] for label in label]
    x = np.asarray(x)
    next_words[i,x] = 1

print "Data preprocessing done"

print "Begin training"
model.fit([images, partial_captions], next_words, batch_size=16, nb_epoch=5)
# Save weights
model.save_weights('scene8_captionlearn.h5')  # always save your weights after training or during training
print "Weights saved to disk"
