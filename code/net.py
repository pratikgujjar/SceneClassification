import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')

# path of folder with training images
train_path = '/home/pratik/git_projects/ContextAwareness/dataset/envelope_train'
# path of folder with test images
test_path = '/home/pratik/git_projects/ContextAwareness/dataset/envelope_test'

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

N_CLASSES = 8
IM_SIZE = (256,256)

# Create the model

model = Sequential()
model.add(Convolution2D(32, 3, 3, input_shape=(3, 256, 256), activation='relu', border_mode='same'))
model.add(Dropout(0.2))
model.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D(64, 5, 5, activation='relu', border_mode='same'))
model.add(MaxPooling2D(pool_size=(4,4)))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu', W_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Dense(N_CLASSES, activation='softmax'))

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

# Show some debug output
print (model.summary())
print 'Trainable weights'
print model.trainable_weights

# Define class mapping
class_map=['coast', 'forest', 'city', 'buildings', 'highway', 'street', 'country', 'mountain']

# Generate training and testing data
train_datagen = ImageDataGenerator(rescale=1. / 255)
train_generator = train_datagen.flow_from_directory(
    train_path,  # this is the target directory
    target_size=IM_SIZE,  # all images will be resized to 299x299 Inception V3 input
    batch_size=8,
    classes=class_map,
    class_mode='categorical')

test_datagen = ImageDataGenerator(rescale=1. / 255)
test_generator = test_datagen.flow_from_directory(
    test_path,  # this is the target directory
    target_size=IM_SIZE,  # all images will be resized to 256x256
    batch_size=8,
    classes=class_map,
    class_mode='categorical')

model.fit_generator(
    train_generator,
    samples_per_epoch=2000,
    nb_epoch=25,
    validation_data=test_generator,
    verbose=2,
    nb_val_samples=80,
    callbacks=[])

predict_generator = test_datagen.flow_from_directory(
    test_path,  # this is the target directory
    target_size=IM_SIZE,  # all images will be resized to 256x256
    batch_size=2,
    classes=class_map,
    class_mode='categorical')

#Save the model
model_json = model.to_json()
with open("scene8_model.json", "w") as json_file:
    json_file.write(model_json)

print "Model saved to disk"

# Save weights
model.save_weights('scene8_pretrain.h5')  # always save your weights after training or during training
print "Weights saved to disk"

# model.predict_generator(
#     predict_generator,
#     val_samples=2,
#     max_q_size=10,
#     pickle_safe=False)

img_path = '/home/pratik/git_projects/ContextAwareness/dataset/envelope_test/coast/1_coast_251.jpg'
img = image.load_img(img_path, target_size=IM_SIZE)
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

preds = model.predict(x)
print('Predicted:', preds)


scores = model.evaluate_generator(
    predict_generator,
    val_samples=10,
    max_q_size=10,
    nb_worker=1,
    pickle_safe=False)

print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# # Compile model
# epochs = 25
# lrate = 0.01
# decay = lrate/epochs
# sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
# model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
# print(model.summary())
#
# X_test = x_test[0:32]
# Y_test = y_test[0:32]
# np.random.seed(seed)
# model.fit(x_train, y_train, validation_data=(X_test, Y_test), nb_epoch=epochs, shuffle=True, batch_size=8)
#
# # Final evaluation of the model
# scores = model.evaluate(x_test, y_test, verbose=0)
# print("Accuracy: %.2f%%" % (scores[1]*100))

# specify number of test images
