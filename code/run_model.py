import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image
import random
from keras import backend as K
import os
K.set_image_dim_ordering('th')
from keras.utils.visualize_util import plot

# path of folder with training images
train_path = '/home/pratik/git_projects/ContextAwareness/dataset/envelope_train'
# path of folder with test images
test_path = '/home/pratik/git_projects/ContextAwareness/dataset/envelope_test'

N_CLASSES = 8
IM_SIZE = (256,256)

# Define class mapping
class_map=['coast', 'forest', 'city', 'buildings', 'highway', 'street', 'country', 'mountain']

# load json and create model
json_file = open('scene8_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# load weights into new model
model.load_weights("scene8_pretrain.h5")
print("Loaded model from disk")

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

# test_datagen = ImageDataGenerator(rescale=1. / 255)
# test_generator = test_datagen.flow_from_directory(
#     test_path,  # this is the target directory
#     target_size=IM_SIZE,  # all images will be resized to 256x256
#     batch_size=8,
#     classes=class_map,
#     class_mode='categorical')
#
# predict = model.predict_generator(
#     test_generator,
#     val_samples=8)
#
# print type(predict)
# print predict

# specify number of test images
batch_size = 5

head = """<html>
<head></head>
<body>
<table style="width:50%">
<tr>
    <th>Image</th>
    <th>Classificaton Scores</th>
</tr>
"""

tail = """
</table>
</body>
</html>
"""

body = """ """


for scene in class_map:
    directory = test_path + '/' + scene + '/'
    for index in range(0, batch_size):
        filename = random.choice(os.listdir(directory))
        print filename
        img_path = directory + filename
        img = image.load_img(img_path, target_size=IM_SIZE)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)

        preds = model.predict(x)
        print('Predicted:', preds)

        predict = class_map[np.argmax(preds)]
        prediction = ""
        for p in predict:
            prediction += str(p) + " \t "

        body += """<tr>	<td> <img src='""" + img_path + """' width="300" height="300"/> </td> <td>"""+str(prediction)+""" </td> </tr>\n"""

    html_file = open('a3_15_' + scene + '.html', 'w')
    message = head+body+tail
    html_file.write(message)
    html_file.close()
    body = """"""
print "Classification scores in HTML format. Done"

plot(model, to_file='model.png')
print "Model visualization drawn to model.png"


