# Part 3 - Making new predictions
import numpy as np
import tensorflow as tf
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator, load_img
import pandas as pd
import os
import matplotlib.pyplot as plt

Image_Width = 128
Image_Height = 128
Image_Size = (Image_Width, Image_Height)
Image_Channels = 3

model = tf.keras.models.load_model("keras_cnn")
print("Model loaded")

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1. / 255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size=(64, 64),
                                                 batch_size=32,
                                                 class_mode='binary')
test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')
control_set = test_datagen.flow_from_directory('dataset/control_set',
                                               target_size=(64, 64),
                                               batch_size=32,
                                               class_mode='binary')
# test_image = image.load_img('dataset/control_set/cat_or_dog_1.jpg', target_size=(64, 64))
# test_image = image.img_to_array(test_image)
# test_image = np.expand_dims(test_image, axis=0)


control_files = os.listdir("dataset/control_set")
test_df = pd.DataFrame({
    'filename': control_files
})
nb_samples = test_df.shape[0]
directory = 'dataset/control_set'
predict = model.predict(control_files, steps=np.ceil(nb_samples))
if predict[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
print(prediction)

test_df['category'] = np.argmax(predict, axis=-1)
label_map = dict((v, k) for k, v in train_datagen.class_indices.items())
test_df['category'] = test_df['category'].replace(label_map)
test_df['category'] = test_df['category'].replace({'dog': 1, 'cat': 0})

control_test = test_df.head(18)
control_test.head()
plt.figure(figsize=(12, 24))
for index, row in control_test.iterrows():
    filename = row['filename']
    category = row['category']
    img = load_img('dataset/control_set/' + filename, target_size=Image_Size)
    plt.subplot(6, 3, index + 1)
    plt.imshow(img)
    plt.xlabel(filename + '(' + "{}".format(category) + ')')
plt.tight_layout()
plt.show()


