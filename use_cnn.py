# Part 3 - Making new predictions
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator, load_img

model = tf.keras.models.load_model("keras_cnn")
print("Model loaded")

train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1. / 255)
training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size=(128, 128),
                                                 batch_size=32,
                                                 class_mode='binary')
test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(128, 128),
                                            batch_size=32,
                                            class_mode='binary')

test_filenames = os.listdir("dataset/control_set")
test_df = pd.DataFrame({
    'filename': test_filenames
})
nb_samples = test_df.shape[0]

sample_test = test_df.head(18)
sample_test.head()

plt.figure(figsize=(6, 6))
for index, row in sample_test.iterrows():
    filename = row['filename']
    img = load_img("dataset/control_set/" + filename, target_size=(128, 128))
    test_image = image.load_img("dataset/control_set/" + filename, target_size=(128, 128))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = model.predict(test_image)
    if result[0][0] == 1:
        prediction = 'dog'
    else:
        prediction = 'cat'
    category = prediction
    axes = plt.subplot(6, 3, index + 1)
    plt.imshow(img)
    plt.xlabel(format(category))
    axes.set_yticks([])
    axes.set_xticks([])
plt.tight_layout()
plt.show()
