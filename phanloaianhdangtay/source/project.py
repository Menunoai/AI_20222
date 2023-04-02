from keras.models import load_model
import keras.utils as image
import numpy as np
import keras
from keras_preprocessing.image import ImageDataGenerator
import keras_preprocessing
import tensorflow as tf
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os
import zipfile
rock_dir = os.path.join('setup/rps/rock')
paper_dir = os.path.join('setup/rps/paper')
scissors_dir = os.path.join('setup/rps/scissors')

print('total training rock images:', len(os.listdir(rock_dir)))
print('total training paper images:', len(os.listdir(paper_dir)))
print('total training scissors images:', len(os.listdir(scissors_dir)))

rock_files = os.listdir(rock_dir)

paper_files = os.listdir(paper_dir)
print(paper_files)

scissors_files = os.listdir(scissors_dir)


pic_index = 2

next_rock = [os.path.join(rock_dir, fname)
             for fname in rock_files[pic_index-2:pic_index]]
next_paper = [os.path.join(paper_dir, fname)
              for fname in paper_files[pic_index-2:pic_index]]
next_scissors = [os.path.join(scissors_dir, fname)
                 for fname in scissors_files[pic_index-2:pic_index]]

# for i, img_path in enumerate(next_rock+next_paper+next_scissors):
#   #print(img_path)
#   img = mpimg.imread(img_path)
#   plt.imshow(img)
#   plt.axis('Off')


TRAINING_DIR = "setup/rps"
training_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

VALIDATION_DIR = "setup/rps-test-set"
validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = training_datagen.flow_from_directory(
    TRAINING_DIR,
    target_size=(150, 150),
    class_mode='categorical',
    batch_size=36





)

validation_generator = validation_datagen.flow_from_directory(
    VALIDATION_DIR,
    target_size=(150, 150),
    class_mode='categorical',
    batch_size=36
)

model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 150x150 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The third convolution
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The fourth convolution
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])


model.summary()

model.compile(loss = 'categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

history = model.fit(train_generator, epochs=25, steps_per_epoch=20, validation_data = validation_generator, verbose = 1, validation_steps=3)

model.save("rps.h5")

import matplotlib.pyplot as plt

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()


plt.show()

model = load_model("rps.h5")
score = model.evaluate(validation_generator, verbose=0)
print("test core : ", score[0])
print('test accuracy: ', score[1])
testfiles = os.path.join("setup/result-test")
test_imgs = os.listdir(testfiles)
for i in test_imgs:
    path=testfiles + "/" + i
    img = mpimg.imread(path)
    plt.imshow(img)
    plt.show()
    img = image.load_img(path, target_size=(150, 150))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    images = np.vstack([x])
    classes = model.predict(images, batch_size=10)
    print(path)
    for i in range(0,3):
        if classes[0][i] == 1:
            if i==0:
                print("paper")
            elif i==1:
                print("rock")
            else:
                print("scissor")

