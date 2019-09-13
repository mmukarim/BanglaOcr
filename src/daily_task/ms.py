from os import listdir

import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential, load_model
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras.preprocessing.image import img_to_array, ImageDataGenerator

# Initializing necessary parameters
EPOCHS = 30
initial_learning_rate = 1e-3
batch_size = 32
default_image_size = tuple((256, 256))
directory_root = '../input/rice-diseases-image-dataset/labelledrice/'
input_width = 256
input_height = 256
input_depth = 3


# Converting images to np arrays
def convert_image_to_array(image_dir):
    try:
        image_from_directory = cv2.imread(image_dir)
        if image_from_directory is not None:
            image_from_directory = cv2.resize(image_from_directory, default_image_size)
            return img_to_array(image_from_directory)
        else:
            return np.array([])
    except Exception as e:
        print(f"Error : {e}")
        return None


# Creating image list and label list
image_list, image_label_list = [], []
try:
    print("[INFO] Loading images ...")
    root_dir = listdir(directory_root)

    for plant_folder in root_dir:
        plant_disease_folder_list = listdir(f"{directory_root}/{plant_folder}")

        for plant_disease_folder in plant_disease_folder_list:
            print(f"[INFO] Processing {plant_disease_folder} ...")
            rice_disease_image_list = listdir(f"{directory_root}/{plant_folder}/{plant_disease_folder}/")

            for image_from_list in rice_disease_image_list:
                image_directory = f"{directory_root}/{plant_folder}/{plant_disease_folder}/{image_from_list}"
                if image_directory.endswith(".jpg") or image_directory.endswith(".JPG"):
                    image_list.append(convert_image_to_array(image_directory))
                    image_label_list.append(plant_disease_folder)
    print("[INFO] Image loading completed")
except Exception as e:
    print(f"Error : {e}")

# Encoding string labels to integers
label_binarizer_for_image_labels = LabelBinarizer()
binarized_image_labels = label_binarizer_for_image_labels.fit_transform(image_label_list)
pickle.dump(label_binarizer_for_image_labels, open('label_transform.pkl', 'wb'))
n_classes = len(label_binarizer_for_image_labels.classes_)

# Normalizing images
normalized_image_list = np.array(image_list, dtype=np.float16) / 225.0

# Splitting image list in train and test
print("[INFO] Spliting data to train, test")
x_train, x_test, y_train, y_test = train_test_split(normalized_image_list, binarized_image_labels, test_size=0.30, random_state=42)

# Creating augmentation object
image_augmentation = ImageDataGenerator(
    rotation_range=30, width_shift_range=0.15,
    height_shift_range=0.15, shear_range=0.15,
    zoom_range=0.2, horizontal_flip=True,
    fill_mode="nearest")

# Proposed model building
proposed_model = Sequential()
input_shape = (input_height, input_width, input_depth)
channel_dimension = -1
if K.image_data_format() == "channels_first":
    inputShape = (input_depth, input_height, input_width)
    channel_dimension = 1
proposed_model.add(Conv2D(32, (3, 3), padding="same", input_shape=input_shape))
proposed_model.add(Activation("relu"))
proposed_model.add(BatchNormalization(axis=channel_dimension))
proposed_model.add(MaxPooling2D(pool_size=(3, 3)))
proposed_model.add(Dropout(0.25))
proposed_model.add(Conv2D(64, (3, 3), padding="same"))
proposed_model.add(Activation("relu"))
proposed_model.add(BatchNormalization(axis=channel_dimension))
proposed_model.add(Conv2D(64, (3, 3), padding="same"))
proposed_model.add(Activation("relu"))
proposed_model.add(BatchNormalization(axis=channel_dimension))
proposed_model.add(MaxPooling2D(pool_size=(2, 2)))
proposed_model.add(Dropout(0.25))
proposed_model.add(Conv2D(128, (3, 3), padding="same"))
proposed_model.add(Activation("relu"))
proposed_model.add(BatchNormalization(axis=channel_dimension))
proposed_model.add(Conv2D(128, (3, 3), padding="same"))
proposed_model.add(Activation("relu"))
proposed_model.add(BatchNormalization(axis=channel_dimension))
proposed_model.add(MaxPooling2D(pool_size=(2, 2)))
proposed_model.add(Dropout(0.25))
proposed_model.add(Flatten())
proposed_model.add(Dense(1024))
proposed_model.add(Activation("relu"))
proposed_model.add(BatchNormalization())
proposed_model.add(Dropout(0.5))
proposed_model.add(Dense(n_classes))
proposed_model.add(Activation("softmax"))

# Compiling model
optimizer = Adam(lr=initial_learning_rate, decay=initial_learning_rate / EPOCHS)
proposed_model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
print("[INFO] training network...")

# Training model
checkpoint_for_best_model = ModelCheckpoint('best_model.h5',
                                            verbose=1, monitor='acc',
                                            save_best_only=True, mode='auto')

model_history = proposed_model.fit_generator(
    image_augmentation.flow(x_train, y_train, batch_size=batch_size),
    validation_data=(x_test, y_test),
    steps_per_epoch=len(x_train) // batch_size,
    epochs=EPOCHS, verbose=1,
    callbacks=[checkpoint_for_best_model])

# Getting accuracy and loss list
train_acc = model_history.history['acc']
val_acc = model_history.history['val_acc']
train_loss = model_history.history['loss']
val_loss = model_history.history['val_loss']
epochs = range(1, len(train_acc) + 1)

# Plotting train and validation accuracy
plt.plot(epochs, train_acc, 'b', label='Training accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
plt.title('Training and Validation accuracy')
plt.legend()
plt.show()

# Plotting train and validation loss
plt.plot(epochs, train_loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and Validation loss')
plt.legend()
plt.show()

# Loading saved model and evaluating
best_trained_model = load_model('best_model.h5')
print("[INFO] Calculating model accuracy")
scores = best_trained_model.evaluate(x_test, y_test)
print(f"Test Accuracy: {scores[1] * 100}")
