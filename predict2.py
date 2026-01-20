import os
import random
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import zipfile
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten
from keras.applications import VGG16
from keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing import image
import random


# Load the trained model
model = load_model('aircraft_damage_classifier.h5')

#set seed for reproducibility
seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)



#set the batch size, epochs
batch_size = 32
n_epochs = 5
img_rows, img_cols = 224, 224
input_shape = (img_rows, img_cols, 3)

#define directories for train, test, and validation splits
extract_path = 'aircraft_damage_dataset_v1'
test_dir = os.path.join(extract_path, 'test')

test_datagen = ImageDataGenerator(rescale = 1./255)

test_generator = test_datagen.flow_from_directory(
    directory = test_dir,
    class_mode = 'binary',
    seed = seed_value,
    batch_size = batch_size,
    shuffle = False,
    target_size = (img_rows, img_cols)
)

print(test_generator.class_indices)

#function to plot a single image and its prediction
def plot_image_with_title (image, model, true_label, predicted_label, class_names):
    plt.figure(figsize = (6,6))
    plt.imshow(image)


    #convert labels from one hot to class indices if neede, but for binary labels it's just 0 or 1
    true_label_name = class_names[true_label] #labels are alreadyb in class indices
    pred_label_name = class_names[predicted_label] #predictions are 0 or 1

    plt.title(f"True: {true_label_name}\nPred: {pred_label_name}")
    plt.axis('off')
    plt.show()

#function to test the model with images from the test set
def test_model_on_image(test_generator, model, index_to_plot = 0):
    #get a batch of images and labels from the test generator
    test_images, test_labels = next(test_generator)
    print(f"Test images batch shape: {test_images.shape}")
    print(f"Test labels batch shape: {test_labels.shape}")

    #make predictions on the batch
    print("Image path: ", test_generator.filepaths[index_to_plot])
    predictions = model.predict(test_images)

    #in binary classification predictions are probabilities (float). convert to binary (0or1)
    predicted_classes = (predictions > 0.5).astype(int).flatten()

    #get the class indices from the test generator and invert them to get calss names
    class_indices = test_generator.class_indices
    print("Predicted labels: ", predicted_classes)
    class_names = {v: k for k, v in class_indices.items()}  #invert the dictionary

    #specify the image to display based on the index
    image_to_plot = test_images[index_to_plot]
    true_label = test_labels[index_to_plot]
    predicted_label = predicted_classes[index_to_plot]
    print(f"True label: {true_label}")
    print(f"Predicted label: {predicted_label}")

    #plot the selected image with its true and predicted labels
    plot_image_with_title(image = image_to_plot, model = model, true_label=true_label, predicted_label = predicted_label, class_names = class_names)


# test
test_model_on_image(test_generator, model, index_to_plot = 1)