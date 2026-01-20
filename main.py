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



import tarfile
import urllib.request
import os
import shutil
import ssl


# Disable SSL verification globally
ssl._create_default_https_context = ssl._create_unverified_context

#url of the tar file
url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/ZjXM4RKxlBK9__ZjHBLl5A/aircraft-damage-dataset-v1.tar'

#define the path to save file
tar_filename = 'aircraft_damage_dataset_v1.tar'
extracted_folder = 'aircraft_damage_dataset_v1' #folder where contents will be extracted

#download the tar file
urllib.request.urlretrieve(url, tar_filename)
print(f'Downloaded {tar_filename}. Extraction will begin now.')

#check if the folder already exists
if os.path.exists(extracted_folder):
    print(f"The folder '{extracted_folder}' already exists. Removing the existing folder.")

    #remove the existing folder to avoid overwritting or duplication
    shutil.rmtree(extracted_folder)
    print(f'Removed the existing folder: {extracted_folder}')

#extract the contents of the tar file
with tarfile.open(tar_filename, 'r') as tar_ref:
    tar_ref.extractall()  #this will extract to the current directory
    print(f"Extracted {tar_filename} successfully.")






#define directories for train, test, and validation splits
extract_path = 'aircraft_damage_dataset_v1'
train_dir = os.path.join(extract_path, 'train')
test_dir = os.path.join(extract_path, 'test')
valid_dir = os.path.join(extract_path, 'valid')



#create image data generators to preprocess the data
train_datagen = ImageDataGenerator(rescale = 1./255)
valid_datagen = ImageDataGenerator(rescale = 1./255)
test_datagen = ImageDataGenerator(rescale = 1./255)


train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size = (img_rows, img_cols),  #resize images to the size VGG16 expects
    batch_size = batch_size,
    seed = seed_value,
    class_mode = 'binary',
    shuffle = True #binary classification: dent vs crack
)




valid_generator = valid_datagen.flow_from_directory(
    directory = valid_dir,
    class_mode = 'binary',
    seed = seed_value,
    batch_size = batch_size,
    shuffle = False,
    target_size = (img_rows, img_cols)
)




test_generator = test_datagen.flow_from_directory(
    directory = test_dir,
    class_mode = 'binary',
    seed = seed_value,
    batch_size = batch_size,
    shuffle = False,
    target_size = (img_rows, img_cols)
)



# Pre trained VGG16 model + higher level layers
base_model = VGG16(weights = 'imagenet', include_top = False, input_shape = (img_rows, img_cols, 3))



output = base_model.layers[-1].output
output = keras.layers.Flatten()(output)
base_model = Model(base_model.input, output)

#freeze the base VGG16 model layers
for layer in base_model.layers:
    layer.trainable = False



#built the custom model
model = Sequential()
model.add(base_model)
model.add(Dense(512, activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(512, activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation = 'sigmoid'))


model.compile(optimizer = Adam(learning_rate = 0.0001), loss = 'binary_crossentropy', metrics = ['accuracy'])


#train the model
history = model.fit(train_generator, #fill in wih the trainning data generator or dataset
                    epochs = n_epochs, #fill in with the no of epochs for training
                    validation_data = valid_generator) #fill in with the validation data generator or dataset



#access the training hiostory
train_history = model.history.history


# visualize training history
#plot loss curve for traininbg and validation sets
plt.title('Training Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.plot(train_history['loss'])
plt.show()

plt.title('Validation Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.plot(train_history['val_loss'])
plt.show()


# Model Evaluation
#evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_generator)
print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_accuracy}')

# Save the trained model to a file
model.save('aircraft_damage_classifier.h5')
print('Model saved as aircraft_damage_classifier.h5')

#functions to visualize predictions
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image

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
    predictions = model.predict(test_images)

    #in binary classification predictions are probabilities (float). convert to binary (0or1)
    predicted_classes = (predictions > 0.5).astype(int).flatten()

    #get the class indices from the test generator and invert them to get calss names
    class_indices = test_generator.class_indices
    class_names = {v: k for k, v in class_indices.items()}  #invert the dictionary

    #specify the image to display based on the index
    image_to_plot = test_images[index_to_plot]
    true_label = test_labels[index_to_plot]
    predicted_label = predicted_classes[index_to_plot]

    #plot the selected image with its true and predicted labels
    plot_image_with_title(image = image_to_plot, model = model, true_label=true_label, predicted_label = predicted_label, class_names = class_names)


# test
test_model_on_image(test_generator, model, index_to_plot = 10)

