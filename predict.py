import os
import random
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
import matplotlib.pyplot as plt

# Load the trained model
model = load_model('aircraft_damage_classifier.h5')

# Path to test image
img_path = 'aircraft_damage_dataset_v1/test/dent/144_10_JPG_jpg.rf.4d008cc33e217c1606b76585469d626b.jpg'

# Load and preprocess the image
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

# Predict
pred = model.predict(img_array)
print("Prediction probabilities:", pred)
predicted_class = np.argmax(pred, axis=1)[0]

# Show image and prediction
plt.imshow(img)
plt.title(f'Predicted class: {predicted_class}')
plt.axis('off')
plt.show()