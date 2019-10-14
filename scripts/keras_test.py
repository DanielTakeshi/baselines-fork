"""Testing keras.
"""
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import sys

# Load the image.
img_path = 'data/elephant.jpeg'
img = image.load_img(img_path, target_size=(223, 223))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)  # shape (1,224,224,3)
x = preprocess_input(x)

# The basic full model
model = ResNet50(weights='imagenet', include_top=False)
preds = model.predict(x)
print(preds.shape)
#print('Predicted:', decode_predictions(preds, top=4)[0])

# Make a session here
sess = tf.Session()
sess.graph.finalize()

# Predict, and decode the results into a list of tuples (class, description,
# probability) (one such list for each sample in the batch)
preds = model.predict(x)
print('Predicted:', decode_predictions(preds, top=4)[0])





## Now see what happens when we don't load everything.
#print('\nNow see what happens when we do not include the top layer')
#model = ResNet50(weights='imagenet', include_top=False)
#features = model.predict(x)
##print(features)
#print(features.shape) # (1,7,7,2048)
#print(type(features))
