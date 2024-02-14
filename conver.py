from keras import layers, callbacks
import tensorflow as tf
from tensorflow import keras
from keras import layers

model = keras.models.load_model('image.keras')
modeljson = model.to_json()
