import tensorflow as tf
import keras
from keras.utils import load_img, img_to_array

model = keras.models.load_model('image.keras')
img_size = (180, 180)
img = load_img("pictures/CT_HEALTHY/10.png", target_size=img_size)
img_array = img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

predictions = model.predict(img_array)
print(predictions)
prediction_for_first_image = predictions[0]

predicted_class = tf.argmax(prediction_for_first_image)
confidence = float(tf.reduce_max(tf.nn.softmax(prediction_for_first_image)))

print(f"Predicted class: {predicted_class}, Confidence: {confidence:.2f}")
