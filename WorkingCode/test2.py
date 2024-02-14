import tensorflow as tf
import keras
from keras.utils import load_img, img_to_array

model = keras.models.load_model('../Working images/CTimage.keras')
img_size = (180, 180)
img = load_img("../Split/CT/adenocarcinoma/ad16_preprocessed.jpg", target_size=img_size)
img_array = img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

predictions = model.predict(img_array)
print(predictions)
prediction_for_first_image = predictions[0]

predicted_class = tf.argmax(prediction_for_first_image)
confidence = float(tf.reduce_max(tf.nn.softmax(prediction_for_first_image)))

print(f"Predicted class: {predicted_class}, Confidence: {confidence:.2f}")

'''
CHECKWHAT IMAGE
0 = CT
1 = MRI
2 = XRAY

XRAY IMAGE
0 = HEALTHY
1 = PNEUMONIA

MRI
0 = HEALTHY
1 = GLIOMA TUMOR
2 = MENGINGIOMA TUMOR
3 = PITUITARY TUMOR
4 = TUMOR

CT
0 = COVID
1 = CAP
2 = HEALTHY
3 = ADENOCARCINOMA
4 = LARGE CELL CARCINOMA
5 = SQUAMOUS CELL CARCINOMA
'''