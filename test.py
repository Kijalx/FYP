import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow import keras
import os

model = keras.models.load_model('model.keras')
folder_path = r"C:\Users\aleks\PycharmProjects\scientificProject4\test"

class_labels = ['dog', 'cat']
threshold = 0.8
image_size = (180, 180)

predicted_covid = []
predicted_normal = []
unrecognized = []

for filename in os.listdir(folder_path):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(folder_path, filename)

        img = keras.utils.load_img(img_path, target_size=(180, 180))
        img_array = keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)

        predictions = model.predict(img_array)
        confidence = predictions[0, 0]

        if confidence > threshold:
            predicted_covid.append((filename, confidence))
        else:
            if 1 - confidence > threshold:
                predicted_normal.append((filename, 1 - confidence))
            else:
                unrecognized.append((filename, confidence))

def display_results(category, results):
    print(f"**Predicted {category.capitalize()}:**\n")
    print("| Image      | Confidence |")
    print("|------------|------------|")
    for result in results:
        print(f"| {result[0]:<10} | {100 * result[1]:.2f}%       |")
    print("\n")

display_results("dog", predicted_covid)
display_results("cat", predicted_normal)
display_results("unrecognized", unrecognized)

img = keras.utils.load_img(
    "test/monkey.jpg", target_size=image_size
)
plt.imshow(img)

img_array = keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

predictions = model.predict(img_array)
score = float(predictions[0])
print(f"This image is {100 * (1 - score):.2f}% cat and {100 * score:.2f}% dog.")