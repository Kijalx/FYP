
import os

from PIL import Image
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow import keras

# Replace 'path/to/your/image.jpg' with the actual path to your image file
image_path = '../test/1.jpg'

# Check if the file exists
if os.path.isfile(image_path):
    # Load the Keras model
    model = keras.models.load_model("save_at_5.keras")

    # Open the image using PIL
    image = Image.open(image_path)

    # Resize the image to match the input size of the model
    image = image.resize((180, 180))

    # Convert the image to a NumPy array
    image_array = np.array(image) / 255.0  # Normalize the pixel values

    # Expand the dimensions to match the model's expected input shape
    input_data = np.expand_dims(image_array, axis=0)

    # Make a prediction using the model
    predictions = model.predict(input_data)

    # Get the predicted class (assuming binary classification)
    predicted_class = predictions[0]

    # Display the image and prediction
    plt.imshow(image)
    plt.title(f"Predicted: {predicted_class}")
    plt.axis('off')
    plt.show()

else:
    print(f"File not found: {image_path}")
    '''
import itertools

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix, classification_report

model = keras.models.load_model("save_at_5.keras")

num_skipped = 0
for folder_name in ("Cat", "Dog"):
    folder_path = os.path.join("PetImages", folder_name)
    for fname in os.listdir(folder_path):
        fpath = os.path.join(folder_path, fname)
        try:
            fobj = open(fpath, "rb")
            is_jfif = tf.compat.as_bytes("JFIF") in fobj.peek(10)
        finally:
            fobj.close()

        if not is_jfif:
            num_skipped += 1
            os.remove(fpath)

print("Deleted %d images" % num_skipped)

image_size = (180, 180)
batch_size = 128

train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
    "PetImages",
    validation_split=0.2,
    subset="both",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)

data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
    ]
)

augmented_train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))
train_ds = train_ds.map(lambda img, label: (data_augmentation(img), label), num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

# Evaluate the model on the validation set
y_true = []
y_pred_probs = []

for images, labels in val_ds:
    y_true.extend(labels.numpy())
    y_pred_probs.extend(model.predict(images).flatten())

y_pred = np.round(y_pred_probs)

# Print confusion matrix
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:\n", cm)

# Plot confusion matrix
plt.figure(figsize=(8, 8))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion matrix')
plt.colorbar()
classes = ["Cat", "Dog"]
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, f"{cm[i, j]}", horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# Print classification report
print("Classification Report:\n", classification_report(y_true, y_pred))
'''