import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import itertools
import keras
import tensorflow as tf

model = keras.models.load_model('image.keras')

y_true = []
y_pred_probs = []
image_size = (180, 180)
batch_size = 64
val_ds = tf.keras.utils.image_dataset_from_directory(
    "preprocessed_images",
    validation_split=0.9,
    subset="validation",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)

for images, labels in val_ds:
    y_true.extend(labels.numpy())
    y_pred_probs.extend(model.predict(images).argmax(axis=1))

y_pred = np.array(y_pred_probs)

classes = ["2COVID", "3CAP", "adenocarcinoma", "CT_HEALTHY", "glioma_tumor",
           "large_cell_carcinoma", "meningioma_tumor", "MRI_HEALTHY",
           "NORMAL", "pituitary_tumor", "PNEUMONIA", "squamous_cell", "tumor"]

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 10))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion matrix')
plt.colorbar()
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

print("Classification Report:\n", classification_report(y_true, y_pred, target_names=classes))