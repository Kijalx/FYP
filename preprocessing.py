import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os

def load_and_preprocess_image(file_path, target_size=(180, 180)):
    try:
        img = image.load_img(file_path, target_size=target_size)
        img_array = image.img_to_array(img)
        img_array /= 255.0
        return img_array
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

output_folder = "preprocessed_images"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

num_processed = 0
num_skipped = 0

for folder_name in ("2COVID", "3CAP", "adenocarcinoma", "CT_HEALTHY", "glioma_tumor",
           "large_cell_carcinoma", "meningioma_tumor", "MRI_HEALTHY",
           "NORMAL", "pituitary_tumor", "PNEUMONIA", "squamous_cell_carcinoma", "tumor"):
    input_folder = os.path.join("pictures", folder_name)
    output_subfolder = os.path.join(output_folder, folder_name)

    if not os.path.exists(output_subfolder):
        os.makedirs(output_subfolder)

    for fname in os.listdir(input_folder):
        fpath = os.path.join(input_folder, fname)
        img = load_and_preprocess_image(fpath, target_size=(180, 180))

        if img is not None:
            output_path = os.path.join(output_subfolder, f"{fname[:-4]}_preprocessed.jpg")
            tf.keras.preprocessing.image.save_img(output_path, img)
            num_processed += 1
        else:
            num_skipped += 1

print(f"Processed {num_processed} images. Skipped {num_skipped} images.")
