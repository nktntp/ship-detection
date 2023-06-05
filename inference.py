import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from PIL import Image 
import numpy as np


def validate_arguments():
    if len(sys.argv) != 3:
        raise ValueError("Exactly two arguments are required.")
    
    first_arg = sys.argv[1]
    second_arg = sys.argv[2]
    
    if first_arg != "--input":
        raise ValueError("First argument must be '--input'.")
    
    if not second_arg:
        raise ValueError("Second argument cannot be empty.")


validate_arguments()


def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    return K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)


def dice_p_bce(in_gt, in_pred):
    return 1e-3 * binary_crossentropy(in_gt, in_pred) - dice_coef(in_gt, in_pred)


def true_positive_rate(y_true, y_pred):
    return K.sum(K.flatten(y_true) * K.flatten(K.round(y_pred))) / K.sum(y_true)


model = keras.models.load_model('model.h5', custom_objects={"dice_p_bce": dice_p_bce,
                                                            "dice_coef": dice_coef,
                                                            "true_positive_rate": true_positive_rate})
image_folder = sys.argv[2]
TARGET_FOLDER = ""
for filename in os.listdir(image_folder):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        image_path = os.path.join(image_folder, filename)
        image = tf.io.read_file(image_path)
        image = tf.cond(tf.io.is_jpeg(image),
                        lambda: tf.io.decode_jpeg(image, channels=3),
                        lambda: tf.io.decode_png(image, channels=3))
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.expand_dims(image, axis=0)
        predicted_image = model.predict(image, verbose=0)
        I = np.squeeze(predicted_image[0])
        I8 = (((I - I.min()) / (I.max() - I.min())) * 255.9).astype(np.uint8)
        im = Image.fromarray(I8)
        im.save(TARGET_FOLDER + f"{filename[:-4]}_prediction.png")