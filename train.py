import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import skimage.io
import os

import warnings
warnings.filterwarnings("ignore")
import gc; gc.enable()

from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from typing import Tuple
import tensorflow as tf


PATH = "../input/airbus-ship-detection/"
TRAIN_PATH = PATH + "train_v2/"
TEST_PATH = PATH + "test_v2/"

def rle_encode(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def rle_decode(mask_rle, shape=(768, 768)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background
    '''
    if type(mask_rle) != str:
        return np.zeros(shape)
    else:
        s = mask_rle.split()
        starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
        starts -= 1
        ends = starts + lengths
        img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
        for lo, hi in zip(starts, ends):
            img[lo:hi] = 1
        return img.reshape(shape).T


masks = pd.read_csv(PATH + "train_ship_segmentations_v2.csv")

masks['count'] = masks.groupby('ImageId')['EncodedPixels'].transform('count')
masks.set_index('ImageId', inplace = True)
mu = masks[masks['count'] <= 1]
mm = masks[masks['count'] > 1]
n = 0 
for mult in mm.index.unique():
    if n % 1000 == 0 : print(n)
    combined = np.zeros((768, 768))
    dfm = mm.loc[mult,:]
    for i, row in dfm.iterrows():
        d = rle_decode(row['EncodedPixels'])
        combined = np.maximum(combined, d)
    new_row = row
    new_row['EncodedPixels'] = rle_encode(combined)
    mu = mu.append(new_row)
    n += 1

mu.reset_index(inplace = True)

mu['file_size_kb'] = mu['ImageId'].map(lambda c_img_id: 
                                        os.stat(os.path.join(PATH+'train_v2/', 
                                        c_img_id)).st_size/1024)
mu = mu[mu['file_size_kb']>50]

mu2 = mu
df_train, df_test = train_test_split(mu2, test_size = 0.5, shuffle=True, random_state = 51)

weights = [0.001,0.02,0.1,4,5,6,7,8,9,10,11,12,13,14,15,16]

df_train['weights'] = df_train['count'].map(lambda c : weights[c])
df_train['freq'] = df_train.groupby('count')['count'].transform('count')

df_sample = df_train.sample(n = 10000, weights = df_train['weights'], random_state = 17)
df_sample_test = df_test.sample(n = 10000, random_state = 24)

def image_gen(df, show = False, print_name = False):
    images = list(df['ImageId'])
    masks = list(df['EncodedPixels'])
    i = 0
    while i < len(images):
        if print_name:
            print(images[i])
        img = skimage.io.imread(TRAIN_PATH + images[i])
        mask = np.reshape(rle_decode(masks[i]),(768,768,1))
        if show == True:
            plt.imshow(img)
            plt.imshow(mask, alpha = 0.4)
        yield (np.array([img/255]), np.array([mask]))
        i += 1
        
def image_gen_batch(df, batch_size = 1):
    images = np.array(df['ImageId'])
    masks = np.array(df['EncodedPixels'])
    ind = np.arange(len(images))
    np.random.shuffle(ind)
    images = images[ind]
    masks = masks[ind]
    i = 0
    while True:
        batch_img = []
        batch_mask = []
        for x in np.arange(batch_size):
            if i == len(images):
                i = 0
                np.random.shuffle(ind)
                images = images[ind]
                masks = masks[ind]
            batch_img.append(skimage.io.imread(TRAIN_PATH + images[i])/255.0)
            batch_mask.append(np.reshape(rle_decode(masks[i]),(768,768,1)))
            i += 1
        yield (np.array(batch_img), np.array(batch_mask))


dg_args = dict(featurewise_center = False, 
                  samplewise_center = False,
                  rotation_range = 15, 
                  width_shift_range = 0.1, 
                  height_shift_range = 0.1, 
                  shear_range = 0.01,
                  zoom_range = [0.5, 1.5],  
                  horizontal_flip = True, 
                  vertical_flip = True,
                  fill_mode = 'reflect',
                  data_format = 'channels_last',
                  brightness_range = [0.5, 1.5])

im_gen = ImageDataGenerator(**dg_args)

dg_args.pop('brightness_range')
mask_gen = ImageDataGenerator(**dg_args)


def aug_gen(in_gen):
    seed = 57
    for in_x, in_y in in_gen:
        g_x = im_gen.flow(255*in_x, 
                             batch_size = in_x.shape[0], 
                             seed = seed, 
                             shuffle=False)
        g_y = mask_gen.flow(in_y, 
                             batch_size = in_y.shape[0], 
                             seed = seed, 
                             shuffle=False)
        yield next(g_x)/255.0, next(g_y)

        
def aug_gen2(in_gen):
    seed = np.random.choice(range(9999))
    for in_x, in_y in in_gen:
        seed = np.random.choice(range(9999))
        img_aug = []
        mask_aug = []
        for i in np.arange(np.shape(in_x)[0]):
            seed = np.random.choice(range(9999))
            g_x = im_gen.flow(255.0*np.array([in_x[i]]), 
                                 batch_size = 1, 
                                 seed = seed, 
                                 shuffle=True)
            g_y = mask_gen.flow(np.array([in_y[i]]), 
                                 batch_size = 1, 
                                 seed = seed, 
                                 shuffle=True)
            img_aug.append(next(g_x)/255.0)
            mask_aug.append(next(g_y))
        yield np.array(img_aug)[:,0,:,:,:], np.array(mask_aug)[:,0,:,:,:]
        

im_gen2 = ImageDataGenerator(**dg_args)

        
gc.collect()

def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    return K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)


def dice_p_bce(in_gt, in_pred):
    return 1e-3*binary_crossentropy(in_gt, in_pred) - dice_coef(in_gt, in_pred)


def true_positive_rate(y_true, y_pred):
    return K.sum(K.flatten(y_true)*K.flatten(K.round(y_pred)))/K.sum(y_true)


def conv2d_block(input_tensor, filters: int, kernel_size: int = 3):
    x = input_tensor
    for i in range(2):
        x = tf.keras.layers.Conv2D(filters=filters,
                                   kernel_size=(kernel_size, kernel_size),
                                   kernel_initializer="he_normal",
                                   padding="same")(x)
        x = tf.keras.layers.Activation("relu")(x)
    return x


def encoder_block(inputs, filters: int, pool_size: Tuple[int, int], dropout: int):
    f = conv2d_block(inputs, filters=filters)
    p = tf.keras.layers.MaxPooling2D(pool_size=pool_size)(f)
    p = tf.keras.layers.Dropout(dropout)(p)
    return f, p


def encoder(inputs):
    f1, p1 = encoder_block(inputs, filters=64, pool_size=(2, 2), dropout=0.2)
    f2, p2 = encoder_block(p1, filters=128, pool_size=(2, 2), dropout=0.2)
    f3, p3 = encoder_block(p2, filters=256, pool_size=(2, 2), dropout=0.5) 
    f4, p4 = encoder_block(p3, filters=512, pool_size=(2, 2), dropout=0.6)
    return p4, (f1, f2, f3, f4)


def bottleneck(inputs):
    bottle_neck = conv2d_block(inputs, filters=1024)
    return bottle_neck


def decoder_block(inputs, conv_output, filters=64, kernel_size=3, strides=3, dropout=0.3):
    u = tf.keras.layers.Conv2DTranspose(filters=filters,
                                        kernel_size=kernel_size,
                                        strides=strides,
                                        padding="same")(inputs)
    c = tf.keras.layers.concatenate([u, conv_output])
    c = tf.keras.layers.Dropout(dropout)(c)
    c = conv2d_block(c, filters=filters, kernel_size=3)
    return c


def decoder(inputs, convs, output_channels):
    f1, f2, f3, f4 = convs
    c6 = decoder_block(inputs, f4, filters=512, kernel_size=(3, 3), strides=(2, 2), dropout=0.3)
    c7 = decoder_block(c6, f3, filters=256, kernel_size=(3, 3), strides=(2, 2), dropout=0.3)
    c8 = decoder_block(c7, f2, filters=128, kernel_size=(3, 3), strides=(2, 2), dropout=0.3)
    c9 = decoder_block(c8, f1, filters=64, kernel_size=(3, 3), strides=(2, 2), dropout=0.3)
    outputs = tf.keras.layers.Conv2D(output_channels, (1, 1), activation='sigmoid')(c9)
    return outputs


def unet():
    inputs = tf.keras.layers.Input(shape=(768, 768, 3,))
    encoder_output, convs = encoder(inputs)
    bottle_neck = bottleneck(encoder_output)
    outputs = decoder(bottle_neck, convs, output_channels=1)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


model = unet()
model.compile(optimizer=tf.keras.optimizers.AdamW(1e-4), loss=dice_p_bce, metrics=[dice_coef, 'binary_accuracy', true_positive_rate])

checkpoint = tf.keras.callbacks.ModelCheckpoint("checkpoints_v%d.out" % v, monitor='val_dice_coef', 
                             verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
EPOCHS = 5
STEPS_PER_EPOCH = 500
BATCH_SIZE = 5
results = model.fit_generator(aug_gen2(image_gen_batch(df_sample, batch_size = BATCH_SIZE)),
                          epochs = EPOCHS, 
                          steps_per_epoch = STEPS_PER_EPOCH,
                          validation_data = image_gen_batch(df_sample_test, batch_size = BATCH_SIZE), 
                          validation_steps = STEPS_PER_EPOCH,
                          callbacks = callbacks_list)
model.save('model.h5')