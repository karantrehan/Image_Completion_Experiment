from __future__ import print_function

import tensorflow as tf
import keras.backend as K
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, Deconv2D
import scipy

import matplotlib.pyplot as plt
import cv2

import os

project_dir = os.getcwd()

sess = tf.Session()
K.set_session(sess)


def completion_function_eval(original, mask):
    generated_image = generation(original)

    return generated_image


with tf.name_scope('Generator_Model'):
    generation = Sequential(name='Generation_Model')
    generation.add(Conv2D(filters=64, kernel_size=(5,5), activation='relu',strides=(1,1), input_shape=(256,256,3), padding='same', name='Conv_1'))
    generation.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu', strides=(2,2), name='Conv_2', padding='same'))
    generation.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu', strides=(1,1), name='Conv_3', padding='same'))
    generation.add(Conv2D(filters=256, kernel_size=(3,3), activation='relu', strides=(2,2), name='Conv_4', padding='same'))
    generation.add(Conv2D(filters=256, kernel_size=(3,3), activation='relu', strides=(1,1), name='Conv_5', padding='same'))
    generation.add(Conv2D(filters=256, kernel_size=(3,3), activation='relu', strides=(1,1), name='Conv_6', padding='same'))
    generation.add(Conv2D(filters=256, kernel_size=(3,3), activation='relu', strides=(1,1), name='Dilated_Conv_1', dilation_rate=2, padding='same'))
    generation.add(Conv2D(filters=256, kernel_size=(3,3), activation='relu', strides=(1,1), name='Dilated_Conv_2', dilation_rate=4, padding='same'))
    generation.add(Conv2D(filters=256, kernel_size=(3,3), activation='relu', strides=(1,1), name='Dilated_Conv_3', dilation_rate=8, padding='same'))
    generation.add(Conv2D(filters=256, kernel_size=(3,3), activation='relu', strides=(1,1), name='Dilated_Conv_4', dilation_rate=16, padding='same'))
    generation.add(Conv2D(filters=256, kernel_size=(3,3), activation='relu', strides=(1,1), name='Conv_7', padding='same'))
    generation.add(Conv2D(filters=256, kernel_size=(3,3), activation='relu', strides=(1,1), name='Conv_8', padding='same'))
    generation.add(Deconv2D(filters=128, kernel_size=(4,4), activation='relu', strides=(2,2), name='DeConv_1', padding='same'))
    generation.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu', strides=(1,1), name='Conv_9', padding='same'))
    generation.add(Deconv2D(filters=64, kernel_size=(4,4), activation='relu', strides=(2,2), name='DeConv_2', padding='same'))
    generation.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu', strides=(1,1), name='Conv_10', padding='same'))
    generation.add(Conv2D(filters=3, kernel_size=(3,3), strides=(1,1), padding='same', name='Output'))


with tf.name_scope('ph_patched_batch'):
    patched_batch_placeholder = K.placeholder(shape=(None, 256, 256, 3), dtype=tf.float32, name='ph_patched_batch')


with tf.name_scope('ph_mask_'):
    mask_placeholder = K.placeholder(shape=(None, 256, 256, 3), dtype=tf.float32, name='ph_mask')

with tf.name_scope('Generator_Network'):
    generated_batch = generation(patched_batch_placeholder)
    with tf.name_scope('Generated_batchX1000'):
        generated_batch = generated_batch*1


loss_dir_name = 'main_session'


init_op = tf.global_variables_initializer()
sess.run(init_op)


saver = tf.train.Saver(keep_checkpoint_every_n_hours=0.1)

# # Restoring the main Session if it exists


if os.path.isfile('saved_sessions/' + loss_dir_name + '/session.index'):
    saver.restore(sess, 'saved_sessions/' + loss_dir_name + '/session')

else:
    print('No session found!!')

# Testing with van data

mask = plt.imread(os.getcwd()+'/data/custom_data/original/mask.png')
image = plt.imread(os.getcwd()+'/data/custom_data/original/image.png')


image = cv2.resize(image, (256,256))
mask = cv2.resize(mask, (256,256))
image= np.expand_dims(image,0)
mask = np.expand_dims(mask, 0)


mean = 0

mean_image = np.ones(np.shape(image))*mean

feed_image = mean_image*mask

feed_image = (1-mask)*image + (mask)*feed_image

test_custom = completion_function_eval(patched_batch_placeholder, mask_placeholder)

with sess.as_default():
    test_custom_value = test_custom.eval(feed_dict={patched_batch_placeholder: feed_image,
                                                    mask_placeholder: mask})

image = (1-mask)*feed_image + mask*test_custom_value[0]

scipy.misc.imsave('data/custom_data/image_result.png', image[0, :, :, :])
