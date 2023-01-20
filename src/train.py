from __future__ import print_function

import tensorflow as tf
import keras.backend as K
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, Deconv2D, Flatten, Dense, Concatenate
import scipy

import pickle
import os

# project_dir = "/usr/src/app"
project_dir = os.getcwd()


sess = tf.Session()
K.set_session(sess)


batch_size = 12
batch_size_validation = 12
learning_rate = 0.0001
weight_GAN = 0.04
weight_L1 = 1
momentum = 0.9

total_no_of_iterations = 200
checkpoints = 40


def pickle_dump(obj, path):
    filename = path + '.pickle'
    pickle_out = open(filename, 'wb')
    pickle.dump(obj, pickle_out)
    pickle_out.close()

    return None


def pickle_load(filename):
    pickle_in = open(filename, 'rb')

    return pickle.load(pickle_in)


def apply_square_patch(images, h, w):
    """ This function applies a square patch """
    patched_images = images
    patched_images[:, h:h+64, w:w+64, 0] = 0
    patched_images[:, h:h+64, w:w+64, 1] = 0
    patched_images[:, h:h+64, w:w+64, 2] = 0

    return patched_images


def completion_function(arg_patched_batch, arg_generated_batch, arg_mask, arg_batch_size):
    """ This function adds the network in-painted patch and the rest of the image from the original file """

    with tf.name_scope('Ops_Completion'):
        completed_batch_patch_area = tf.multiply(arg_mask, arg_generated_batch)
        negative_mask = 1 - arg_mask
        completed_batch_outside_patch_area = tf.multiply(negative_mask, arg_patched_batch)
        completed_batch = tf.add(completed_batch_patch_area, completed_batch_outside_patch_area)
        completed_batch = tf.reshape(completed_batch, shape=(arg_batch_size, 256, 256, 3))

    return completed_batch


def surrounding_patch(arg_completed_batch, arg_mask_local_dis, arg_batch_size):
    """ This function cuts a 128*128 area around the 64*64 patch for Local Discriminator part """

    with tf.name_scope('Ops_Surrounding_Patch'):
        completed_batch_surrounding_patch = tf.boolean_mask(arg_completed_batch, arg_mask_local_dis)
        completed_batch_surrounding_patch = tf.reshape(completed_batch_surrounding_patch, shape=(arg_batch_size,128, 128,3))

    return completed_batch_surrounding_patch


# Generator network using the Keras sequential API

with tf.name_scope('Generator_Model'):
    generation = Sequential(name='Generation_Model')
    generation.add(Conv2D(filters=64, kernel_size=(5,5), activation='relu', strides=(1,1), input_shape=(256,256,3), padding='same', name='Conv_1'))
    generation.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu', strides=(2,2), name='Conv_2', padding='same'))
    generation.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu', strides=(1,1), name='Conv_3', padding='same'))
    generation.add(Conv2D(filters=256, kernel_size=(3,3), activation='relu', strides=(2,2), name='Conv_4', padding='same'))
    generation.add(Conv2D(filters=256, kernel_size=(3,3), activation='relu',  strides=(1,1), name='Conv_5', padding='same'))
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

with tf.name_scope('ph_original_batch'):
    original_batch_placeholder = K.placeholder(shape=(batch_size, 256, 256, 3), dtype=tf.float32, name='ph_original_batch')

with tf.name_scope('ph_mask_'):
    mask_placeholder = K.placeholder(shape=(None, 256, 256, 3), dtype=tf.float32, name='ph_mask')

with tf.name_scope('ph_mask_local_dis'):
    mask_local_dis_placeholder = K.placeholder(shape=(None, 256, 256, 3), dtype=tf.bool, name='ph_mask_local_dis')

with tf.name_scope('Generator_Network'):
    with tf.name_scope('Generated_batch'):
        generated_batch = generation(patched_batch_placeholder)

        # generated_batch = generated_batch

completed_batch = completion_function(patched_batch_placeholder, generated_batch, mask_placeholder, batch_size)

completed_batch_surrounding_patch = surrounding_patch(completed_batch, mask_local_dis_placeholder, batch_size)

# An Image Generator class object with pre-processing parameters given as arguments

image_gen = ImageDataGenerator(featurewise_center=False,
                               samplewise_center=False,
                               featurewise_std_normalization=False,
                               samplewise_std_normalization=False,
                               zca_whitening=False,
                               zca_epsilon=1e-6,
                               rotation_range=0.,
                               width_shift_range=0.,
                               height_shift_range=0.,
                               shear_range=0.,
                               zoom_range=0.,
                               channel_shift_range=0.,
                               fill_mode=None,
                               cval=0.,
                               horizontal_flip=False,
                               vertical_flip=False,
                               rescale=1.0/255.0,
                               preprocessing_function=None,
                               data_format=K.image_data_format())


def generator_loss(arg_generated_batch, arg_original_batch, arg_mask):
    """ This calculates L1 loss over generated image and original image """

    diff = tf.subtract(arg_generated_batch, arg_original_batch)
    diff = arg_mask*diff
    loss = tf.reduce_mean(tf.abs(diff))

    return loss


# Global and Local Discriminator network using Keras Sequential API


with tf.name_scope('Discriminator_Model'):
    global_discriminator = Sequential(name='Global_Dicriminator_Model')
    global_discriminator.add(Conv2D(64,  kernel_size=(5,5), strides=(2,2), activation='relu', name='Global_Dis_Conv_1', padding='same', input_shape=(256,256,3)))
    global_discriminator.add(Conv2D(128, kernel_size=(5,5), strides=(2,2), activation='relu', name='Global_Dis_Conv_2', padding='same'))
    global_discriminator.add(Conv2D(256, kernel_size=(5,5), strides=(2,2), activation='relu', name='Global_Dis_Conv_3', padding='same'))
    global_discriminator.add(Conv2D(512, kernel_size=(5,5), strides=(2,2), activation='relu', name='Global_Dis_Conv_4', padding='same'))
    global_discriminator.add(Conv2D(512, kernel_size=(5,5), strides=(2,2), activation='relu', name='Global_Dis_Conv_5', padding='same'))
    global_discriminator.add(Conv2D(512, kernel_size=(5,5), strides=(2,2), activation='relu', name='Global_Dis_Conv_6', padding='same'))
    global_discriminator.add(Flatten())
    global_discriminator.add(Dense(1024, activation='relu'))

    local_discriminator = Sequential(name='Local_Discriminator_Model')
    local_discriminator.add(Conv2D(64,  kernel_size=(5,5), strides=(2,2), activation='relu', name='local_Dis_Conv_1', padding='same', input_shape=(128,128,3)))
    local_discriminator.add(Conv2D(128, kernel_size=(5,5), strides=(2,2), activation='relu', name='local_Dis_Conv_2', padding='same'))
    local_discriminator.add(Conv2D(256, kernel_size=(5,5), strides=(2,2), activation='relu', name='local_Dis_Conv_3', padding='same'))
    local_discriminator.add(Conv2D(512, kernel_size=(5,5), strides=(2,2), activation='relu', name='local_Dis_Conv_4', padding='same'))
    local_discriminator.add(Conv2D(512, kernel_size=(5,5), strides=(2,2), activation='relu', name='local_Dis_Conv_5', padding='same'))
    local_discriminator.add(Flatten())
    local_discriminator.add(Dense(1024, activation='relu'))

# Concatenating the 1024 and 1024 dimension vectors from the Local and Global Discriminator Network

concat_layer = Concatenate(axis=-1)

concat_dense_layer = Dense(1, activation=None)


def discriminator_function(arg_image_batch, arg_mask, arg_batch_size):
    """ This function calculates probability of image being real """

    with tf.name_scope('Discriminator_Network'):
        image_batch_surrounding_patch = surrounding_patch(arg_image_batch, arg_mask, arg_batch_size)
        local_discriminator_fc = local_discriminator(image_batch_surrounding_patch)
        global_discriminator_fc = global_discriminator(arg_image_batch)
        concat_value = concat_layer([local_discriminator_fc, global_discriminator_fc])
        probability_real = concat_dense_layer(concat_value)

    return probability_real


D_logit_original = discriminator_function(original_batch_placeholder, mask_local_dis_placeholder, batch_size)
D_logit_completed = discriminator_function(completed_batch, mask_local_dis_placeholder, batch_size)

trainable_vars = tf.trainable_variables()

# Variables to be trained during the Generator part training

Gen_trainable_vars = []

[Gen_trainable_vars.append(var) for var in trainable_vars if 'Gen' in var.name]

# Variables to be trained during the Discriminator part training

Dis_trainable_vars = []
[Dis_trainable_vars.append(var) for var in trainable_vars if 'Dis' in var.name]


with tf.name_scope('loss_Gen_L1'):
    loss_Gen_L1_raw = generator_loss(generated_batch, original_batch_placeholder, mask_placeholder)
    loss_Gen_L1 = weight_L1 * loss_Gen_L1_raw


with tf.name_scope('loss_Gen_GAN'):
    loss_Gen_GAN = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(D_logit_completed),
                                                                          logits=D_logit_completed))

with tf.name_scope('loss_Dis'):
    with tf.name_scope('loss_Dis_cross_entropy_original'):
        loss_Dis_cross_entropy_original = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(D_logit_original),
                                                                                                 logits=D_logit_original))

    with tf.name_scope('loss_Dis_cross_entropy_completed'):
        loss_Dis_cross_entropy_completed = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(D_logit_completed),
                                                                                                  logits=D_logit_completed))

    loss_Dis = loss_Dis_cross_entropy_original + loss_Dis_cross_entropy_completed


with tf.name_scope('loss_Gen'):
    loss_Gen = weight_L1 * loss_Gen_L1_raw + weight_GAN * loss_Gen_GAN


with tf.name_scope('train_op_Gen_L1'):
    Gen_L1_Optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                              beta1=momentum)
    train_op_Gen_L1 = Gen_L1_Optimizer.minimize(loss_Gen_L1, var_list=Gen_trainable_vars)


with tf.name_scope('train_op_Dis'):
    Dis_Optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                           beta1=momentum)
    train_op_Dis = Dis_Optimizer.minimize(loss_Dis, var_list=Dis_trainable_vars)


with tf.name_scope('train_op_Gen_L1_GAN'):
    Gen_L1_GAN = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                        beta1=momentum)
    train_op_Gen_L1_GAN = Gen_L1_GAN.minimize(loss_Gen, var_list=Gen_trainable_vars)


tf.summary.scalar('loss_Gen_L1', loss_Gen_L1)
tf.summary.scalar('loss_Gen_GAN', loss_Gen_GAN)
tf.summary.scalar('loss_Gen_L1_GAN', loss_Gen)
tf.summary.scalar('loss_Dis_cross_entropy_completed', loss_Dis_cross_entropy_completed)
tf.summary.scalar('loss_Dis_cross_entropy_original', loss_Dis_cross_entropy_original)
tf.summary.scalar('loss_Dis', loss_Dis)


merged_scalar = tf.summary.merge_all()

merged_image = tf.summary.image('completed_image', completed_batch, max_outputs=batch_size)

# loss_dir_name = 'lr_' + str(learning_rate) + '_mom_' + str(momentum) + '_w_l1_' + str(weight_L1) + '_w_gan_' + str(weight_GAN)
loss_dir_name = 'main_session'

loss_validation_writer = tf.summary.FileWriter(project_dir + '/' + 'logs/loss/validation/' + loss_dir_name, sess.graph)
image_validation_writer = tf.summary.FileWriter(project_dir + '/' + 'logs/image/validation/' + loss_dir_name, sess.graph)

loss_train_writer = tf.summary.FileWriter(project_dir + '/' + 'logs/loss/train/' + loss_dir_name, sess.graph)
image_train_writer = tf.summary.FileWriter(project_dir + '/' + 'logs/image/train/' + loss_dir_name, sess.graph)


if __name__ == "__main__":

    # training image generator

    image_batch_train = image_gen.flow_from_directory(project_dir + '/data/my_places_train',  # # Restore to da
                                                      batch_size=batch_size,
                                                      target_size=(256, 256), shuffle=False)

    # validation image generator

    image_batch_validation = image_gen.flow_from_directory(project_dir + '/data/my_places_val',  # # Restore to da
                                                           batch_size=batch_size_validation,
                                                           target_size=(256, 256), shuffle=False)

    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    saver = tf.train.Saver(keep_checkpoint_every_n_hours=0.1)

    # Restoring the main Session if it exists

    if os.path.isfile(project_dir + '/' + 'saved_sessions_new/' + loss_dir_name + '/session.index'):
        saver.restore(sess, project_dir + '/' + 'saved_sessions_new/' + loss_dir_name + '/session')
        pickle_in = open(project_dir + '/' + 'saved_sessions_new/' + loss_dir_name + '/curr_iteration.pickle', 'rb')
        curr_iteration = pickle.load(pickle_in)
        curr_iteration = curr_iteration + 1

    else:
        curr_iteration = 0



    i = curr_iteration

    with sess.as_default():

        while i != total_no_of_iterations:

          #  i = curr_iteration

            # Training Generator Network using L1 loss

            if i < 100:
                print('Training only Generator with Euclidean Loss -', ' iter:', i, ' lr: ', learning_rate, ' wt_L1: ', weight_L1, ' wt_GAN: ', weight_GAN, ' batch_size: ', batch_size, ' momentum: ', momentum, sep='')
                h = np.random.randint(64, 128)
                w = np.random.randint(64, 128)
                mask = np.zeros(shape=(1, 256, 256, 3), dtype='float32')
                mask[:, h:h + 64, w:w + 64, 0:3] = 1.0
                mask_local_dis = np.zeros(shape=(batch_size, 256, 256, 3), dtype='float32')
                mask_local_dis[:, h-32:h+96, w-32:w+96, 0:3] = 1.0
                mask_local_dis = mask_local_dis.astype(dtype='bool')

                original_batch = image_batch_train.next()[0]
                patched_original_batch = apply_square_patch(np.copy(original_batch), h, w)
                validation_batch = image_batch_validation.next()[0]
                patched_validation_batch = apply_square_patch(np.copy(validation_batch), h, w)

                # The commented code below is for generating a test batch at the beginning of the training but it is
                # better to the load the batch manually and dump it using pickle as batch generated by default may not
                # be good enough for testing

                if i == 0:
                    test_mask = np.copy(mask)
                    test_mask_local_dis = np.copy(mask_local_dis)
                    test_batch_validation = np.copy(validation_batch)
                    test_batch_validation_patched = np.copy(patched_validation_batch)

                    test_batch_train = np.copy(original_batch)
                    test_batch_train_patched = apply_square_patch(np.copy(test_batch_train), h, w)

                if os.path.isdir(project_dir + '/' + 'results/validation/' + loss_dir_name) is False:
                    os.makedirs(project_dir + '/' + 'results/validation/' + loss_dir_name)

                if os.path.isdir(project_dir + '/' + 'results/train/' + loss_dir_name) is False:
                    os.makedirs(project_dir + '/' + 'results/train/' + loss_dir_name)

                    for j in range(batch_size_validation):
                        filename = project_dir + '/' + 'results/validation/' + loss_dir_name + '/image_' + str(j) + '_original_to_be_patched.jpg'
                        scipy.misc.imsave(filename, test_batch_validation_patched[j])

                    for j in range(batch_size_validation):
                        filename = project_dir + '/' + 'results/train/' + loss_dir_name + '/image_' + str(j) + '_original_to_be_patched.jpg'
                        scipy.misc.imsave(filename, test_batch_train_patched[j])

                    test_batch_validation_value = completed_batch.eval(feed_dict={patched_batch_placeholder: test_batch_validation_patched,
                                                                                  mask_placeholder: test_mask})

                    test_batch_train_value = completed_batch.eval(feed_dict={patched_batch_placeholder: test_batch_train_patched,
                                                                             mask_placeholder: test_mask})

                    for j in range(batch_size):
                        filename = project_dir + '/' + 'results/validation/' + loss_dir_name + '/image_' + str(j) + '_no_training.jpg'
                        scipy.misc.imsave(filename, test_batch_validation_value[j])

                    for j in range(batch_size):
                        filename = project_dir + '/' + 'results/train/' + loss_dir_name + '/image_' + str(j) + '_no_training.jpg'
                        scipy.misc.imsave(filename, test_batch_train_value[j])

                    # Saving the first test batch in pickle format so that it can always be loaded from the disk

                    pickle_dump(test_mask, project_dir + '/' + 'results/validation/' + loss_dir_name + '/test_mask')
                    pickle_dump(test_mask, project_dir + '/' + 'results/train/' + loss_dir_name + '/test_mask')

                    pickle_dump(test_mask_local_dis, project_dir + '/' + 'results/validation/' + loss_dir_name + '/test_mask_local_dis')
                    pickle_dump(test_mask_local_dis, project_dir + '/' + 'results/train/' + loss_dir_name + '/test_mask_local_dis')

                    pickle_dump(test_batch_validation, project_dir + '/' + 'results/validation/' + loss_dir_name + '/test_batch_validation')
                    pickle_dump(test_batch_train, project_dir + '/' + 'results/train/' + loss_dir_name + '/test_batch_train')

                    pickle_dump(test_batch_validation_patched, project_dir + '/' + 'results/validation/' + loss_dir_name + '/test_batch_validation_patched')
                    pickle_dump(test_batch_train_patched, project_dir + '/' + 'results/train/' + loss_dir_name + '/test_batch_train_patched')

                train_op_Gen_L1.run(feed_dict={original_batch_placeholder: original_batch,
                                               mask_placeholder: mask,
                                               patched_batch_placeholder: patched_original_batch})

                if i % checkpoints == 0:

                    test_mask = pickle_load(project_dir + '/' + 'results/validation/' + loss_dir_name + '/test_mask.pickle')

                    test_mask_local_dis = pickle_load(project_dir + '/' + 'results/validation/' + loss_dir_name + '/test_mask_local_dis.pickle')

                    test_batch_validation = pickle_load(project_dir + '/' + 'results/validation/' + loss_dir_name + '/test_batch_validation.pickle')

                    test_batch_validation_patched = pickle_load(project_dir + '/' + 'results/validation/' + loss_dir_name + '/test_batch_validation_patched.pickle')

                    test_batch_validation_value = completed_batch.eval(feed_dict={patched_batch_placeholder: test_batch_validation_patched,
                                                                                  mask_placeholder: test_mask})
                    for j in range(batch_size):
                        filename = project_dir + '/' + 'results/validation/' + loss_dir_name + '/image_' + str(j) + '_only_gen.jpg'
                        scipy.misc.imsave(filename, test_batch_validation_value[j])

                    test_mask = pickle_load(project_dir + '/' + 'results/train/' + loss_dir_name + '/test_mask.pickle')

                    test_mask_local_dis = pickle_load(project_dir + '/' + 'results/train/' + loss_dir_name + '/test_mask_local_dis.pickle')

                    test_batch_train = pickle_load(project_dir + '/' + 'results/train/' + loss_dir_name + '/test_batch_train.pickle')

                    test_batch_train_patched = pickle_load(project_dir + '/' + 'results/train/' + loss_dir_name + '/test_batch_train_patched.pickle')

                    test_batch_train_value = completed_batch.eval(feed_dict={patched_batch_placeholder: test_batch_train_patched,
                                                                             mask_placeholder: test_mask})
                    for j in range(batch_size):
                        filename = project_dir + '/' + 'results/train/' + loss_dir_name + '/image_' + str(j) + '_only_gen.jpg'
                        scipy.misc.imsave(filename, test_batch_train_value[j])

            # Training Discriminator Network using Cross Entropy loss

            elif 100 <= i < 140:

                print('Training only Discriminator with Cross Entropy Loss -', ' iter:', i, ' lr: ', learning_rate, ' wt_L1: ', weight_L1, ' wt_GAN: ', weight_GAN, ' batch_size: ', batch_size, ' momentum: ', momentum, sep='')
                h = np.random.randint(64, 128)
                w = np.random.randint(64, 128)
                mask = np.zeros(shape=(1, 256, 256, 3), dtype='float32')
                mask[:, h:h + 64, w:w + 64, 0:3] = 1.0
                mask_local_dis = np.zeros(shape=(batch_size, 256, 256, 3), dtype='float32')
                mask_local_dis[:, h - 32:h + 96, w - 32:w + 96, 0:3] = 1.0
                mask_local_dis = mask_local_dis.astype(dtype='bool')
                original_batch = image_batch_train.next()[0]
                patched_original_batch = apply_square_patch(np.copy(original_batch), h, w)

                train_op_Dis.run(feed_dict={patched_batch_placeholder: patched_original_batch,
                                            original_batch_placeholder: original_batch,
                                            mask_local_dis_placeholder: mask_local_dis,
                                            mask_placeholder: mask})

                if i % checkpoints == 0:

                    test_mask = pickle_load(project_dir + '/' + 'results/validation/' + loss_dir_name + '/test_mask.pickle')

                    test_mask_local_dis = pickle_load(project_dir + '/' + 'results/validation/' + loss_dir_name + '/test_mask_local_dis.pickle')

                    test_batch_validation = pickle_load(project_dir + '/' + 'results/validation/' + loss_dir_name + '/test_batch_validation.pickle')

                    test_batch_validation_patched = pickle_load(project_dir + '/' + 'results/validation/' + loss_dir_name + '/test_batch_validation_patched.pickle')

                    test_batch_validation_value = completed_batch.eval(feed_dict={patched_batch_placeholder: test_batch_validation_patched,
                                                                                  mask_placeholder: test_mask})

                    for j in range(batch_size):
                        filename = project_dir + '/' + 'results/validation/' + loss_dir_name + '/image_' + str(j) + '_only_dis.jpg'
                        scipy.misc.imsave(filename, test_batch_validation_value[j])

                    test_mask = pickle_load(project_dir + '/' + 'results/train/' + loss_dir_name + '/test_mask.pickle')

                    test_mask_local_dis = pickle_load(project_dir + '/' + 'results/train/' + loss_dir_name + '/test_mask_local_dis.pickle')

                    test_batch_train = pickle_load(project_dir + '/' + 'results/train/' + loss_dir_name + '/test_batch_train.pickle')

                    test_batch_train_patched = pickle_load(project_dir + '/' + 'results/train/' + loss_dir_name + '/test_batch_train_patched.pickle')

                    test_batch_train_value = completed_batch.eval(feed_dict={patched_batch_placeholder: test_batch_train_patched,
                                                                  mask_placeholder: test_mask})

                    for j in range(batch_size):
                        filename = project_dir + '/' + 'results/train/' + loss_dir_name + '/image_' + str(j) + '_only_dis.jpg'
                        scipy.misc.imsave(filename, test_batch_train_value[j])

            #  Training Generator Network using L1 and GAN loss and Discriminator Network using Cross Entropy loss

            elif 141 <= i < total_no_of_iterations:

                print('Training both Gen, Dis -', ' iter:', i, ' lr: ', learning_rate, ' wt_L1: ', weight_L1, ' wt_GAN: ', weight_GAN, ' batch_size: ', batch_size, ' momentum: ', momentum, sep='')
                h = np.random.randint(64, 128)
                w = np.random.randint(64, 128)
                mask = np.zeros(shape=(1, 256, 256, 3), dtype='float32')
                mask[:, h:h + 64, w:w + 64, 0:3] = 1.0
                mask_local_dis = np.zeros(shape=(batch_size, 256, 256, 3), dtype='float32')
                mask_local_dis[:, h - 32:h + 96, w - 32:w + 96, 0:3] = 1.0
                mask_local_dis = mask_local_dis.astype(dtype='bool')
                original_batch = image_batch_train.next()[0]
                patched_original_batch = apply_square_patch(np.copy(original_batch), h, w)

                train_op_Gen_L1_GAN.run(feed_dict={patched_batch_placeholder: patched_original_batch,
                                                   original_batch_placeholder: original_batch,
                                                   mask_local_dis_placeholder: mask_local_dis,
                                                   mask_placeholder: mask})

                train_op_Dis.run(feed_dict={patched_batch_placeholder: patched_original_batch,
                                            original_batch_placeholder: original_batch,
                                            mask_local_dis_placeholder: mask_local_dis,
                                            mask_placeholder: mask})

                if i % checkpoints == 0:

                    test_mask = pickle_load(project_dir + '/' + 'results/validation/' + loss_dir_name + '/test_mask.pickle')

                    test_mask_local_dis = pickle_load(project_dir + '/' + 'results/validation/' + loss_dir_name + '/test_mask_local_dis.pickle')

                    test_batch_validation = pickle_load(project_dir + '/' + 'results/validation/' + loss_dir_name + '/test_batch_validation.pickle')

                    test_batch_validation_patched = pickle_load(project_dir + '/' + 'results/validation/' + loss_dir_name + '/test_batch_validation_patched.pickle')

                    test_batch_validation_value = completed_batch.eval(feed_dict={patched_batch_placeholder: test_batch_validation_patched,
                                                                                  mask_placeholder: test_mask})

                    for j in range(batch_size):
                        filename = project_dir + '/' + 'results/validation/' + loss_dir_name + '/image_' + str(j) + '_both_gen_dis.jpg'
                        scipy.misc.imsave(filename, test_batch_validation_value[j])

                    test_mask = pickle_load(project_dir + '/' + 'results/train/' + loss_dir_name + '/test_mask.pickle')

                    test_mask_local_dis = pickle_load(project_dir + '/' + 'results/train/' + loss_dir_name + '/test_mask_local_dis.pickle')

                    test_batch_train = pickle_load(project_dir + '/' + 'results/train/' + loss_dir_name + '/test_batch_train.pickle')

                    test_batch_train_patched = pickle_load(project_dir + '/' + 'results/train/' + loss_dir_name + '/test_batch_train_patched.pickle')

                    test_batch_train_value = completed_batch.eval(feed_dict={patched_batch_placeholder: test_batch_train_patched,
                                                                             mask_placeholder: test_mask})

                    for j in range(batch_size):
                        filename = project_dir + '/' + 'results/train/' + loss_dir_name + '/image_' + str(j) + '_both_gen_dis.jpg'
                        scipy.misc.imsave(filename, test_batch_train_value[j])

            else:
                pass

            if i % checkpoints == 0:

                if os.path.isdir(project_dir + '/' + 'saved_sessions_new/' + loss_dir_name) is False:
                    os.makedirs(project_dir + '/' + 'saved_sessions_new/' + loss_dir_name)

                print('Saving checkpoints...')
                saver.save(sess, project_dir + '/' + 'saved_sessions_new/' + loss_dir_name + '/session')  # # Saving Session
                pickle_out = open(project_dir + '/' + 'saved_sessions_new/' + loss_dir_name + '/curr_iteration.pickle', 'wb')
                pickle.dump(curr_iteration, pickle_out)
                pickle_out.close()

                test_mask = pickle_load(project_dir + '/' + 'results/validation/' + loss_dir_name + '/test_mask.pickle')

                # Loading test batch for calculating loss for Tensorboard

                test_mask_local_dis = pickle_load(project_dir + '/' + 'results/validation/' + loss_dir_name + '/test_mask_local_dis.pickle')

                test_batch_validation = pickle_load(project_dir + '/' + 'results/validation/' + loss_dir_name + '/test_batch_validation.pickle')

                test_batch_validation_patched = pickle_load(project_dir + '/' + 'results/validation/' + loss_dir_name + '/test_batch_validation_patched.pickle')

                test_batch_train = pickle_load(project_dir + '/' + 'results/train/' + loss_dir_name + '/test_batch_train.pickle')

                test_batch_train_patched = pickle_load(project_dir + '/' + 'results/train/' + loss_dir_name + '/test_batch_train_patched.pickle')

                loss_validation_tensorboard = sess.run(merged_scalar, feed_dict={patched_batch_placeholder: np.copy(test_batch_validation_patched),
                                                                                 mask_placeholder: test_mask,
                                                                                 original_batch_placeholder: np.copy(test_batch_validation),
                                                                                 mask_local_dis_placeholder: test_mask_local_dis})

                loss_train_tensorboard = sess.run(merged_scalar, feed_dict={patched_batch_placeholder: np.copy(test_batch_train_patched),
                                                                            mask_placeholder: test_mask,
                                                                            original_batch_placeholder: np.copy(test_batch_train),
                                                                            mask_local_dis_placeholder: test_mask_local_dis})

                image_validation_tensorboard = sess.run(merged_image, feed_dict={patched_batch_placeholder: np.copy(test_batch_validation_patched),
                                                                                 mask_placeholder: test_mask,
                                                                                 original_batch_placeholder: np.copy(test_batch_validation),
                                                                                 mask_local_dis_placeholder: test_mask_local_dis})

                image_train_tensorboard = sess.run(merged_image, feed_dict={patched_batch_placeholder: np.copy(test_batch_train_patched),
                                                                            mask_placeholder: test_mask,
                                                                            original_batch_placeholder: np.copy(test_batch_train),
                                                                            mask_local_dis_placeholder: test_mask_local_dis})

                loss_validation_writer.add_summary(loss_validation_tensorboard, i)
                image_validation_writer.add_summary(image_validation_tensorboard, i)

                loss_train_writer.add_summary(loss_train_tensorboard, i)
                image_train_writer.add_summary(image_train_tensorboard, i)

            i = i + 1
            curr_iteration = i

    loss_validation_writer.close()
    image_validation_writer.close()
    loss_train_writer.close()
    image_train_writer.close()
