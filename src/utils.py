import os

import numpy as np
import matplotlib.pyplot as plt
import cv2


project_dir = os.getcwd()


class Mask(object):
    def __init__(self, h, w, x_start, y_start, mask_height, mask_width, obj_name):
        self.mask = self.create_mask(h, w, x_start, y_start, mask_height, mask_width)
        self.obj_name = obj_name

    @classmethod
    def create_mask(cls, h, w, x_start, y_start, mask_height, mask_width):
        mask = np.zeros((h, w, 3))

        x_end = x_start + mask_width
        y_end = y_start + mask_height

        mask[x_start:x_end, y_start:y_end, 0] = 1
        mask[x_start:x_end, y_start:y_end, 1] = 1
        mask[x_start:x_end, y_start:y_end, 2] = 1

        return mask

    def show_mask(self):
        if self.mask is None:
            return 'Please create mask first using create_mask method'
        plt.imshow(self.mask)
        plt.show()

    def save_mask(self, path):
        plt.imsave(path, self.mask)

    def create_and_save_patched_image_and_mask(self, image_path, save_path):
        image_name = image_path.rsplit('/', 1)[1]
        image_name = image_name.rsplit('.', 1)[0]

        image = plt.imread(image_path)
        image = cv2.resize(image, (256, 256))

        patched_image = np.copy(image)

        patched_image[:,:,0] = image[:,:,0] * (1 - self.mask[:,:,0])
        patched_image[:,:,1] = image[:,:,1] * (1 - self.mask[:,:,1])
        patched_image[:,:,2] = image[:,:,2] * (1 - self.mask[:,:,2])

        plt.imshow(patched_image)
        plt.show()

        plt.imsave(save_path + '/' + image_name + '_' + self.obj_name + '.png', patched_image)
        plt.imsave(save_path + '/' + image_name + '_' + self.obj_name + '_mask.png', self.mask)


image_path = '/home/karan/PycharmProjects/Image_Completion_Experiment/data/custom_data/test_images/interior.jpg'
destination_dir = '/home/karan/PycharmProjects/Image_Completion_Experiment/data/test_images'


mask1 = Mask(256, 256, 40, 40, 50, 50, '1')
mask1.create_and_save_patched_image_and_mask(image_path, destination_dir)

mask2 = Mask(256, 256, 160, 170, 40, 60, '2')
mask2.create_and_save_patched_image_and_mask(image_path, destination_dir)

mask3 = Mask(256, 256, 60, 90, 60, 30, '3')
mask3.create_and_save_patched_image_and_mask(image_path, destination_dir)

mask4 = Mask(256, 256, 96, 96, 64, 64, '4')
mask4.create_and_save_patched_image_and_mask(image_path, destination_dir)

# mask2 = Mask(256, 256, 160, 170, 40, 60)
# mask3 = Mask(256, 256, 60, 90, 60, 30)
# mask4 = Mask(256, 256, 96, 96, 64, 64)
#
# mask1.create_and_save_patched_image_and_mask('/home/karan/PycharmProjects/Image_Completion_Experiment/data/my_places_val/original/Places365_test_00000126.jpg')

# mask1.show_mask()
# mask2.show_mask()
# mask3.show_mask()
# mask4.show_mask()

# mask1.save_mask(project_dir + '/data/test_images/mask1.png')
# mask2.save_mask(project_dir + '/data/test_images/mask2.png')
# mask3.save_mask(project_dir + '/data/test_images/mask3.png')
# mask4.save_mask(project_dir + '/data/test_images/mask4.png')
