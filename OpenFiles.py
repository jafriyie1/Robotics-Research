import os
from PIL import Image
import numpy as np
import skimage.io as io
from pathlib import Path

class OpenFiles():

    def __init__(self, path):
        self.__path = path

    def get_images(self):
        file_list = []
        np_images = []
        for files in os.listdir(self.__path):
            if files.endswith(".jpg"):
                file_list.append(files)

        for x in file_list:
            tmp_file = self.__path+"/"+x
            img = io.imread(tmp_file)
            np_images.append(img)
        return np_images


    def transform_images(self):
        temp_img = io.imread(self.__path)
        #print(temp_img.shape)
        temp_img = temp_img.reshape(1, 256,256,3)
        return temp_img
