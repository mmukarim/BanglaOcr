import os
import random
from abc import ABC

import pandas as pd
import keras
import cv2
import numpy as np
from utils.string_label_converter import StrLabelConverter

CHAR_VECTOR = "০১২৩৪৫৬৭৮৯"

letters = list(CHAR_VECTOR)


def text_to_labels(text):
    return list(map(lambda x: letters.index(x), text))

class DataGenerator:
    def __init__(self, batch_size, root_directory, csv_path, text_file_path, downsample_factor, max_text_len=11):
        self.batch_size = batch_size
        self.root_directory = root_directory
        self.data = pd.read_csv(csv_path, header=None)
        self.text_file_path = text_file_path
        self.max_width = 365
        self.max_height = 62
        self.downsample_factor = downsample_factor
        self.img_list = os.listdir(self.root_directory)
        self.n = len(self.img_list)-1
        self.indexes = list(range(self.n))
        self.images = np.zeros((self.n, self.max_height, self.max_width, 3))
        self.texts = []
        self.cur_index = 0
        self.max__text_length = max_text_len

    def build_data(self):
        print(self.n, "Image loading started")
        for index in range(self.n):
            image_data = cv2.imread(str(self.root_directory) + str(self.data.iloc[index, 0]))
            print(image_data.shape)
            image_height, image_width, image_channels = image_data.shape
            delta_width = self.max_width - image_width
            delta_height = self.max_height - image_height

            # Calculate padding
            top = bottom = delta_height // 2
            if delta_height % 2 != 0:
                bottom = top + 1
            left = right = delta_width // 2
            if delta_width % 2 != 0:
                right = left + 1

            # Add padding to image data
            image_data = cv2.copyMakeBorder(image_data, top, bottom, left, right, cv2.BORDER_CONSTANT)

            # Normalize image data
            normalized_image = cv2.normalize(image_data, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)

            # Get label from csv
            label_from_csv = self.data.iloc[index, 1]

            # Get text label from text file using label from csv
            fp = open(self.text_file_path, encoding='utf8')
            lines = fp.readlines()
            text_label = lines[int(label_from_csv)]
            fp.close()
            self.images[index, :, :, :] = normalized_image
            self.texts.append(text_label)
        print(self.n, "Image loading ended")

    def next_sample(self):
        self.cur_index += 1
        if self.cur_index >= self.n:
            self.cur_index = 0
            random.shuffle(self.indexes)
        return self.images[self.indexes[self.cur_index]], self.texts[self.indexes[self.cur_index]]

    def next_batch(self):       ## batch size만큼 가져오기
        label_converter = StrLabelConverter()
        while True:
            X_data = np.ones([self.batch_size, self.max_height, self.max_width, 3])     # (bs, 128, 64, 1)
            Y_data = np.ones([self.batch_size, self.max__text_length])             # (bs, 9)
            input_length = np.ones((self.batch_size, 1)) * (self.max_width // self.downsample_factor - 2) # (bs, 1)
            label_length = np.zeros((self.batch_size, 1))           # (bs, 1)

            for i in range(self.batch_size):
                image, text = self.next_sample()
                # image = image.T
                # print(image.shape)
                # # image = np.expand_dims(image, -1)
                text = text.replace('\n', '')
                X_data[i] = image
                sequence_label = np.zeros(len(Y_data[i]))
                labeled_text = text_to_labels(text)
                for j in range(len(labeled_text)):
                    sequence_label[j] = labeled_text[j]
                Y_data[i] = sequence_label
                label_length[i] = len(text)

            # dict 형태로 복사
            inputs = {
                'the_input': X_data,  # (bs, 128, 64, 1)
                'the_labels': Y_data,  # (bs, 8)
                'input_length': input_length,  # (bs, 1) -> 모든 원소 value = 30
                'label_length': label_length  # (bs, 1) -> 모든 원소 value = 8
            }
            outputs = {'ctc': np.zeros([self.batch_size])}   # (bs, 1) -> 모든 원소 0
            yield (inputs, outputs)
