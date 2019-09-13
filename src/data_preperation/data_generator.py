import os

import pandas as pd
import keras
import cv2
import numpy as np
from utils.string_label_converter import StrLabelConverter


class DataGenerator(keras.utils.Sequence):
    def __init__(self, batch_size, root_directory, csv_path, text_file_path, shuffle=True):
        self.batch_size = batch_size
        self.root_directory = root_directory
        self.data = pd.read_csv(csv_path, header=None)
        self.text_file_path = text_file_path
        self.max_width = 365
        self.max_height = 62
        self.shuffle = shuffle
        self.img_list = os.listdir(self.root_directory)
        self.n = len(self.img_list)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Get image data
        image_data = cv2.imread(str(self.root_directory) + str(self.data.iloc[index, 0]))
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

        # Get integer sequence  from text label
        label_converter = StrLabelConverter()
        integer_sequence_label = label_converter.decode_data(text_label)
        np_sequence_label = np.zeros(10)
        for i in range(len(integer_sequence_label)):
            np_sequence_label[i] = integer_sequence_label[i]
        print(np_sequence_label.shape)
        integer_sequence_label_np = np.array(integer_sequence_label, int)
        normalized_image = np.reshape(normalized_image, (-1, 62, 365, 3))
        # print(text_label, integer_sequence_label_np)
        # return normalized_image, integer_sequence_label
        input_length = np.ones(normalized_image.shape[2])
        label_length = np.ones(len(integer_sequence_label))
        inputs = {
            'the_input': normalized_image,  # (bs, 128, 64, 1)
            'the_labels': np_sequence_label,  # (bs, 8)
            'input_length': input_length,  # (bs, 1) -> 모든 원소 value = 30
            'label_length': label_length  # (bs, 1) -> 모든 원소 value = 8
        }
        outputs = {'ctc': np.zeros([1])}
        return inputs, outputs

