# from keras.layers import Conv2D, LSTM
# from keras.models import Model, Sequential
# from data_preperation import data_generator
#
# train_root_directory = '/home/bjit-541/practiceProjects/Bangla-OCR/data/train/'
# train_csv_path = '/home/bjit-541/practiceProjects/Bangla-OCR/data/train/train_labels.csv'
#
# test_root_directory = '/home/bjit-541/practiceProjects/Bangla-OCR/data/test/'
# test_csv_path = '/home/bjit-541/practiceProjects/Bangla-OCR/data/test/test_labels.csv'
#
# text_file_path = '/home/bjit-541/practiceProjects/Bangla-OCR/data/digit_label.txt'
#
# train_data = data_generator.DataGenerator(2, train_root_directory, train_csv_path, text_file_path)
# test_data = data_generator.DataGenerator(2, test_root_directory, test_csv_path, text_file_path)
#
# height = 62
# width = 365
# shape = 3
#
# model = Sequential()
# input_shape = (height, width, shape)
#
# model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape, activation='relu'))
# model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape, activation='relu'))
#
# model.compile(loss='categorical_crossentropy', optimizer='Adam')
#
# model.fit_generator(train_data, epochs=1)