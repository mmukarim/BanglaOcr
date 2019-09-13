from keras import backend as K
from keras.optimizers import Adadelta
from keras.callbacks import EarlyStopping, ModelCheckpoint
from models.crnn import get_model
from data_preperation.data_generator_crnn import DataGenerator

K.set_learning_phase(0)

# # Model description and training

model = get_model(training=True)

train_root_directory = '/home/bjit-541/practiceProjects/Bangla-OCR/data/train/'
train_csv_path = '/home/bjit-541/practiceProjects/Bangla-OCR/data/train/train_labels.csv'

test_root_directory = '/home/bjit-541/practiceProjects/Bangla-OCR/data/test/'
test_csv_path = '/home/bjit-541/practiceProjects/Bangla-OCR/data/test/test_labels.csv'

text_file_path = '/home/bjit-541/practiceProjects/Bangla-OCR/data/digit_label.txt'

train_data = DataGenerator(10, train_root_directory, train_csv_path, text_file_path, 2)
train_data.build_data()

test_data = DataGenerator(10, test_root_directory, test_csv_path, text_file_path, 2)
test_data.build_data()

early_stop = EarlyStopping(monitor='loss', min_delta=0.001, patience=4, mode='min', verbose=1)
checkpoint = ModelCheckpoint(filepath='LSTM+BN5--{epoch:02d}--{val_loss:.3f}.hdf5', monitor='loss', verbose=1, mode='min', period=1)

model.summary()

# the loss calc occurs elsewhere, so use a dummy lambda func for the loss
model.compile(loss={'ctc': lambda y_true, y_pred: y_pred - y_true}, optimizer='adam')

# captures output of softmax so we can decode the output during visualization
model.fit_generator(train_data.next_batch(),
                    steps_per_epoch=int(train_data.n/10),
                    epochs=2,
                    # callbacks=[checkpoint],
                    validation_data=test_data.next_batch(),
                    validation_steps=int(test_data.n/10),
                    verbose=1)
