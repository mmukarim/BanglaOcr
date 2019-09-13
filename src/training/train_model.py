from data_preperation import data_generator
from models import model_from_pretrained_resnet50


train_root_directory = '/home/bjit-541/practiceProjects/Bangla-OCR/data/train/'
train_csv_path = '/home/bjit-541/practiceProjects/Bangla-OCR/data/train/train_labels.csv'

test_root_directory = '/home/bjit-541/practiceProjects/Bangla-OCR/data/test/'
test_csv_path = '/home/bjit-541/practiceProjects/Bangla-OCR/data/test/test_labels.csv'

text_file_path = '/home/bjit-541/practiceProjects/Bangla-OCR/data/digit_label.txt'

train_data = data_generator.DataGenerator(1, train_root_directory, train_csv_path, text_file_path)
test_data = data_generator.DataGenerator(1, test_root_directory, test_csv_path, text_file_path)


height = 62
width = 365
shape = 3
model = model_from_pretrained_resnet50.PreTrainedResnet50()
model = model.setup()
model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit_generator(train_data,
                    steps_per_epoch=10,
                    epochs=20,
                    validation_data=test_data,
                    validation_steps=10,
                    verbose=1
                    )
