import numpy as np


# to generate test case
def create_main_list():
    audio_list = []
    for i in range(101):
        audio_array = np.random.randint(10, size=np.random.choice(100000))
        audio_list.append(audio_array)
    return audio_list


# get new sliced list
def create_sliced_list(audio_list, items, sliced_length):
    sliced_list = []
    for i in range(items):
        temp = np.zeros(1000)
        minimum_length = sliced_length
        audio_number = np.random.choice(len(audio_list))
        audio = audio_list[audio_number]
        audio_length = len(audio)
        if audio_length > minimum_length:
            audio_length = minimum_length
        for j in range(audio_length):
            temp[j] = audio[j]
        sliced_list.append(temp)
    return sliced_list


audio_list = create_main_list()
items = 32
sliced_length = 1000
sliced_list = create_sliced_list(audio_list, items, sliced_length)

for i in range(32):
    print(sliced_list[i])
