import random
import string


def create_random_string():
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(10))


def create_test_data_set(number_of_samples, number_of_classes):
    random_list: str > int = {}
    for i in range(number_of_samples):
        random_string = create_random_string()
        random_list[random_string] = random.randrange(number_of_classes)
    return random_list


def get_keys_by_value(dict_of_elements, value_to_find):
    list_of_keys = list()
    list_of_items = dict_of_elements.items()
    for item in list_of_items:
        if item[1] == value_to_find:
            list_of_keys.append(item[0])
    return list_of_keys


def solve(samples, n):
    list_of_classes = list(set(samples.values()))
    list_of_new_samples = {}
    if len(list_of_classes) > n:
        new_list_of_classes = set()
        while len(new_list_of_classes) < n:
            new_list_of_classes.add(random.choice(list_of_classes))
        for int_class in new_list_of_classes:
            list_of_keys = []
            list_of_files = get_keys_by_value(samples, int_class)
            list_of_keys.append(random.choice(list_of_files))
            list_of_new_samples[int_class] = list_of_keys
    else:
        no_of_data_per_class = n // len(list_of_classes)
        for int_class in list_of_classes:
            list_of_keys = []
            list_of_files = get_keys_by_value(samples, int_class)
            for i in range(no_of_data_per_class):
                list_of_keys.append(random.choice(list_of_files))
            list_of_new_samples[int_class] = list_of_keys
        new_list_of_classes = set()
        while len(new_list_of_classes) < n % len(list_of_classes):
            new_list_of_classes.add(random.choice(list_of_classes))
        for i in new_list_of_classes:
            list_of_files = get_keys_by_value(samples, i)
            list_of_keys = list_of_new_samples[i]
            list_of_keys.append(random.choice(list_of_files))
            list_of_new_samples[i] = list_of_keys
    return list_of_new_samples


no_of_classes = 3
no_of_test_samples = 150
no_of_target_samples = 300

test_list = create_test_data_set(no_of_test_samples, no_of_classes)
list_of_samples = solve(test_list, no_of_target_samples)

if no_of_classes > no_of_target_samples:
    for item in list_of_samples.values():
        assert (len(item) == 1)
else:
    for item in list_of_samples.values():
        assert (
                len(item) == no_of_target_samples // no_of_classes
                or
                len(item) == (no_of_target_samples // no_of_classes) + 1
        )
