import numpy as np


def flood_fill(row, column):
    global array_for_flood_fill
    if row < 0 or column < 0 or row >= rows or column >= cols:
        return
    elif array_for_flood_fill[row][column] == 1 and flag[row][column] == 0:
        global global_count
        global_count += 1
        flag[row][column] = 1
        flood_fill(row+1, column)
        flood_fill(row-1, column)
        flood_fill(row, column+1)
        flood_fill(row, column-1)


def get_largest_object(input_array):
    global flag, rows, cols, global_count, array_for_flood_fill
    global_count = 0
    max_count = 0
    rows = len(input_array)
    cols = len(list(zip(*input_array)))
    flag = np.zeros((rows, cols))
    array_for_flood_fill = input_array

    for i in range(rows):
        for j in range(cols):
            if input_array[i][j] == 1 and flag[i][j] == 0:
                flood_fill(i, j)
            elif global_count:
                if global_count >= max_count:
                    max_count = global_count
                global_count = 0
    return max_count


def create_test_array(size):
    return np.random.random_integers(0, 1, array_size)


# input_array = ([[0, 0, 0, 0, 0, 0, 0],
#                 [1, 1, 0, 0, 0, 0, 0],
#                 [1, 1, 1, 0, 0, 0, 0],
#                 [0, 0, 0, 1, 1, 1, 0],
#                 [1, 1, 0, 0, 0, 0, 0],
#                 [1, 1, 1, 1, 0, 0, 0]])

array_size = (6, 7)
input_array = create_test_array(array_size)
print(input_array)
result = get_largest_object(input_array)
print(result)





