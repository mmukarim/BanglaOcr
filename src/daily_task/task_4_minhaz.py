image = [[1, 2, 4, 2, 3],
         [4, 3, 0, 1, 5],
         [5, 0, 1, 4, 3]]

image_height = len(image)
image_width = len(list(zip(*image)))
m = 2
n = 2


def get_window_size_and_initial_vector_for_subarray(m, n, image_width, image_height):
    window_size_x = int(image_width // m)
    window_size_y = int(image_height // n)
    m_n_list = []

    for i in range(m):
        temp = []
        for j in range(n):
            temp.append([window_size_y, window_size_x])
        m_n_list.append(temp)
    print(m_n_list)
    # for i in range(image_width % m):
    #     for j in range(n):
    #         m_n_list[i][j][0] = m_n_list[i][j][0] + 1
    #
    # for i in range(m):
    #     for j in range(image_height % n):
    #         m_n_list[i][j][1] = m_n_list[i][j][1] + 1

    x_y_list = []
    for i in range(m):
        temp_list = []
        for j in range(n):
            x = 0
            y = 0
            if i > 0:
                for k in range(i):
                    x += m_n_list[k][j][1]
            if j > 0:
                for l in range(j):
                    y += m_n_list[i][l][0]
            temp = [x, y]
            temp_list.append(temp)
        x_y_list.append(temp_list)
    return m_n_list, x_y_list


def get_subarray(window_size, subarray_vector):
    subarray = []
    for i in range(m):
        temp_array = []
        for j in range(n):
            temp_list = []
            for x_m in range(subarray_vector[i][j][0], (subarray_vector[i][j][0] + window_size[i][j][1])):
                temp = []
                for y_n in range(subarray_vector[i][j][1], (subarray_vector[i][j][1] + window_size[i][j][0])):
                    temp.append(image[x_m][y_n])
                temp_list += temp
            temp_array.append(temp_list)
        subarray.append(temp_array)
    return subarray


def get_pooled_list(subarray, m, n, pool_type='max'):
    final_list = []
    for i in range(m):
        temp = []
        for j in range(n):
            if pool_type == 'max':
                temp.append(max(subarray[i][j]))
            elif pool_type == 'avg':
                temp.append(sum(subarray[i][j])/float(len(subarray[i][j])))
        final_list.append(temp)
    return final_list


window_size, initial_vector_for_subarray = get_window_size_and_initial_vector_for_subarray(m, n, image_width, image_height)
subarray = get_subarray(window_size, initial_vector_for_subarray)

pooled_list = get_pooled_list(subarray, m, n, 'max')
print(pooled_list)
