resolution = 128

# for i in range(resolution):
#     for j in range(resolution):
#         row = [0 for i in range(resolution * resolution)]
#         cell_count = j + i * resolution
#         # print(cell_count)
#         # print(cell_count % resolution)
#
#         if (i == 0 or i == resolution - 1) and (j == 0 or j == resolution-1):
#             row[cell_count] = 2
#         elif i == 0 or i == resolution - 1 or j == 0 or j == resolution - 1:
#             row[cell_count] = 3
#         else:
#             row[cell_count] = 4
#
#         row[cell_count + 1] = 1
#
#         print(row)

def normal_matrix():
    values = []
    col_indices = []
    row_indices = [0]

    for i in range(resolution):
        for j in range(resolution):
            cell_count = j + i * resolution
            # row = [0 for i in range(resolution * resolution)]

            neighbors = []

            if i - 1 >= 0:
                neighbors.append((i - 1) * resolution + j)
            if i + 1 < resolution:
                neighbors.append((i + 1) * resolution + j)
            if j - 1 >= 0:
                neighbors.append(i * resolution + (j - 1))
            if j + 1 < resolution:
                neighbors.append(i * resolution + (j + 1))

            # row[cell_count] = len(neighbors)

            for neighbor in neighbors:
                # row[neighbor] = 1
                values.append(1)
                col_indices.append(neighbor)

            values.append(len(neighbors))
            col_indices.append(cell_count)
            row_indices.append(len(values))

            # print(row)
    return (values, col_indices, row_indices)

def csr_matrix(values, col_indices, row_indices):
    mat = [[0 for i in range(resolution * resolution)] for i in range(resolution * resolution)]

    for i in range(len(values)):
        val = values[i]
        mat[calc_row_index(i, row_indices)][col_indices[i]] = val

    for row in mat:
        print(row)

def calc_row_index(val_index, row_indices):
    for i in range(len(row_indices) - 1):
        if val_index >= row_indices[i] and val_index < row_indices[i+1]:
            return i

if __name__ == "__main__":
    (values, col_indices, row_indices) = normal_matrix()
    print("values:    " + str(len(values)) + " col_indices: " + str(len(col_indices)) + " row_indices: " + str(len(row_indices)))
    # print("normal: " + str(resolution * resolution * resolution * resolution))
    # values = [3, 4, 5, 7, 2, 6]
    # col_indices = [2, 4, 2, 3, 1, 2]
    # row_indices = [0, 2, 4, 4, 6]
    csr_matrix(values, col_indices, row_indices)
