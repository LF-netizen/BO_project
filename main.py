import copy

import numpy as np


def matrix_reduction(original_matrix: np.ndarray) -> (np.ndarray, np.ndarray, int):
    cost = 0
    print(original_matrix)
    buffor_matrix = copy.deepcopy(original_matrix)
    for i in range(len(buffor_matrix)):
        if not np.any(buffor_matrix[i] == 0):
            minimum = np.min(buffor_matrix[i])
            cost += minimum
            buffor_matrix[i] = buffor_matrix[i] - minimum
    print(buffor_matrix, cost)
    buffor_matrix = buffor_matrix.T
    for j in range(len(buffor_matrix)):
        if not np.any(buffor_matrix[j] == 0):
            minimum = np.min(buffor_matrix[j])
            cost += minimum
            buffor_matrix[j] = buffor_matrix[j] - minimum
    buffor_matrix = buffor_matrix.T
    print(buffor_matrix, cost)
    return original_matrix, buffor_matrix, cost


def main():
    matrix = np.random.randint(10, size=(6, 6))

    print(matrix_reduction(matrix))


if __name__ == '__main__':
    main()
