import copy
import numpy as np
from typing import List, Tuple, Optional


def get_independent_zeros(matrix_after_reduction: np.ndarray) -> Optional[List[Tuple[int, int]]]:
    """
    :param matrix_after_reduction: kwadratowa macierz liczb nieujemnych, zredukowana
    :return: lista indeksów zer niezależnych lub -None-, gdy brak rozwiązania
    """

    points: List[Tuple[int, int]] = []
    _m: np.ndarray = matrix_after_reduction.copy()

    for _ in range(_m.shape[0]):
        # zlicz zera w każdym wierszu
        zeros: np.ndarray = np.count_nonzero(_m == 0, axis=1)

        # test czy w iteracji znaleziono już zero niezależne
        zero_found: bool = False

        # jeśli w wierszu jest jedno zero, to jest niezależne
        temp: np.ndarray = np.nonzero(zeros == 1)[0]
        if temp.size > 0:
            # wyznaczenie współrzędnych
            row_idx: int = temp[0]
            col_idx: int = np.where(_m[row_idx, :] == 0)[0][0]
            points.append((row_idx, col_idx))

            # usunięcie wiersza i kolumny z macierzy
            _m[:, col_idx] = _m[:, col_idx] + np.inf
            _m[row_idx, :] = _m[row_idx, :] + np.inf

            zero_found = True

        # jeśli w wierszu jest więcej niż jedno zero, to wybierz pierwsze
        if not zero_found:
            temp: np.ndarray = np.nonzero(zeros > 1)[0]
            if temp.size > 0:
                # wyznaczenie współrzędnych
                row_idx: int = temp[0]
                col_idx: int = np.where(_m[row_idx, :] == 0)[0][0]
                points.append((row_idx, col_idx))

                # usunięcie wiersza i kolumny z macierzy
                _m[:, col_idx] = _m[:, col_idx] + np.inf
                _m[row_idx, :] = _m[row_idx, :] + np.inf

                zero_found = True

        # jeśli brak zer w macierzy, to zakończ
        if not zero_found:
            break

    return points if len(points) == _m.shape[0] else None


def min_lines(m, zeros_dependant):
    X, _ = m.shape
    zeros = m == 0
    zeros = zeros.astype(int)
    # indeksy oznaczonych kolumn/wierszy
    marked_rows = set()
    marked_cols = set()
    for i, j in zeros_dependant:
        zeros[i, j] = -1
    
    for idx, row in enumerate(zeros):
        for el in row:
            if el == -1: # zero niezależne
                break
        else:
            marked_rows.add(idx)

    while True:
        changes = False

        for row in range(X):
            for col in range(X):
                # zero zależne i oznaczony wiersz
                if zeros[row, col] == 1 and row in marked_rows and col not in marked_cols:
                    marked_cols.add(col)
                    changes = True

        for row in range(X):
            for col in range(X):
                # zero niezależne i oznaczona kolumna
                if zeros[row, col] == -1 and col in marked_cols and row not in marked_rows:
                    marked_rows.add(row)
                    changes = True
        
        if not changes:
            break
    
    all_rows = set([row_idx for row_idx in range(X)])

    return all_rows.difference(marked_rows), marked_cols


def matrix_reduction(original_matrix: np.ndarray) -> (np.ndarray, np.ndarray, int):
    cost = 0
    buffor_matrix = copy.deepcopy(original_matrix)

    
    for row in buff_matrix:
        if not np.any(row== 0):
            minimum = np.min(row)
            cost += minimum
            row = row - minimum
    
    buffor_matrix = buffor_matrix.T
    
    for row in buff_matrix:
        if not np.any(row== 0):
            minimum = np.min(row)
            cost += minimum
            row = row - minimum
            
    buffor_matrix = buffor_matrix.T
    
    return original_matrix, buffor_matrix, cost


def main():
    def TEST_get_independent_zeros():
        m1 = np.array([[15, 15, 0],
                       [0, 0, 10],
                       [5, 5, 0]])
        m2 = np.array([[10, 10, 0],
                       [0, 0, 15],
                       [0, 0, 0]])
        m3 = np.array([[25, 40, 0],
                       [0, 0, 20],
                       [30, 0, 0]])
        m4 = np.array([[0, 0, 0, 0, 0],
                       [1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1]])
        m5 = np.zeros((10, 10))
        m6 = np.array([[1, 0],
                       [0, 0]])
        m7 = np.array([[0, 1],
                       [0, 0]])
        m8 = np.array([[0, 0],
                       [1, 0]])
        m9 = np.array([[0, 0],
                       [0, 1]])
        m10 = np.array([[1, 1],
                        [0, 0]])
        m11 = np.array([[1, 0],
                        [1, 0]])
        m12 = np.array([[0, 1],
                        [0, 1]])
        m13 = np.array([[0, 0],
                        [1, 1]])
        print(get_independent_zeros(m1))
        print(get_independent_zeros(m2))
        print(get_independent_zeros(m3))
        print(get_independent_zeros(m4))
        print(get_independent_zeros(m5))
        print(get_independent_zeros(m6))
        print(get_independent_zeros(m7))
        print(get_independent_zeros(m8))
        print(get_independent_zeros(m9))
        print(get_independent_zeros(m10))
        print(get_independent_zeros(m11))
        print(get_independent_zeros(m12))
        print(get_independent_zeros(m13))

    matrix = np.random.randint(10, size=(6, 6))

    TEST_get_independent_zeros()


if __name__ == '__main__':
    main()
