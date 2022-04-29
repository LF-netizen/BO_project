import copy

import numpy as np

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
    matrix = np.random.randint(10, size=(6, 6))



if __name__ == '__main__':
    main()
