# Insert the element to the matrix in format [row, column, element]
def insert_element(matrix, row, column):
    element = int(input("Enter element"))
    matrix.append([row, column, element])

# Delete the element for the 'row' and 'column' provided by the user
def delete_element(matrix, row, column):
    pos = 0
    for each in matrix:
        if row == each[0] and column == each[1]:
            matrix.pop(pos)
            break
        pos = pos + 1

# Sorts the sparse matrix with below Algo
# If <Row, 0> value is greater than <Row, 1>
#    swap <Row,0> with <Row, 1>
# If <Row, 0> is same as <Row, 1>
#   Check <Row,1> with <Row, 1>
#   If <Row, 1> value is greater than <Row, 1>
#       swap <Row,0> with <Row, 1>
def sort(matrix):
    max_row = -1
    max_col = -1
    for i in range(0, len(matrix)):
        first = matrix[i]
        for j in range(i + 1, len(matrix)):
            second = matrix[j]

            if max_row < second[0]:
                max_row = second[0]
            if max_col < second[1]:
                max_col = second[1]
            if first[0] > second[0]:
                matrix[i] = second
                matrix[j] = first
                first = matrix[i]

            if first[0] == second[0]:
                if first[1] > second[1]:
                    matrix[i] = second
                    matrix[j] = first
                    first = matrix[i]
            #print(' i = ', i , '; j = ', j, 'matrix = ', matrix)
    return max_row, max_col

#Initialize the matrix
def init(matrix):
    row = int(input("Enter row count"))
    column = int(input("Enter column count"))
    insert_element(matrix, row, column)

#Print matrix
def print_matrix(matrix):
    print('============== Sparse Matrix ==================\n')
    max_row, max_col = sort(matrix)
    pos = 0
    for i in range(0, max_row + 1):
        for j in range(0, max_col + 1):
            element = matrix[pos]
            if element[0] == i and element[1] == j:
                print(element[2], end='\t')
                if pos < len(matrix) - 1:
                    pos = pos + 1
            else:
                print(0, end='\t')
        print('\n')



matrix = []
init(matrix)
init(matrix)
init(matrix)
init(matrix)
print_matrix(matrix)

