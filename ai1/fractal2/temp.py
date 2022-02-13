
def sortedSquaredArray(array):
    result = []
    for each in array:
        result.append(each ** 2)
        
    result.sort()
    return result

print(sortedSquaredArray([1, 4, 9, 25, 36, 64, 81]))    