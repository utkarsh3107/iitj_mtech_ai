#Function accepts Array and k value as input
def problem1(array, k):
    dictionary = {}
    result = 0
    # Initialize a dictionary where key represents the number and count represents the number of elements
    # which are received entered
    for each in range(0, len(array)):
        value = array[each]
        if value in dictionary:
            count = dictionary.get(value)
            count = count + 1
            dictionary[value] = count
        else:
            dictionary[value] = 1

    # We know if
    # a - b = k, then a = k + b
    # The same idea we can use where we for every 'a' we try to find 'k - a' which is actually 'b'.
    # There can be multiple numbers present in each key. To compute for the total number of possible counts
    # we will multiply the count with count - 1
    for key, count in dictionary.items():
        element = key * 2 - k
        if element == key:
            if count > 1:
                result = result + (count * (count - 1))
        else:
            if element in dictionary:
                result = result + count

    return result



array = [2, 6, 4]
k = 2
print(problem1(array, k))