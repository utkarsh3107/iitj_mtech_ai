"""

@author Utkarsh Thusoo
"""

elements = [10, 34, 48, 59, 63, 74, 85, 120, 140]
reverse_elements = [140, 120, 85, 74, 63, 59, 48, 34, 10]
repeated1 = [1, 3, 3, 4, 4, 5, 6, 7]


def basic_sort(array):
    for x in range(0, len(array)):
        for y in range(x + 1, len(array)):
            if array[x] < array[y]:
                temp = array[x]
                array[x] = array[y]
                array[y] = temp


def binary_search(element, left, right):
    if left <= right:
        mid = int((left + right) / 2)

        if element == elements[mid]:
            return mid
        elif element > elements[mid]:
            return binary_search(element, mid + 1, right)
        else:
            return binary_search(element, left, mid - 1)

    return -1


def reverse_binary_search(element, left, right):
    if left <= right:
        mid = int((left + right) / 2)

        if element == reverse_elements[mid]:
            return mid
        elif element < reverse_elements[mid]:
            return reverse_binary_search(element, mid + 1, right)
        else:
            return reverse_binary_search(element, left, mid - 1)

    return -1


def find_first(array, element, left, right):

    if right - left <= 1:
        if array[left] == element:
            return left
        elif array[right] == element:
            return right
        else:
            return -1

    mid = int((left + right) / 2)

    if array[mid] == element:
        return find_first(array, element, left, mid)
    elif array[mid] > element:
        return find_first(array, element, left, mid - 1)
    else:
        return find_first(array, element, mid + 1, right)


def find_last(array, element, left, right):

    if right - left <= 1:
        if array[right] == element:
            return right
        elif array[left] == element:
            return left
        else:
            return -1

    mid = int((left + right) / 2)

    if array[mid] == element:
        return find_last(array, element, mid, right)
    elif array[mid] > element:
        return find_last(array, element, left, mid - 1)
    else:
        return find_last(array, element, mid + 1, right)


print("Ascending Order: element %d found at %d" % (120, binary_search(120, 0, len(elements) - 1)))
print("Descending Order: element %d found at %d" % (120, reverse_binary_search(120, 0, len(reverse_elements) - 1)))
print("First instance of element %d found at %d" % (1, find_first(repeated1, 1, 0, len(repeated1) - 1)))
print("Last instance of element %d found at %d" % (1, find_last(repeated1, 3, 0, len(repeated1) - 1)))

for each in range(0, 6):
    print("Count of %d in list is %d" % (each,
                                         (find_last(repeated1, each, 0, len(repeated1) - 1) -
                                          find_first(repeated1, each, 0, len(repeated1) - 1)
                                          ) + 1))
