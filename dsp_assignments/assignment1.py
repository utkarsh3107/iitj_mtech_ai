"""
Below file would contain the python code for Assignment 1.

@author Utkarsh Thusoo
"""


"""
Q1. Alice has some cards with numbers written on them. She arranges the cards
in decreasing order. She challenges Bob to pick out the card containing number
120 by turning over as few cards as possible. Write a recursive function to help
Bob locate the card containing number 120. Show assumptions and actions step
by step. The numbers on card are as below.

10 34 48 59 63 74 85 120 140

Write an algorithm or programming code in any preferable language (using
C/C++/Java/Python) and also give dry run for the same (Take any example
of your choice for dry run).

Sol: The given question expects us to take a list as input and convert it in descending order and then perform binary
sort on the given list
"""


def descending_sort(array):
    """
    Function accepts an list as input and sorts the list in descending order.
    :param array: list of elements received
    :return: void
    """
    for x in range(0, len(array)):
        for y in range(x + 1, len(array)):
            if array[x] < array[y]:
                temp = array[x]
                array[x] = array[y]
                array[y] = temp


def reverse_binary_search(array, element, left, right):
    """
    This function expects an input list which should be in descending order and would return index if the element
    is found. -1 if the element is not found

    :param array: The input list which is expected in descending order
    :param element: The element we are supposed to find
    :param left: The left index for the search
    :param right: The right index for the search
    :return: index of the element if found, -1 if element not found.
    """
    if left <= right:
        mid = int((left + right) / 2)

        if element == array[mid]:
            return mid
        elif element < array[mid]:
            return reverse_binary_search(array, element, mid + 1, right)
        else:
            return reverse_binary_search(array, element, left, mid - 1)

    return -1


def find_element():
    """
    Orchestrator method which sorts and executes the list of elements
    :return: void
    """
    elements = [10, 34, 48, 59, 63, 74, 85, 120, 140]
    descending_sort(elements)
    print("Element %d found at %d\n" % (120, reverse_binary_search(elements, 120, 0, len(elements) - 1)))


find_element()

"""
Q2. Given a group of n (index: 0-n-1) marbles, all marbles have a specific
number based on their type. These marbles are stored in an array. Find the
marble type [i.e: its number] which has occurred maximum times in an
array.
Can the above solution be solved in less than O(nlogn) time complexity. If
yes provide the approach. Justify your answer.

Sol: The given question is a classic case of binary search. I'm assuming that marbles are all represented as numbers
and are already given to us in sorted order. For the scenario we can have a special case of binary search where it would
return us the position of the first and last instance of the element in the given list. We will call binary search first
where we will find left most index of the element and another time to find right most index. Once we have both the
indexes we can simply find the solution.
"""


def find_first(array, element, left, right):
    """
    The given function is a customized approach of binary search. We perform binary search with below conditions:

    If middle_element is the target_number
        Call binary_search(left_index, middle) :
        Try to find if there are more instances of this element to the left with same middle as the index
    else if middle_element > target_number
        Call binary_search(left_index, middle -1 ) :
            Try to find if there are more instances of this element to the left by reducing the index
    else
        Call binary_search(middle + 1, right_index) :
            Try to find if there are more instances of this element to the right

    Once right_index - left_index <= 1
        if left_index is element return left element
        else if right_index is element return right element
        else -1

    :param array: Given list of elements in ascending order
    :param element: The element supposed to find
    :param left: left index of element
    :param right: right index of element
    :return: -1 if element not found, index otherwise
    """
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
    """
    The given function is a customized approach of binary search. We perform binary search with below conditions:

    If middle_element is the target_number
        Call binary_search(middle, right_index) :
        Try to find if there are more instances of this element to the right with same middle as the index
    else if middle_element > target_number
        Call binary_search(left_index, middle -1 ) :
            Try to find if there are more instances of this element to the left by reducing the index
    else
        Call binary_search(middle + 1, right_index) :
            Try to find if there are more instances of this element to the right

    Once right_index - left_index <= 1
        if right_index is element return right element
        else if left_index is element return left element
        else -1

    :param array: Given list of elements in ascending order
    :param element: The element supposed to find
    :param left: left index of element
    :param right: right index of element
    :return: -1 if element not found, index otherwise
    """
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


def count_elements():
    """
    Orchestrator method which find the counts the number of elements in the list
    :return: void
    """
    elements = [1, 3, 3, 4, 4, 5, 6, 7]
    for each in range(0, 6):
        left_index = find_first(elements, each, 0, len(elements) - 1)
        if left_index < 0:
            print("Element %d not in the list" % each)
            continue
        right_index = find_last(elements, each, 0, len(elements) - 1)
        print("Count of %d in list is %d" % (each, (right_index - left_index) + 1))


count_elements()
