from queue import Queue
## Problem 1

n = int(input())
print(n)
A_size = int(input())
print(A_size)
A = [int(x) for x in input().split()]  

#B = Queue(maxsize = A_size)
B = [0] * A_size
counter = [0] * A_size

for each in A:
    B.put(each)
    counter[each] = counter[each] + 1

    if counter[each] <= 1:
        print(B.)



