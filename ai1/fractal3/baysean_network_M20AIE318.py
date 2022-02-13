import itertools

def hacker_input():
    node_values = {}
    adj_matrix = []
    test_cases = []
    n = 0
    N = int(input())
    #print(N)
    n = n + N

    for each in range(1, n + 1):
        node_values[each - 1] = tuple(i.strip() for i in input().rstrip("\n").split(","))

    for each in range(n + 1, (2 * n) + 1):
        adj_matrix.append([int(i) for i in input().rstrip("\n").split(" ")])

    m = int(input())
    #print(m)

    test_counts = {}
    for each in range((2 * n) + 2, m + (2 * n) + 2):
        test_case = tuple(input().rstrip("\n").split(","))
        test_cases.append(test_case)
        i = 0

        for node_value in test_case:
            each_dict = {}

            if i in test_counts:
                each_dict = test_counts[i]
            else:
                for each_node_val in node_values[i]:
                    each_dict[each_node_val] = 0
                test_counts[i] = each_dict

            val = each_dict[node_value]
            each_dict[node_value] = val + 1
            i = i + 1

    return N, node_values, adj_matrix, m, test_cases, test_counts

def read_input():
    val = "test.txt"

    # Using readlines()
    file1 = open(val, 'r')

    node_values = {}
    adj_matrix = []
    test_cases = []
    input_list = file1.readlines()
    n = 0
    N = int(input_list[n].rstrip("\n"))

    #print(N)
    n = n + N

    for each in range(1, n + 1):
        node_values[each - 1] = tuple(i.strip() for i in input_list[each].rstrip("\n").split(","))
    #print(node_values)

    for each in range(n + 1, (2 * n) + 1):
        adj_matrix.append([int(i) for i in input_list[each].rstrip("\n").split(" ")])

    #print(adj_matrix)

    m = int(input_list[(2 * n) + 1])
    #print(m)

    test_counts = {}
    for each in range((2 * n) + 2, m + (2 * n) + 2):
        test_case = tuple(input_list[each].rstrip("\n").split(","))
        test_cases.append(test_case)
        i = 0

        for node_value in test_case:
            each_dict = {}

            if i in test_counts:
                each_dict = test_counts[i]
            else:
                for each_node_val in node_values[i]:
                    each_dict[each_node_val] = 0
                test_counts[i] = each_dict

            val = each_dict[node_value]
            each_dict[node_value] = val + 1
            i = i + 1

    #print(test_counts)
    #print(test_cases)

    return N, node_values, adj_matrix, m, test_cases, test_counts


def create_truth_table(node, values, parents):
    return list(itertools.product(values[node], repeat=len(parents) + 1))


def convert(_list):
    return tuple(_list)


def baysean_network():
    tuple = hacker_input()
    N = tuple[0]
    values = tuple[1]
    adj_matrix = tuple[2]
    m = tuple[3]
    testcases = tuple[4]
    test_counts = tuple[5]

    result = {}
    for row in range(0, N):
        parents = []
        for col in range(0, N):
            if adj_matrix[col][row] == 1:
                parents.append(col)

        if len(parents) == 0:
            val_dict = test_counts[row]
            probs = []
            for key, val in test_counts[row].items():
                probs.append(format(val / m, ".4f"))
            result[row] = probs
        else:
            truth_table = create_truth_table(row, values, parents)
            elements = len(parents) + 1
            total = parents
            total.insert(0, row)
            temp_dict = {}
            result[row] = temp_dict
            pos = 0

            for each_case in truth_table:
                temp_dict[pos] = 0
                for testcase in testcases:
                    new_case = []
                    for each in total:
                        new_case.append(testcase[each])
                    each_test_case = convert(new_case)
                    if each_test_case == each_case:
                        count = temp_dict[pos]
                        temp_dict[pos] = count + 1
                pos = pos + 1

    for key, value in result.items():
        if type(value) == list:
            for each in value:
                print(each, end=" ")
            print()
        else:
            split = len(values[key])
            mid = int(len(value) / split)
            for curr in range(0, mid):
                total = 0
                for temp_test in range(curr, len(value), mid):
                    total = total + value[temp_test]
                for temp_test in range(0, len(value), mid):
                    if total == 0:
                        value[curr + temp_test] = format(0, ".1f")
                    else:
                        int_val = int(value[curr + temp_test] / total)
                        float_val = value[curr + temp_test] / total
                        if float_val - int_val == 0:
                            value[curr + temp_test] = format(int_val, ".1f")
                        else:
                            value[curr + temp_test] = format(float_val, ".4f")
            for each in value.values():
                print(each, end=" ")
            print()


baysean_network()
