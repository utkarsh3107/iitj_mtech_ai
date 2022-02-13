# Actual Code
def compute(pe, ne):
    pass

def split_input(kb):
    positive_elements = []
    negative_elements = []
    for each in kb:
        flag = True
        for every in each:
            if every.find('!') >= 0:
                negative_elements.append(each)
                flag = False
                break
        if flag:
            positive_elements.append(each)
    return positive_elements, negative_elements


def parse_input(line):
    knowledge_base = []
    for each in line:
        _element = each.split('|')
        knowledge_base.append(_element)

    return split_input(knowledge_base)


def iter_input(lines):
    knowledge_base = []
    first = True
    last = True
    count = 0

    # Strips the newline character
    for line in lines:
        count += 1

        if first:
            first = False
            continue

        knowledge_base.append(line.strip())
        #print("Line{}: {}".format(count, line.strip()))
    expected = knowledge_base.pop()
    print(parse_input(knowledge_base))


def read_input():
    # val = input("Enter your value: ")
    # print(val)

    val = "file1.txt"

    # Using readlines()
    file1 = open(val, 'r')
    iter_input(file1.readlines())


read_input()
