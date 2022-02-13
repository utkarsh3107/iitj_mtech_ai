class BST:

    def __init__(self, data=None):
        self.left = None
        self.right = None
        self.data = data

    def insert(self, data):

        if data == self.data:
            return

        if data < self.data:
            # Add data in left subtree
            if self.left:
                self.left.insert(data)
            else:
                self.left = BST(data)
        else:
            if self.right:
                self.right.insert(data)
            else:
                self.right = BST(data)

    def max_height(self, node):
        if node is None:
            return 0;

        else:

            left_subtree = node.max_height(node.left)
            right_subtree = node.max_height(node.right)

            if left_subtree > right_subtree:
                return left_subtree + 1
            else:
                return right_subtree + 1

    def getMinimumKey(self, curr):
        if curr is not None:
            while curr.left:
                curr = curr.left

        return curr

    def delete(self, root, key):
        parent = None
        curr = root

        while curr and curr.data != key:
            parent = curr

            if key < curr.data:
                curr = curr.left
            else:
                curr = curr.right

        if curr is None:
            return root

        if curr.left is None and curr.right is None:

            if curr != root:
                if parent.left == curr:
                    parent.left = None
                else:
                    parent.right = None

            else:
                root = None

        elif curr.left and curr.right:
            successor = self.getMinimumKey(curr.right)
            val = successor.data
            self.delete(root, successor.data)
            curr.data = val

        else:
            if curr.left:
                child = curr.left
            else:
                child = curr.right

            if curr != root:
                if curr == parent.left:
                    parent.left = child
                else:
                    parent.right = child

            else:
                root = child

        return root

    def search(self, root, element):
        height = 0
        width = 1
        while root is not None:
            if root.data == element:
                return height, width

            if root.data < element:
                root = root.right
                width = width * 2
            else:
                root = root.left
                width = (width * 2) - 1

            height = height + 1

        return -1

    def print(self, root):
        if root is BST:
            return

        result = []
        queue = [root]

        curr_height = 0
        height = self.max_height(root)

        while True:
            count = len(queue)

            if curr_height >= height:
                break

            result.append('{')
            while count > 0:
                temp = queue.pop(0)

                if temp.data == 'x':
                    if curr_height >= height:
                        count = count - 1
                        continue
                else:
                    result.append(temp.data)

                if count > 1:
                    result.append(',')

                if temp.left:
                    queue.append(temp.left)
                else:
                    queue.append(BST('x'))

                if temp.right:
                    queue.append(temp.right)
                else:
                    queue.append(BST('x'))

                count = count - 1

            result.append('}')
            result.append(',')
            curr_height = curr_height + 1

        del result[-1]
        for each in result:
            print(each, end=' ')


nums = [12, 6, 18, 19, 21, 11, 3, 5, 4, 24, 18]
bst = BST(nums[0])

for num in range(1, len(nums)):
    bst.insert(nums[num])

# bst.print(bst)

bst.delete(bst, 12)
bst.print(bst)
# for num in range(0, len(nums)):
# print(bst.search(bst, nums[num]))
# print(bst.search(bst, num))
# print(bst.search(bst, num))

# print(isinstance(bst.search(bst, 10), int))
