package wallmart;

public class Test {

    private Node head;

    public static void main(String[] args){
        Test test = new Test();
        test.insert(5);
        test.insert(2);
        test.insert(1);
        test.insert(3);
        test.insert(4);
        test.printInorder();
    }

    public void printInorder(){
        inorder(this.head);
    }

    public void inorder(Node curr){
        if (curr != null) {
            inorder(curr.left);
            System.out.println(curr.key);
            inorder(curr.right);
        }
    }

    public void insert(int key){
        this.head = insert(this.head, key);
    }

    public Node insert(Node curr, int key){
        //If head is null then simply insert the element
        if(curr == null){
            curr =  new Node(key);
            return curr;
        }

        if(curr.key < key){
            curr.right =  insert(curr.right, key);
        }else if(curr.key > key){
            curr.left =  insert(curr.left, key);
        }

        return curr;
    }
}


class Node{
    int key;
    Node left, right;

    Node(int key){
        this.key = key;
        this.left = null;
        this.right = null;
    }
}