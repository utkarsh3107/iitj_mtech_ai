package executor;

public class Test {

    private Node head = null;

    public boolean insert(int val){
        Node newNode = new Node(val);
        //if head is empty
        if(head == null){
            head = newNode;
            return true;
        }

        Node current = head;
        Node previous = null;
        while(current.next != null && val > current.val){
            //ITERATE until we reach a node which is greater than new node
            previous = current;
            current = current.next;
        }

        if(previous == null){
            //UPDATE HEAD HERE
            head = newNode;
            newNode.next = current;
        }else{
            //HEAD is not the smallest
            previous.next = newNode;
            newNode.next = current;
        }
        return true;
    }

    public void print(){
        Node curr = head;
        while(curr != null){
            System.out.print(curr.val + " ");
            curr = curr.next;
        }
    }

    public static void main(String[] args){
        Test test = new Test();
        test.insert(5);
        test.insert(4);
        test.insert(3);
        test.insert(2);
        test.insert(1);
        test.print();
    }
}

class Node{
    Node next;
    int val;

    Node(int val){
        this.val = val;
    }
}
