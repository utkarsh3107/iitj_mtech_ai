import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.Queue;

public class Q1{
    public static void main(String args[]) throws IOException{
    	System.out.println("Hello developer");
        int no_of_students,list_size;
        String input;
        
        BufferedReader reader = new BufferedReader(new InputStreamReader(System.in)); 
        Queue<Integer> queue = new LinkedList<>();
        System.out.println("enter no of students");
        input=reader.readLine();
        no_of_students=Integer.parseInt(input);
        System.out.println("enter the list size");
        input=reader.readLine();
        list_size=Integer.parseInt(input);
        int list_a[] = new int[no_of_students+1];
        
        System.out.println("read the blood donation students sequence");
        input=reader.readLine();
        System.out.println("input: "+input);
        
        int i=0;
        int val=0;
        for(String each:input.split("\\ ")) {
        	val=Integer.parseInt(each);
        	queue.add(val);
        	++list_a[val];
        	//System.out.println("queue.peek(): "+queue.peek()+" number: "+list_a[queue.peek()]);
        	while(!queue.isEmpty() && list_a[queue.peek()]!=1)
        		queue.poll();
        	if(queue.isEmpty())
    			System.out.print("0 ");
    		else
    			System.out.print(queue.peek()+" ");
        }
        reader.close();
    }
}