import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.LinkedList;
import java.util.Queue;
import java.util.Stack;

public class Q3 {
	public static void main(String args[]) throws NumberFormatException, IOException {
		String input;
		int stack_size;
		Stack<Integer> stack1 = new Stack<>();
		Stack<Integer> stack2 = new Stack<>();
		BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
		
		//reading the stack size
		stack_size = Integer.parseInt(reader.readLine());
		Queue<Integer> delayed = new LinkedList<>();
		// reading the stack elements
		input = reader.readLine();
		int i=0;
		for(String each:input.split("\\ ")) {
        	if(i+1==stack_size)
        		break;
		stack1.add(Integer.parseInt(each));
		}
		
		// reading the order list
		input = reader.readLine();
		String[] list=input.split("\\ ");
		int val;
		for(i=0;i<list.length;i++) {
			if(!stack1.isEmpty()) {
				val=stack1.pop();
				if(Integer.parseInt(list[i])==val) {
					stack2.push(val);
				}else {
					delayed.add(val);
					// if we are not at final element
					if(!stack1.isEmpty()) {
						stack2.push(stack1.pop());
					}
					// check if we are at last element
					if(stack1.isEmpty() && list.length%2==0)
						--i;
				}	
			}else {
				val=delayed.poll();
				//System.out.println("poping ele from queue: "+val);
				//System.out.println("poping ele from list: "+list[i]);
				if(Integer.parseInt(list[i])==val) {
					stack2.push(val);
				}
			}
		}
		/*
		Iterator<Integer> iterator = stack2.iterator();
		while(iterator.hasNext()) {
			System.out.print(iterator.next()+", ");
		}
		iterator = delayed.iterator();
		System.out.println("");
		while(iterator.hasNext()) {
			System.out.print(iterator.next()+", ");
		}*/
		
		if(stack2.size()==list.length) {
			System.out.println("Yes");
		}else {
			System.out.println("No");
		}
	}
}