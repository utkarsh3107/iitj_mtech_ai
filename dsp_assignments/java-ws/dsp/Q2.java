import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Stack;

public class Q2 {
	public static void main(String args[]) throws NumberFormatException, IOException {
		String input;
		int stack_size;
		Stack<Integer> stack = new Stack<>();
		BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
		
		//reading the stack size
		stack_size = Integer.parseInt(reader.readLine());
		
		// reading the stack elements
		input = reader.readLine();
		int i=0;
		for(String each:input.split("\\ ")) {
        	if(i+1==stack_size)
        		break;
		stack.add(Integer.parseInt(each));
		}
		
		// reading the order list
		input = reader.readLine();
		String[] list=input.split("\\ ");
		i=0;
		for(;!stack.isEmpty() && stack.pop()==Integer.parseInt(list[i]);i++);		
		if((!stack.isEmpty()) || (i!=list.length)) {
			System.out.println("NO");
		}else {
			System.out.println("YES");
		}
	}
}