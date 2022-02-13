package dsp.assignment2;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Stack;

public class Ass2_Q3 {
    
    public static void main(String[] args) throws Exception{
        BufferedReader reader = new BufferedReader(new InputStreamReader(System.in)); 
        Stack<Integer> s1 = new Stack<>();
        Stack<Integer> s2 = new Stack<>();
        List<Integer> delayed = new ArrayList<>();
        
        //Reading input from the user
        int N = Integer.parseInt(reader.readLine());
        int[] output = new int[N];
        String[] s1_str = reader.readLine().split(" ");
        String[] output_str = reader.readLine().split(" ");

        //Converting the input from list to stack
        for(int i = 0; i < s1_str.length; i++){
            s1.push(Integer.parseInt(s1_str[i]));
        }

        //Converting output expected list to integer
        for(int i = 0; i < output_str.length; i++){
            output[i] = Integer.parseInt(output_str[i]);
        }

        // Iternating over stack and if the element matches the curr element of stack put it in s2
        // otheriwse add the element to delayed stack 
        int output_ptr = 0;
        for(int i =0 ; i < N; i++){
            int curr = s1.pop();
            if(output[output_ptr] ==  curr){
                s2.push(curr);
                output_ptr++;
            }else{
                delayed.add(curr);
            }
        }

        // Merge delayed array with stack s2
        for(int i = 0 ; i< delayed.size(); i++){
            s2.push(delayed.get(i));
        }

        //Compare s2 and delayed stack if elements match print 'YES' else print 'NO'
        for(int i = N - 1; i >= 0 ; i--){
            if(s2.pop() != output[i]){
                System.out.println("NO");
                return;
            }
        }

        System.out.println("YES");
    }
}
