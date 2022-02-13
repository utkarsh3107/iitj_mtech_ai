package dsp.assignment2;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Stack;

public class Ass2_Q2{
    
    public static void main(String[] args) throws Exception{
        BufferedReader reader = new BufferedReader(new InputStreamReader(System.in)); 
        Stack<Integer> s1 = new Stack<>();
        Stack<Integer> s2 = new Stack<>();
        Stack<Integer> s3 = new Stack<>();

         //Reading input from the user
         int N = Integer.parseInt(reader.readLine());
         int[] output = new int[N];
         String[] s1_str = reader.readLine().split(" ");
         String[] output_str = reader.readLine().split(" ");
 
         /*
         int N = 5;
         int[] output = new int[N];
         String[] s1_str = {"5", "4", "3", "2", "1"};
         //String[] output_str = {"1", "2", "3", "4", "5"};
         //String[] output_str = {"1", "2", "3", "5", "4"};
         String[] output_str = {"3", "1", "4", "2", "5"};*/

         //Converting the input from list to stack
         for(int i = 0; i < s1_str.length; i++){
             s1.push(Integer.parseInt(s1_str[i]));
         }
 
         //Converting output expected list to integer
         for(int i = 0; i < output_str.length; i++){
             output[i] = Integer.parseInt(output_str[i]);
         }

        // Iternating over stack and if the element matches the curr element of stack put it in s2
        // otheriwse put the element in s3
        int output_ptr = 0;
        for(int i =0 ; i < N; i++){
            int curr = s1.pop();
            if(output[output_ptr] ==  curr){
                s2.push(curr);
                output_ptr++;
            }else{
                s3.push(curr);
            }
        }

        //Merging stack s3 with stack s2
        while(!s3.isEmpty()){
            s2.push(s3.pop());
        }

        //Compare s2 and output stack if elements match print 'YES' else print 'NO'
        for(int i = N - 1; i >= 0 ; i--){
            if(s2.pop() != output[i]){
                System.out.println("NO");
                return;
            }
        }

        System.out.println("YES");
    }
}
