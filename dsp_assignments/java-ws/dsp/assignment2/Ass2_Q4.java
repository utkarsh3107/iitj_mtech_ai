package dsp.assignment2;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.Stack;

public class Ass2_Q4 {

    public static void main(String[] args) throws Exception
    {
        BufferedReader reader = new BufferedReader(new InputStreamReader(System.in)); 
        Stack<Integer> s = new Stack<>();
        Stack<Integer> input = new Stack<>();

        int N = Integer.parseInt(reader.readLine());
        int nge[] = new int[N];
        String[] input_str = reader.readLine().split(" ");
        int[] arr = new int[N];

        //Converting string input to integer list
        for(int i = 0; i < N; i++){
            arr[i] = Integer.parseInt(input_str[i]);
        }
        
        //Converting the input from list to stack
        for(int i = 0; i < arr.length; i++){
            input.push(arr[i]);
        }

        int ptr = 0;
        while(!input.isEmpty()){
            int curr = input.pop();
            if (!s.empty())
            {
                while (!s.empty() && s.peek() <= arr[ptr])
                {
                    s.pop();
                }
            }
            nge[ptr] = s.empty() ? -1 : s.peek();
            s.push(arr[ptr]);
        }

        // iterate for rest of the elements
        for (int i = N - 1; i >= 0; i--)
        {
            if (!s.empty())
            {
                while (!s.empty() && s.peek() <= arr[i])
                {
                    s.pop();
                }
            }
            nge[i] = s.empty() ? -1 : s.peek();
            s.push(arr[i]);
        }
        for (int i = 0; i < arr.length; i++)
            System.out.print(nge[i] + " ");
    }

}
