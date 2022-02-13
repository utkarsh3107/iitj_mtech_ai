package dsp.assignment2;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.LinkedList;
import java.util.Queue;

public class Ass2_Q1 {
    
    public static void main(String[] args) throws Exception{
        BufferedReader reader = new BufferedReader(new InputStreamReader(System.in)); 
        Queue<Integer> b = new LinkedList<>();

        //Reading inputs from user
        int N = Integer.parseInt(reader.readLine());
        int a_size = Integer.parseInt(reader.readLine());
        String[] input = reader.readLine().split(" ");
        //Counter array which will store number of times a student has donated blood.
        int[] counter = new int[N+1];
        int[] a = new int[a_size];
    
        //Converting string input to integer list
        for(int i = 0; i < a_size; i++){
            a[i] = Integer.parseInt(input[i]);
        }
    
        //Iterating over each element in list a 
        for(int each : a){
            //Incrementing the blood donated counter for that student
            counter[each]++;
            //Adding the student to queue only if the student has donated blood for the first time
            if(counter[each] <= 1){
                b.add(each);
            }
    
            //If the student has only given blood once, print the front of queue
            if(counter[each] <= 1){
                System.out.print(b.peek() + " ");
            }else{
                //Iterate over the queue until you find the student who has donated blood only once.
                while(!b.isEmpty() && counter[b.peek()] > 1){
                    b.poll();
                }
    
                //If the list b is empty then print 0 else print the front of queue b.
                if(b.isEmpty()){
                    System.out.print("0 ");
                }else{
                    System.out.print(b.peek() +" ");
                }
            }
        }
    }
}
