package wallmart;

import java.io.*;
import java.util.HashMap;
import java.util.Map;

public class Test2 {

    private Map<String, String> value = new HashMap<>();


    // start with {
    // keys will be before :
    // values will be after :
    // each key, value will be identified via a start of " end of "
    // end with }

    public void read(){
        State currState = State.START;
        try(BufferedReader reader = new BufferedReader(new FileReader("/Users/utkarsh/Desktop/study/iitj_mtech_ai/GTA/project/Assignments/lib/src/main/resources/input.json"))){
            String currLine = null;
            while((currLine = reader.readLine()) != null) {
                currLine = currLine.trim();
                if(currLine.equals("{") || currLine.equals("}") || currLine.equals("")){
                    //TODO Check someething here
                    continue;
                }

                String[] input = currLine.split(":");

                if(input.length > 2){
                    //Not a valid json. Throw exception
                    throw new IllegalArgumentException();
                }
                value.put(input[0].substring(input[0].indexOf("\""), input[0].lastIndexOf("\"")),
                        input[1].substring(input[1].indexOf("\""), input[1].lastIndexOf("\"")));
            }
            System.out.println(value);
        }catch(Exception ex){
            ex.printStackTrace();
        }
    }

    public void read(String input){
        input = input.trim();

        if(!input.startsWith("{") && input.endsWith("}")){
            //TODO Invalid string
            return;
        }

        String newStr = input.substring(1, input.length() - 1);

        String[] inputs = newStr.split(",");

        for(String eachInput : inputs){

        }
    }

    public static void main(String[] args){
        new Test2().read();
    }
}

enum State{
    START,END,MIDDLE;
}