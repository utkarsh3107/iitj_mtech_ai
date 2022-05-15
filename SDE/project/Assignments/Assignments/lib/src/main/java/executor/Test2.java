package executor;

public class Test2 {
    
    public void merge(int[] arr, int leftIndex, int middle, int rightIndex){
        System.out.println("3: ("+leftIndex+", "+middle+", "+rightIndex+")");

        //create subarrays
        int[] leftArray = new int[middle - leftIndex];
        int[] rightArray = new int[rightIndex - middle];

        for(int i = 0; i < leftArray.length;i++){
            leftArray[i] = arr[leftIndex + i];
        }

        for(int i = 0; i < rightArray.length;i++){
            rightArray[i] = arr[middle + i];
        }

        int leftCounter = 0,rightCounter = 0;
        int arrayPos = leftIndex;
        while(leftCounter < leftArray.length && rightCounter < rightArray.length){
            if(leftArray[leftCounter] <= rightArray[rightCounter]){
                arr[arrayPos] = leftArray[leftCounter];
                leftCounter++;
            }else{
                arr[arrayPos] = rightArray[rightCounter];
                rightCounter++;
            }
            arrayPos++;
        }

        while(leftCounter < leftArray.length){
            arr[arrayPos] = leftArray[leftCounter];
            leftCounter++;
            arrayPos++;
        }


        while(rightCounter < rightArray.length){
            arr[arrayPos] = rightArray[rightCounter];
            rightCounter++;
            arrayPos++;
        }
    }

    public void mergeSort(int[] arr, int leftIndex, int rightIndex){
        if(leftIndex < rightIndex){
            int mid = (leftIndex + rightIndex - 1 ) /2;
            System.out.println("1: ("+leftIndex+", "+mid+", "+rightIndex+")");
            mergeSort(arr, leftIndex, mid);
            System.out.println("2: ("+leftIndex+", "+mid+", "+rightIndex+")");
            mergeSort(arr, mid + 1, rightIndex);
            merge(arr, leftIndex, mid, rightIndex);
        }
    }

    public void print(int[] arr){
        for(int each : arr){
            System.out.print(each + " ");
        }
    }


    public static void main(String[] args){
        int[] arr = new int[] {5,4,3,2,1};
        Test2 obj = new Test2();
        obj.mergeSort(arr,0, arr.length - 1);
        obj.print(arr);
    }
}
