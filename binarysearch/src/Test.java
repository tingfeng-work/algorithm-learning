import java.util.Arrays;

public class Test {
    public static void main(String[] args) {
        int[] nums = {1,2,3,4};
        int[] ints = Arrays.copyOfRange(nums, 0, 2);
        for (int anInt : ints) {
            System.out.println(anInt);
        }
    }
}
