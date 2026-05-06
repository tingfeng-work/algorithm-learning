import java.util.List;

public class Test {
   public static void main(String[] args) {
        int[] nums = {0,0,0,1000000000,1000000000,1000000000,1000000000};
        int target = 1000000000;
        Solution solution = new Solution();
        List<List<Integer>> lists = solution.fourSum(nums, target);
        System.out.println(lists);
    }
}
