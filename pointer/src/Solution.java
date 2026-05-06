import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public class Solution {
    public int[] twoSum(int[] numbers, int target) {
        /* 有序非递减数组中找出两个数和为 target，返回数组下标（index1，index2），
        规定 index1 < index2，每个输入对应唯一答案
        * 思路：相向双指针，指向最小与最大，如果指向元素之和大于 target，最大的移动，反之最小的移动
        优化：先判断是否与 target 相等，相等直接返回；后判断大于小于
        * */
        int right = numbers.length - 1;
        for (int left = 0; left < right; ) {
            int sum = numbers[left] + numbers[right];
            if (sum == target) return new int[]{left + 1, right + 1};
            if (sum > target) {
                right--;
            } else {
                left++;
            }
        }
        return null;
    }

    public List<List<Integer>> threeSum(int[] nums) {
        /*题目：三数之和
         * 一个整数数组 nums，判断是否存在三元组
         * [nums[i], nums[j], nums[k]] 满足 i != j、i != k 且 j != k ，同时还满足 nums[i] + nums[j] + nums[k] == 0
         * 返回所有和为 0 的不重复三元组
         * 思路：nums[i] + nums[j] + nums[k] == 0 转化为 -nums[i] == nums[j] + nums[k]；
         * 再将数组排序，转化为两数和
         * 优化：在 i 和 j 跳过重复数字时，只有在记录答案时才需要跳过
         * */
        // 构造答案
        List<List<Integer>> ans = new ArrayList<>();
        // 数组排序
        Arrays.sort(nums);
        int len = nums.length;
        for (int i = 0; i < len - 2; i++) {
            // 枚举 nums[i]
            if (nums[i] + nums[i + 1] + nums[i + 2] > 0) break;
            if (nums[i] + nums[len - 2] + nums[len - 1] < 0) continue;
            if (i > 0 && nums[i] == nums[i - 1]) continue;
            int left = i + 1;
            int right = len - 1;
            while (left < right) {
                int sum = nums[i] + nums[left] + nums[right];
                if (sum > 0) {
                    right--;
                } else if (sum < 0) {
                    left++;
                } else {
                    ans.add(List.of(nums[i], nums[left], nums[right]));
                    for (left++; left < right && nums[left] == nums[left - 1]; left++) ;// 跳过重复数字
                    for (right--; left < right && nums[right] == nums[right + 1]; right--) ;// 跳过重复数字
                }

            }
        }
        return ans;
    }

    public int countPairs(List<Integer> nums, int target) {
        //题目：2824：统计和小于目标的下标对数目
        //给你一个下标从 0 开始长度为 n 的整数数组 nums 和一个整数 target ，
        //请你返回满足 0 <= i < j < n 且 nums[i] + nums[j] < target 的下标对 (i, j) 的数目。
        // 思路：因为是统计数目，所以对数组排序并不影响最终的答案，所以排序后就转化两数之和的问题
        // 排序
        Collections.sort(nums);
        int ans = 0;

        for (int i = 0; i < nums.size(); i++) {
            int j = nums.size() - 1;
            while (i < j) {
                int sum = nums.get(i) + nums.get(j);
                if (sum >= target) {
                    j--;
                } else {
                    ans = ans + j - i;
                    break;
                }
            }
        }

        return ans;
    }

    public int threeSumClosest(int[] nums, int target) {
        /*题目：16 最接近的三数之和
         * 在数组中找到三个不同元素之和最接近 target，返回最接近的和的值
         * 每组输入只有一个解
         * 思路：返回的三数之和的值，所以对数组进行排序不影响最后的结果
         * 对数组排序后，转化为类似三数之和
         * 枚举 target-nums[i] == nums[j] + nums[k]
         * 遍历一次数组，记录过程中最接近的值
         * */
        Arrays.sort(nums);

        int ans = Integer.MAX_VALUE / 2;
        int len = nums.length;

        for (int i = 0; i < len - 2; i++) {
            if (i > 0 && nums[i] == nums[i - 1])
                continue;
            int s = nums[i] + nums[i + 1] + nums[i + 2];
            if (s > target) {
                if (Math.abs(s - target) < Math.abs(ans - target))
                    ans = s;
                break;
            }
            s = nums[i] + nums[len - 1] + nums[len - 2];
            if (s < target) {
                if (Math.abs(s - target) < Math.abs(ans - target))
                    ans = s;
                continue;
            }
            int j = i + 1;
            int k = len - 1;
            while (j < k) {
                int sum = nums[i] + nums[j] + nums[k];
                if (sum == target) {
                    ans = sum;
                    return ans;
                }
                if (Math.abs(sum - target) < Math.abs(ans - target))
                    ans = sum;
                if (sum < target) {
                    j++;
                } else {
                    k--;
                }
            }
        }
        return ans;
    }

    public List<List<Integer>> fourSum(int[] nums, int target) {
        /* 题目：18.四数之和
         * 给定一个数组，找出四元组的和为target且不重复，可以按任意顺序返回答案
         * 思路：由于按任意顺序返回答案，所以对nums排序不影响结果，
         * 排序后 nums[a] + nums[b] + nums[c] + nums[d] == target
         * 可以转化为枚举nums[a] 满足 nums[b] + nums[c] + nums[d] = target - nums[a] 三数之和，以此类推
         * */
        Arrays.sort(nums);

        List<List<Integer>> ans = new ArrayList<>();

        int len = nums.length;
        for (int a = 0; a < len - 3; a++) {
            if (a > 0 && nums[a] == nums[a - 1]) continue; // 当前元素与上一个相同
            long s = 0L + nums[a] + nums[a + 1] + nums[a + 2] + nums[a + 3];
            if (s > target) break; // 最小都大于target
            s = 0L + nums[a] + nums[len - 1] + nums[len - 2] + nums[len - 3];
            if (s < target) continue; //最大都小于target
            for (int b = a + 1; b < len - 2; b++) {
                if (b > a + 1 && nums[b] == nums[b - 1]) continue;
                s = 0L + nums[b] + nums[b + 1] + nums[b + 2];
                if (s > target - nums[a]) break;
                s = 0L + nums[b] + nums[len - 1] + nums[len - 2];
                if (s + nums[a] < target) continue;
                int c = b + 1;
                int d = len - 1;
                while (c < d) {
                    int sum = nums[a] + nums[b] + nums[c] + nums[d];
                    if (sum == target) {
                        ans.add(List.of(nums[a], nums[b], nums[c], nums[d]));
                        for (c++; c < d && nums[c] == nums[c - 1]; c++) ;//跳过重复
                        for (d--; c < d && nums[d] == nums[d + 1]; d--) ;//跳过重复
                    } else if (sum < target) {
                        c++;
                    } else {
                        d--;
                    }
                }
            }
        }

        return ans;
    }

    public int triangleNumber(int[] nums) {
        /*611. 有效三角形的个数
         * 给定一个包含非负整数的数组 nums ，返回其中可以组成三角形三条边的三元组个数。
         * 三角形满足两边之和大于第三边，所以也就转化为枚举 nums[a] 求满足 nums[b]+nums[c] > nums[a] 的个数
         * 由于是个数问题，排序不影响答案，先对数组排序
         * 倒着枚举满足两小边之和大于大边，则符合题意
         * */
        int ans = 0;

        Arrays.sort(nums);
        for (int c = nums.length - 1; c >= 2; c--) {
            int x = nums[c];
            int s = nums[c - 1] + nums[c - 2];
            if (s < x) continue;
            int left = 0;
            int right = c - 1;
            while (left < right) {
                int sum = nums[left] + nums[right];
                if (sum > x) {
                    ans = ans + right - left;
                    right--;
                } else {
                    left++;
                }
            }
        }

        return ans;
    }
}
