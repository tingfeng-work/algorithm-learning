import java.awt.event.HierarchyBoundsAdapter;
import java.util.*;

public class Solution {
    public int[] dailyTemperatures(int[] temperatures) {
        /*
         * 739. 每日温度
         * 给定一个整数数组 temperatures ，表示每天的温度，返回一个数组 answer ，
         * 其中 answer[i] 是指对于第 i 天，下一个更高温度出现在几天后。
         * 如果气温在这之后都不会升高，请在该位置用 0 来代替。
         * 思路：暴力枚举需要 O(n*2)，怎么优化
         * 使用单调递增栈维护下一个更大元素的位置，这样每次遍历到元素时，可以快速找到比它大的第一个元素的位置
         * */
        int n = temperatures.length;
        int[] ans = new int[n];
        Deque<Integer> st = new ArrayDeque<>();

        for (int i = n - 1; i >= 0; i--) {
            int t = temperatures[i];
            while (!st.isEmpty() && t >= temperatures[st.peek()]) {
                st.pop();
            }
            if (!st.isEmpty()) {
                Integer j = st.peek();
                ans[i] = j - i;
            }
            st.push(i);
        }
//        for (int i = 0; i < n; i++) {
//            int t = temperatures[i];
//            while (!st.isEmpty() && t > temperatures[st.peek()]) {
//                Integer j = st.pop();
//                ans[j] = i-j;
//            }
//            st.add(i);
//        }


        return ans;


    }

    public int trap(int[] height) {
        /*
         * 42. 接雨水
         * 给定 n 个非负整数表示每个宽度为 1 的柱子的高度图，计算按此排列的柱子，下雨之后能接多少雨水。
         * 思路：之前的双向指针是纵向计算，也就假设每个位置有水桶
         * 利用单调栈的思路可以实现横着计算
         * 首先接水需要找到高度差，以及宽度，而高度差是由下一个更大的元素决定的，所以可以用单调栈
         * */
        int n = height.length;
        int ans = 0;
        Deque<Integer> st = new ArrayDeque<>(n);

        for (int i = 0; i < n; i++) {
            int h = height[i];
            while (!st.isEmpty() && h >= height[st.peek()]) {
                Integer bottom = st.pop();
                if (st.isEmpty())
                    break;
                int left = st.peek();
                ans += (Math.min(height[left], h) - height[bottom]) * (i - left - 1);
            }
            st.push(i);
        }
        return ans;

    }

    public int[] maxSlidingWindow(int[] nums, int k) {
        /*
         * 239. 滑动窗口最大值
         * 给你一个整数数组 nums，有一个大小为 k 的滑动窗口从数组的最左侧移动到数组的最右侧。
         * 你只可以看到在滑动窗口内的 k 个数字。滑动窗口每次只向右移动一位。返回 滑动窗口中的最大值 。
         * 思路：维护一个单调递减的队列，遍历到元素 x 时，判断它与队尾元素的大小，如果大于队尾元素，则循环删除队尾
         * 此时队首元素就是窗口中的最大值
         * */
        int n = nums.length;
        Deque<Integer> deque = new ArrayDeque<>();
        int[] ans = new int[n - k + 1];
        for (int i = 0; i < n; i++) {
            int x = nums[i];
            while (!deque.isEmpty() && x >= nums[deque.getLast()]) {
                deque.removeLast();
            }
            deque.addLast(i);
            int left = i - k + 1;
            if (deque.getFirst()<left) {
                deque.removeFirst();
            }
            if (left>=0) {
                ans[left]=nums[deque.getFirst()];
            }
        }
        return ans;

    }
}
