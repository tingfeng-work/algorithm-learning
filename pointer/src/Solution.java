import javax.management.remote.JMXServiceURL;
import javax.xml.stream.FactoryConfigurationError;
import java.util.*;

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

    public int maxArea(int[] height) {
        /*
         * 11.盛水最多的容器
         * 长度为 n 的数组 height，表示第 i 个位置的高度，求任意两个高度构成的最大面积
         * 思路：考虑任意两个线段构成的面积，对于短线段不动，他们中间的任意线段不会与短线段构成更大的面积
         * 所以每次移动的都必然是短线段，然后一头一尾把所有线段包含进来，遍历后得到的就是答案
         * */
        int len = height.length;
        int left = 0;
        int right = len - 1;
        int ans = 0;
        while (left < right) {
            int area = (right - left) * Math.min(height[left], height[right]);
            ans = Math.max(ans, area);
            if (height[left] > height[right]) {
                right--;
            } else
                left++;
        }
        return ans;
    }

    public int trap(int[] height) {
        /*
         * 42.接雨水
         * 数组表示每个宽度为1的圆柱的高度，计算以此排列的柱子，能装多少水
         * 思路：对每个桶能装多少水，取决于短边有多高，而这个短边是当前柱体左边或者右边的最大高度
         * 所以用两个数组，分别正向遍历与倒序遍历，存储第 i 个位置的前i个圆柱的最大高度与从i个位置以后的圆柱的最大高度
         * 最后装的水由当前圆柱高度以及当前圆柱左边最大高度与右边最大高度决定
         * */
//        int ans = 0;
//        int len = height.length;
//        int[] pre_max = new int[len];
//        int[] suf_max = new int[len];
//
//        // 初始化前缀高度
//        pre_max[0] = height[0];
//        for (int i = 1; i < pre_max.length; i++) {
//            pre_max[i] = Math.max(pre_max[i - 1], height[i]);
//        }
//
//        //初始化后缀高度
//        suf_max[len - 1] = height[len - 1];
//        for (int i = suf_max.length - 2; i >= 0; i--) {
//            suf_max[i] = Math.max(height[i], suf_max[i + 1]);
//        }
//
//        // 计算答案
//        for (int i = 0; i < height.length; i++) {
//            ans = ans + Math.min(pre_max[i], suf_max[i]) - height[i];
//        }
//
//        return ans;
        // 空间上优化，在初始化前缀高度与后缀高度时，相当于对一个数组进行相向遍历，联系相向指针
        // 移动指针所指高度小的指针，因为圆柱装水由小高度决定
        int left = 0;
        int right = height.length - 1;
        int ans = 0;
        int pre = height[left];
        int suf = height[right];
        while (left < right) {
            if (pre < suf) {
                ans = ans + pre - height[left];
                left++;
                pre = Math.max(pre, height[left]);
            } else {
                ans = ans + suf - height[right];
                right--;
                suf = Math.max(suf, height[right]);
            }
        }
        return ans;
    }

    public boolean isPalindrome(String s) {
        /*
         * 125.验证回文串
         * 不考虑大小写、空格等非字母数字字符，正反读都一样，则是回文串
         * */
//        String lowerCase = s.toLowerCase();
//        char[] charArray = lowerCase.toCharArray();
//
//        List<Character> list = new ArrayList<>();
//        for (int i = 0; i < charArray.length; i++) {
//            if (isLegal(charArray[i])) {
//                list.add(charArray[i]);
//            }
//        }
//        int len = list.size();
//        int right = len - 1;
//        int left = 0;
//        while (left <= right) {
//            if (list.get(left++) != list.get(right--))
//                return false;
//        }
//        return true;
        // 优化：库函数 Character.isLetterOrDigit
        String lowerCase = s.toLowerCase();
        char[] charArray = lowerCase.toCharArray();
        int left = 0;
        int right = charArray.length - 1;
        while (left < right) {
            if (!Character.isLetterOrDigit(charArray[left])) left++;
            else if (!Character.isLetterOrDigit(charArray[right])) right--;
            else if (charArray[left] == charArray[right]) {
                left++;
                right--;
            } else return false;
        }
        return true;
    }

//    private boolean isLegal(char c) {
//        if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9'))
//            return true;
//        return false;
//    }

    public int minimumRefill(int[] plants, int capacityA, int capacityB) {
        /*
         * 2105.给植物浇水
         * A 与 B 给植物浇水，A 从左往右，B 从右往左；他们给植物浇水需要的时间是相同的，
         * 必须满足容量大于等于植物需要的水才能进行浇水，如果 A B 到达同一株植物，水桶容量多的人浇水
         * 求给所有植物浇水需要补充几次水
         * 思路：这是一个天然的相向指针的问题，根据题意两个指针同时移动，指向同一位置时，当前水多的浇水
         * 条件：max(plants[i]) <= capacityA, capacityB <= 109
         * */
        int ans = 0;
        int alice = capacityA, bob = capacityB;// 当前水量
        int len = plants.length;
        int left = 0, right = len - 1;
        while (left < right) {
            if (alice >= plants[left]) {
                alice = alice - plants[left++];
            } else {
                if (alice != capacityA) {
                    ans++;
                    alice = capacityA - plants[left++];
                }
            }
            if (bob >= plants[right]) {
                bob = bob - plants[right--];
            } else {
                if (bob != capacityB) {
                    ans++;
                    bob = capacityB - plants[right--];
                }
            }
        }
        if (left == right) {
            // 浇水浇到同一位置
            if (alice >= bob) {
                if (alice < plants[left]) {
                    ans++;
                }
            } else {
                if (bob < plants[right]) ans++;
            }

        }

        return ans;
    }


    public int minSubArrayLen(int target, int[] nums) {
        /*
         * 209.长度最小的子数组
         * 给定一个含有 n 个正整数的数组和一个正整数 target，求数组中满足和大于等于 target 的最小子数组
         * 注意：子数组是有序且连续下标构成，不能跳过
         * 暴力做法：枚举每个元素，嵌套循环找到该元素为首的满足条件的最小子数组，由于都是正整数，往后添加元素不符合题意
         * 这里暴力枚举答案始终构成一个窗口，由于该数组都为正整数，所以往窗口中添加元素只会更满足条件，这就是单调性；
         * */
        int left = 0;
        int sum = 0;
        int len = nums.length;
        int ans = len + 1;
        for (int right = 0; right < len; right++) {
            sum = sum + nums[right];
            while (sum >= target) {
                // 此时窗口中的子数组满足条件
                ans = Math.min(ans, right - left + 1);
                sum = sum - nums[left];
                left++;
            }
        }
        return ans <= len ? ans : 0;
    }

    public int lengthOfLongestSubstring(String s) {
        /*
         * 3.无重复字符的最长字串
         * 给定一个字符串 s ，请你找出其中不含有重复字符的 最长 子串 的长度。
         * 可以用一个hash表来存储当前窗口中是否出现重复字符
         * 只有每次添加进来的字符会导致重复，所以每次添加进来后，窗口左端需要移动保证无重复字符
         * */
        char[] charArray = s.toCharArray();
        int len = charArray.length;
        int left = 0;
        int[] cnt = new int[128];
        int ans = 0;
        for (int right = 0; right < len; right++) {
            cnt[charArray[right]]++;
            while (cnt[charArray[right]] > 1) {
                // 窗口内有重复元素，左端移动
                cnt[charArray[left]]--;
                left++;
            }
            ans = Math.max(ans, right - left + 1);
        }
        return ans;
    }


    public int numSubarrayProductLessThanK(int[] nums, int k) {
        /*
         * 713. 乘积小于 K 的子数组
         * 给你一个整数数组 nums 和一个整数 k ，请你返回子数组内所有元素的乘积严格小于 k 的连续子数组的数目
         * */
        if (k <= 1) return 0;
        int len = nums.length;
        int left = 0;
        int ans = 0;
        int product = 1;
        for (int right = 0; right < len; right++) {
            product = product * nums[right];
            while (product >= k) {
                // 窗口不满足条件
                product = product / nums[left];
                left++;
            }
            ans = ans + right - left + 1;
        }
        return ans;
    }

    public int maxSubarrayLength(int[] nums, int k) {
        /*
         * 2958.最多k个重复元素的最长子数组
         * 类似无重复元素的最长字串
         * */
        int len = nums.length;
        int ans = 0;
        int left = 0;
        Map<Integer, Integer> map = new HashMap<>();
        for (int right = 0; right < len; right++) {
            map.merge(nums[right], 1, Integer::sum);
            while (map.get(nums[right]) > k) {
                // 窗口不满足条件
                map.compute(nums[left], (key, i) -> i - 1);
                left++;
            }
            ans = Math.max(ans, right - left + 1);
        }
        return ans;

    }

    public int longestSemiRepetitiveSubstring(String s) {
        /*
         * 2730.最长的半重复子字符串
         * 如果一个字符串 t 中至多有一对相邻字符是相等的，那么称这个字符串 t 是 半重复的
         * 给你一个下标从 0 开始的字符串 s ，这个字符串只包含 0 到 9 的数字字符。返回 s 中最长的半重复字串长度
         * 每次向右扩展时，判断添加当前元素后，窗口内是否满足条件
         * */
        char[] charArray = s.toCharArray();
        int len = charArray.length;
        if (len <= 2) return len;
        int left = 0;
        int ans = 0;
        boolean flag = false; // 表示窗口内现在已经出现一对相邻字符相等了；
        for (int right = 1; right < len; right++) {
            if (charArray[right] == charArray[right - 1]) {
                // 表示当前添加的字符构成半重复
                while (flag) {
                    // 此时窗口内不满足题意，左端点需要移动
                    if (charArray[left] == charArray[left + 1]) {
                        flag = false;
                    }
                    left++;
                }
                flag = true;
            }
            ans = Math.max(ans, right - left + 1);
        }
        return ans;
    }

    public int longestOnes(int[] nums, int k) {
        /*
         * 1004.最大连续 1 的个数Ⅲ
         * 给定一个二进制数组 nums 和一个整数 k，假设最多可以翻转 k 个 0 ，则返回执行操作后 数组中连续 1 的最大个数 。
         * 思路：滑动窗口：不断向右枚举，当枚举到的答案是 0 时，发生翻转；翻转后如果次数为负，左端点向右枚举，元素退出窗口，直到当前元素为 0
         * */
        int len = nums.length;
        int ans = 0;
        int left = 0;
        for (int right = 0; right < len; right++) {
            if (nums[right] == 0) {
                k--;
            }
            while (k < 0) {
                // 此时窗口包含的元素不符合题意
                if (nums[left] == 0)
                    k++;
                left++;
            }
            ans = Math.max(right - left + 1, ans);
        }

        return ans;

    }

    public long countSubarrays(int[] nums, int k) {
        /*
         * 2962. 统计最大元素出现至少 K 次的子数组
         * 统计数组中最大元素至少出现 k 次的子数组数目
         * 首先，这道题要使用滑动窗口的话，需要提前遍历数组，直到最大元素是多少
         * 还有一个问题是满足窗口条件时怎么统计数目可以做到不重不漏
         * 向右枚举，当加入当前窗口元素为最大元素时，k--；
         * 当 k = 0 时，表示当前窗口构成的子数组恰好符合题意，那么当前数组继续向右枚举依然符合题意，则符合题意的数组有 len-right+1 个
         * 同时 k = 0 时，移动左指针，如果当前移动的元素不为最大元素，则窗口内子数组依然符合题意，更新答案
         * */
        int len = nums.length;
        int max = 0;
        long ans = 0;
        for (int i = 0; i < len; i++) {
            if (nums[i] > max)
                max = nums[i];
        }

        int left = 0;
        for (int right = 0; right < len; right++) {
            if (nums[right] == max) {
                k--;
            }
            while (k == 0) {
                // 此时窗口内元素构成的子数组满足题意，要更新答案
                ans = ans + len - right;
                if (nums[left] == max) k++;
                left++;
            }
        }
        return ans;

    }

    public long countSubarrays(int[] nums, long k) {
        /*
         * 2302. 统计得分小于 K 的子数组数目
         * 一个数组的得分为数组之和乘以数组长度
         * 给你一个正整数数组 nums 和一个整数 k ，请你返回 nums 中分数 严格小于 k 的 非空整数子数组数目。
         * 由于元素越多，窗口越不满足条件，符合单调性，可以使用滑动窗口
         * */
        int len = nums.length;
        long ans = 0;
        int left = 0, score = 0, sum = 0;
        for (int right = 0; right < len; right++) {
            sum = sum + nums[right];
            score = sum * (right - left + 1);
            while (score >= k) {
                // 当前窗口内包含的元素不符合题意
                sum = sum - nums[left++];
                score = sum * (right - left + 1);
            }
            ans = ans + right - left + 1;
        }
        return ans;
    }

    public int minOperations(int[] nums, int x) {
        /*
         * 1658. 将 x 减到 0 的最小操作数
         * 每次操作只能删除数组最左或最右的元素，然后从 x 中减去操作的值，如果 x 恰好为 0，则返回操作的最小操作次数，否则返回 -1
         * 思路：其实就是变相求一定条件下的子数组满足子数组内的和为 数组总和-x，原数组的长度减去当前数组的长度就是操作数，求这个子数组的最长长度
         * 由于数组元素为正，元素越多和越大，故可以考虑滑动窗口，而滑动窗口向右枚举与左指针移动天然符合只能删除数组最左或最有的元素
         * */
        int len = nums.length;
        int sum = 0;
        for (int num : nums) {
            sum = sum + num;
        }
        if (sum < x) return -1;
        int target = sum - x; //数组元素和为target的最长长度
        int ans = -1, left = 0;
        sum = 0;
        for (int right = 0; right < len; right++) {
            sum = sum + nums[right];
            while (sum > target) {
                sum = sum - nums[left++];
            }
            if (sum == target) ans = Math.max(ans, right - left + 1);
        }
        return ans == -1 ? ans : len - ans;

    }

    public int minLength(int[] nums, int k) {
        /*
         * 3795. 不同元素和至少为 K 的最短子数组长度
         * 求子数组中出现的不同的值之和至少为 k 的最小长度
         * 思路：每次向右拓展元素时，如果当前值第一次出现，则把它加在和中
         * 窗口左端点移动时，如果当前值不再出现，则从和中删除
         * */
        int len = nums.length;
        int ans = len + 1, left = 0, sum = 0;
        Map<Integer, Integer> cnt = new HashMap<>();
        for (int right = 0; right < len; right++) {
            int x = nums[right]; // 当前操作的值
            Integer count = cnt.merge(x, 1, Integer::sum);
            if (count == 1) {
                // 第一次出现
                sum = sum + x;
            }
            while (sum >= k) {
                ans = Math.min(ans, right - left + 1);
                x = nums[left];
                count = cnt.merge(x, -1, Integer::sum);
                if (count == 0) {
                    // 删除后窗口内没有该元素
                    sum = sum - x;
                }
                left++;
            }
        }
        return ans == len + 1 ? -1 : ans;


    }

    public String minWindow(String s, String t) {
        /*
         * 76. 最小覆盖子串
         * 思路：先统计 t 串中字符的出现次数
         * 向右拓展窗口，每次拓展后如果当前元素包含在 t 串中，则计数减一
         * 在全包含后，即满足题目要求时，窗口向左拓展，并记录答案，这个过程是求最短的长度的子串
         * */
        char[] S = s.toCharArray();
        char[] T = t.toCharArray();
        int len1 = S.length, len2 = T.length;
        if (len2 > len1) return "";
        HashMap<Character, Integer> map = new HashMap<>();
        for (char c : T) {
            map.merge(c, 1, Integer::sum);
        }
        int left = 0, ansLen = len1 + 1, ansLeft = -1, missing = len2;
        for (int right = 0; right < len1; right++) {
            char c = S[right]; // 当前操作的字符
            if (map.containsKey(c)) {
                if (map.get(c) > 0) missing--;
                map.put(c, map.get(c) - 1);
            }
            // 当前字符已经完全包含了
            while (missing == 0) {
                // 窗口内元素符合题意
                int curLen = right - left + 1;
                if (curLen < ansLen) {
                    ansLen = curLen;
                    ansLeft = left;
                }
                c = S[left++];
                if (map.containsKey(c)) {
                    if (map.get(c) == 0) missing++;
                    map.put(c, map.get(c) + 1);
                }
            }

        }
        return ansLen==len1+1?"":s.substring(ansLeft,ansLeft+ansLen);
    }

}
