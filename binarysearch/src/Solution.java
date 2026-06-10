import javax.xml.stream.FactoryConfigurationError;
import java.util.*;
import java.util.logging.Level;

public class Solution {
    public int[] searchRange(int[] nums, int target) {
        /*
        * 34. 在排序数组中查找元素的第一个和最后一个位置
        * 给你一个按照非递减顺序排列的整数数组 nums，和一个目标值 target。请你找出给定目标值在数组中的开始位置和结束位置。
          如果数组中不存在目标值 target，返回 [-1, -1]。
        * 思路：考虑在非递减数组中，找到大于等于 target 的第一个下标
        * 如果挨个比较，时间复杂度是O(n)，并没有用到有序数组这个性质
        * 因为是有序的，找到中间大小的数与target比较，一次可以排除一半的元素，类似的分治下去。
        * 该问题就可以转化为求第一个大于等于 target 的位置，以及最后一个小于 target+1 的位置
        * */
        int start = binarySearch(nums, target);
        if (start == nums.length || nums[start] != target) return new int[]{-1, -1};
        int end = binarySearch(nums, target + 1) - 1;
        return new int[]{start, end};
    }

    public int maximumCount(int[] nums) {
        /*
         * 2529. 正整数和负整数的最大计数
         * 给你一个按 非递减顺序 排列的数组 nums ，返回正整数数目和负整数数目中的最大值。
         * 注意：0 既不是正整数也不是负整数。
         * 思路：题目可以转化为：找到第一个大于零的下标，找到第一个小于零的下标
         * */
        int len = nums.length;
        int neg = binarySearch(nums, 0) - 1; // 表示第一个小于 0 的下标
        neg = neg + 1; //再加一表示负数的个数
        int pos = binarySearch(nums, 1);
        pos = len - pos;
        return Math.max(neg, pos);
    }

    public int[] successfulPairs(int[] spells, int[] potions, long success) {
        /*
         * 2300. 咒语和药水的成功对数
         * 给你两个正整数数组 spells 和 potions ，长度分别为 n 和 m ，
         * spells[i] 表示第 i 个咒语的能量强度，potions[j] 表示第 j 瓶药水的能量强度。
         * 同时给你一个整数 success 。一个咒语和药水的能量强度 相乘 如果 大于等于 success ，那么它们视为一对 成功 的组合。
         * 请你返回一个长度为 n 的整数数组 pairs，其中 pairs[i] 是能跟第 i 个咒语成功组合的 药水 数目。
         * 思路：求得是数目，所以药水的顺序与答案无关，可以将药水数组进行排序，再用二分查找 target = success/spells[i]，就可以求出数量了
         * */
        int n = spells.length, m = potions.length;
        int[] ans = new int[n];
        // 药水排序
        Arrays.sort(potions);

        for (int i = 0; i < ans.length; i++) {
            long target = (success + spells[i] - 1) / spells[i]; // 向上取整
            ans[i] = m - binarySearch(potions, target);
        }
        return ans;

    }

    private int binarySearch(int[] nums, long target) {
        int left = -1;
        int right = nums.length;
        while (left + 1 < right) {
            // 开区间
            int mid = left + (right - left) / 2;
            int x = nums[mid];
            if (x < target) left = mid;
            else right = mid;
        }
        return right;
    }

    public int findTheDistanceValue(int[] arr1, int[] arr2, int d) {
        /*
         * 1385. 两个数组间的距离值
         * 给你两个整数数组 arr1 ， arr2 和一个整数 d ，请你返回两个数组之间的 距离值 。
         * 「距离值」 定义为符合此距离要求的元素数目：对于元素 arr1[i] ，不存在任何元素 arr2[j] 满足 |arr1[i]-arr2[j]| <= d
         * 思路：由于是求数目，所以可以对 arr2 排序，arr1中的这个元素和arr2中任何一个元素相减再求绝对值后都是大于d的，则答案加1
         * 相当于 arr2 中的所有元素都满足小于 arr1[i] -d 或 大于arr[i] + d；进一步相当于找到一个元素在arr1[i] -d 与 arr[i] + d
         * 之间，则该元素不符合条件；进一步找到第一个大于等于arr1[i] -d的数，如果该数都大于arr[i] + d则符合条件
         * */
        int len = arr2.length;
        Arrays.sort(arr2);
        int ans = 0;
        for (int x : arr1) {
            int target = x - d;
            int index = binarySearch(arr2, target);
            if (index == len || arr2[index] > x + d) ans++;
        }
        return ans;
    }

    public long countFairPairs(int[] nums, int lower, int upper) {
        /*
        * 2563. 统计公平数对的数目
        * 给你一个下标从 0 开始、长度为 n 的整数数组 nums ，和两个整数 lower 和 upper ，返回 公平数对的数目 。
        如果 (i, j) 数对满足以下情况，则认为它是一个 公平数对 ：
        0 <= i < j < n，且
        lower <= nums[i] + nums[j] <= upper
        * 思路：排序后，并不影响计数，枚举 nums[i] 计算满足 lower-nums[i] <=nums[j] <= upper-nums[i] 的个数，j>i
        * */
        long ans = 0;
        Arrays.sort(nums);
        int len = nums.length;
        for (int i = 0; i < len - 1; i++) {
            int x = nums[i];
            int index1 = binarySearch(nums, i, len, lower - x);
            int index2 = binarySearch(nums, i, len, upper - x + 1) - 1;
            ans += index2 - index1 + 1;
        }
        return ans;
    }

    public int minEatingSpeed(int[] piles, int h) {
        /*
         * 875. 爱吃香蕉的珂珂
         * 这里有 n 堆香蕉，第 i 堆中有 piles[i] 根香蕉。警卫已经离开了，将在 h 小时后回来。
         * 珂珂可以决定她吃香蕉的速度 k （单位：根/小时）。
         * 每个小时，她将会选择一堆香蕉，从中吃掉 k 根。如果这堆香蕉少于 k 根，她将吃掉这堆的所有香蕉，然后这一小时内不会再吃更多的香蕉。
         * 它必须在警卫回来前吃掉所有香蕉，返回她可以在 h 小时内吃掉所有香蕉的最小速度 k（k 为整数）。
         * 思路：利用二分查找搜索答案空间，找到数组中最大值和最小值，则答案在这个闭区间
         * 在这个闭区间上利用二分搜索答案
         * */
        int max = piles[0];
        for (int pile : piles) {

            if (pile > max) max = pile;
        }
        int left = 0, right = max + 1; // 开区间写法
        while (left + 1 < right) {
            int mid = (left + right) >>> 1;
            if (canEat(piles, mid, h)) right = mid;
            else left = mid;
        }
        return right;

    }

    private boolean canEat(int[] piles, int speed, int h) {
        // 判断在 speed，h 时间内，能否吃完；
        int time = 0;
        for (int pile : piles) {
            time = time + (pile + speed - 1) / speed;
            if (time > h) return false;
        }
        return true;
    }

    public long minimumTime(int[] time, int totalTrips) {
        /*
         * 2187. 完成旅途的最少时间
         * 给你一个数组 time ，其中 time[i] 表示第 i 辆公交车完成 一趟旅途 所需要花费的时间
         * 每辆公交车可以 连续 完成多趟旅途，也就是说，一辆公交车当前旅途完成后，可以 立马开始 下一趟旅途。
         * 每辆公交车 独立 运行，也就是说可以同时有多辆公交车在运行且互不影响
         * 给你一个整数 totalTrips ，表示所有公交车 总共 需要完成的旅途数目。请你返回完成 至少 totalTrips 趟旅途需要花费的 最少 时间。
         * 思路：假定所有旅途均由 time 中最大的公交车来完成，则一共需要 totalTrips * max(time) 的记为 right 的时间，
         * 则最终最少时间在 left = 1 与right中间产生，且如果 mid 时间能完成，则 大于mid 也能完成；mid时间不能完成，小于 mid 的也不能完成
         * 由于这个性质，可以用二分查找搜索答案
         * */
        int max = time[0];
        for (int i : time) {
            if (i > max) max = i;
        }
        long left = 0, right = (long) max * totalTrips + 1;
        while (left + 1 < right) {
            // 开区间写法
            long mid = (left + right) >>> 1;
            if (canFish(mid, time, totalTrips)) right = mid;
            else left = mid;
        }
        return right;
    }

    private boolean canFish(long hour, int[] time, int totalTrips) {
        long trips = 0;
        for (int i : time) {
            trips = trips + hour / i;
            if (trips >= totalTrips)
                return true;
        }
        return false;
    }

    public int hIndex(int[] citations) {
        /*
         * 275. H 指数 II
         * 给你一个整数数组 citations ，其中 citations[i] 表示研究者的第 i 篇论文被引用的次数，citations 已经按照 非降序排列 。
         * 计算并返回该研究者的 h 指数。
         * h 指数的定义：h 代表“高引用次数”（high citations），
         * 一名科研人员的 h 指数是指他（她）的 （n 篇论文中）至少 有 h 篇论文分别被引用了至少 h 次。
         * 思路：对于 x 如果 x 满足条件，则小于 x 也满足条件；x 不满足条件，则大于 x 也不满足条件
         * 利用这个性质，二分搜索答案，现在就是要确定答案所在的区间，左端点表示有可能的最小答案，citations[0]，由于是升序数组
         * 右端点最大的值
         * */
        int len = citations.length;
        int left = 0, right = len + 1;
        while (left + 1 < right) {
            int mid = (left + right) >>> 1;
            if (citations[len - mid] >= mid) left = mid;
            else right = mid;
        }
        return left;
    }

    public int maxNumberOfAlloys(int n, int k, int budget, List<List<Integer>> composition, List<Integer> stock, List<Integer> cost) {
        /*
         * 2861. 最大合金数
         * 假设你是一家合金制造公司的老板，你的公司使用多种金属来制造合金。
         * 现在共有 n 种不同类型的金属可以使用，并且你可以使用 k 台机器来制造合金。
         * 每台机器都需要特定数量的每种金属来创建合金。
         * 对于第 i 台机器而言，创建合金需要 composition[i][j] 份 j 类型金属。
         * 最初，你拥有 stock[x] 份 x 类型金属，而每购入一份 x 类型金属需要花费 cost[x] 的金钱。
         * 给你整数 n、k、budget，
         * 下标从 1 开始的二维数组 composition，两个下标从 1 开始的数组 stock 和 cost，
         * 请你在预算不超过 budget 金钱的前提下，最大化公司制造合金的数量。
         * 所有合金都需要由同一台机器制造。返回公司可以制造的最大合金数。
         * 思路：由于能制造 x 份合金，则小于 x 也一定能制作，如果不能制作 x 份合金，则大于 x 也一定不能
         * 利用这个性质，可以用二分查找之红蓝染色法来解决，接下来就需要确定答案区间：
         * 左端点可以简单确定为0，即最少一个也不能制作，
         * 右端点则可以假设cost全为1的情况下能制作的最多合金
         * */
        int mx = stock.getFirst();
        for (Integer i : stock) {
            if (i > mx) mx = i;
        }
        long left = 0;
        long right = mx + budget + 1;
        while (left + 1 < right) {
            long mid = (left + right) >>> 1;
            if (check(mid, n, k, budget, composition, stock, cost)) left = mid;
            else right = mid;
        }
        return (int) left;

    }

    private boolean check(long num, int n, int k, int budget, List<List<Integer>> composition, List<Integer> stock, List<Integer> cost) {
        // 验证在题目所给条件下能否制作 num 个合金
        for (int i = 0; i < k; i++) {
            List<Integer> integers = composition.get(i); // 表示用第 i 台机器制作一个合金需要的各种金属的数量
            long cost_i = 0;
            for (int j = 0; j < n; j++) {
                long nums = integers.get(j) * num - stock.get(j); // 表述需要购买的金属数量
                if (nums > 0) cost_i = cost_i + nums * cost.get(j);// 表示需要的购买合金数量
                if (cost_i > budget) {
                    // 表示该机器不能完成制作
                    break;
                }
            }
            if (cost_i <= budget) return true;
        }
        return false;
    }

    public int minimizeArrayValue(int[] nums) {
        /*
        * 2439. 最小化数组中的最大值
        * 给你一个下标从 0 开始的数组 nums ，它含有 n 个非负整数。
        * 每一步操作中，你需要：
        选择一个满足 1 <= i < n 的整数 i ，且 nums[i] > 0 。
        将 nums[i] 减 1 。
        将 nums[i - 1] 加 1 。
        你可以对数组执行 任意 次上述操作，请你返回可以得到的 nums 数组中 最大值 最小 为多少。
        * 思路：如果拿到一个数 x 判断该数是不是 nums 中的最大值最小的情况，好判断；同时如果一个数 x 满足条件，则小于 x 越满足
        * 如果 x 不满足条件，则大于 x 越不满足，所以用二分猜答案
        * */
        int mx = nums[0];
        for (int num : nums) {
            if (num > mx) mx = num;
        }
        int left = nums[0] - 1, right = mx + 1;
        while (left + 1 < right) {
            int mid = (left + right) >>> 1;
            if (check(nums, mid)) right = mid;
            else left = mid;
        }
        return right;
    }

    private boolean check(int[] nums, int target) {
        long extra = 0;
        for (int i = nums.length - 1; i > 0; i--) {
            long newNum = nums[i] + extra;
            extra = Math.max(0, newNum - target);
        }
        return nums[0] + extra <= target;
    }

    public int maximumTastiness(int[] price, int k) {
        /*
         * 2517. 礼盒的最大甜蜜度
         * 给你一个正整数数组 price ，其中 price[i] 表示第 i 类糖果的价格，另给你一个正整数 k 。
         * 商店组合 k 类 不同 糖果打包成礼盒出售。礼盒的 甜蜜度 是礼盒中任意两种糖果 价格 绝对差的最小值。
         * 返回礼盒的 最大 甜蜜度。
         * 思路：最小值中的最大值，经典的用二分查找找答案，
         * 假定 x 是甜蜜度，要求礼盒中选出 k 中糖果，其中选出的任意两种糖果价格差大于 x，由于 x 越小越满足条件，x 越大越不满足条件
         * 同时给定一个甜蜜度 x，然我们判断是否满足条件，
         * 也就是是否存在从 price 中选择 k 元素，其中任意两个元素差是否都大于 x且必须存在两个元素差等于x
         * 二分猜答案，左端点 0：所有糖果价格一样；右端点 max(price)-min(price)
         * 这里有一个问题，如果这样做，答案区间不是连续的，即有可能搜索到 3 时，但是不存在两糖果价格之间差等于3
         * 所以需要构造一个连续的答案搜索区间，怎么构造？
         * 将数组排序，排序后相邻两种糖果价格绝对差构成答案区间，绝对值差的最小值，如果答案由不是在相邻糖果差之间构成，则不满足最小值这个条件，因为相当于 x 变大了
         * */
        Arrays.sort(price);
        int len = price.length;
        int[] diff = new int[len - 1];
        for (int i = 0; i < diff.length; i++) {
            diff[i] = Math.abs(price[i] - price[i + 1]);
        }
        Arrays.sort(diff);
        int left = -1, right = len - 1;
        while (left + 1 < right) {
            int mid = (left + right) >>> 1;
            if (k == 2 || check(mid, diff, k)) left = mid;
            else right = mid;
        }
        return diff[left];


    }

    private boolean check(int mid, int[] diff, int k) {
        // 判断diff中是否存在 k-1 个子数组且这k-1个子数组不能中的元素不能重复和满足大于等于 diff[mid]
        int sum = 0;
        int target = diff[mid];
        for (int i = 0; i < diff.length; i++) {
            sum = sum + diff[i];
            if (sum >= target) {
                k--;
                sum = 0;
            }
            if (k == 1) return true;
        }
        return false;
    }

    public int findPeakElement(int[] nums) {
        /*
         * 162.寻找峰值
         * 峰值元素是指其值严格大于左右相邻值的元素。
         * 给你一个整数数组 nums，找到峰值元素并返回其索引
         * 数组可能包含多个峰值，在这种情况下，返回 任何一个峰值 所在位置即可。
         * 你必须实现时间复杂度为 O(log n) 的算法来解决此问题。
         * 注意：对于所有有效的 i 都有 nums[i] != nums[i + 1]
         * 思路：根据提示：主要要判断 nums[i] 与 nums[i+1]的大小
         * 对于一个数 nums[i]，如果它小于 nums[i+1] 则它一定在峰值的左侧 红色
         * 如果它大于 nums[i+1] 则它一定为峰值或者在峰值的右侧 蓝色
         * 利用这个“单调性”可以对数组进行二分查找
         * 数组最后一个数一定在峰值右侧或为右侧即为蓝色
         * 这种二分查找不同的是不是在有序数组上进行的，那是怎么想到用二分的呢？
         * 首先二分的本质，是利用条件、性质，一次排除一半的搜索空间
         * 这道题能够二分有以下条件：峰值一定存在，而由于 nums[i] != nums[i + 1]，所以要么大、要么小，
         * 如果大则它可能是峰值也有可能不是，而如果小则它一定不是峰值，
         * 由于这种大小关系一定会传递下去，而峰值又是一定存在的，所以能让我们一次排除一半的搜索空间
         * */
        int left = -1, right = nums.length - 1;
        while (left + 1 < right) {
            int mid = (left + right) >>> 1;
            if (nums[mid] > nums[mid + 1]) {
                right = mid;
            } else left = mid;
        }
        return right;
    }

    public int findMin(int[] nums) {
        /*
         * 153. 寻找旋转排序数组中的最小值
         * 已知一个长度为 n 的数组，预先按照升序排列，经由 1 到 n 次 旋转 后，得到输入数组。
         * 例如，原数组 nums = [0,1,2,4,5,6,7] 在变化后可能得到：
         * 若旋转 4 次，则可以得到 [4,5,6,7,0,1,2]
         * 就是两段递增数组
         * 请你找出并返回数组中的 最小元素 。你必须设计一个时间复杂度为 O(log n) 的算法解决此问题。
         * 思路：类似162，这个数组的最后一个元素要么在最小元素的右边，要么是最小元素
         * nums[mid] 与 最后一个元素比较大小，如果大于最后一个元素，说明在最小值的左边，红色
         * 如果小于最后一个元素说明在最小元素右边，蓝色
         * 这道题为什么能一次排除一半？首先最小值一定存在，且元素不相等，说明可以判断大小，
         * 但是和谁判断，需要找到一个标准，最后一个元素往往很特殊，他要么在最小值右侧，要么为最小值
         * */
        int len = nums.length;
        int left = -1, right = len - 1;
        while (left + 1 < right) {
            int mid = (left + right) >>> 1;
            if (nums[mid] < nums[len - 1]) right = mid;
            else left = mid;
        }
        return right;
    }


    private int binarySearch(int[] nums, int left, int right, int target) {
        while (left + 1 < right) {
            int mid = (left + right) >>> 1;
            if (nums[mid] < target) left = mid;
            else right = mid;
        }
        return right;
    }

    public int search(int[] nums, int target) {
        /*
         * 33. 搜索旋转排序数组
         * 整数数组 nums 按升序排列，数组中的值 互不相同
         * nums 在预先未知的某个下标 k（0 <= k < nums.length）上进行了 向左旋转，
         * 使数组变为 [nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]]（下标 从 0 开始 计数）。
         * 例如， [0,1,2,4,5,6,7] 下标 3 上向左旋转后可能变为 [4,5,6,7,0,1,2] 。
         * 给你 旋转后 的数组 nums 和一个整数 target ，如果 nums 中存在这个目标值 target ，则返回它的下标，否则返回 -1 。
         * 你必须设计一个时间复杂度为 O(log n) 的算法解决此问题。
         * 思路：用153找到最小值，然后在判断target可能在哪段数组中，在该段有序数组中进行二分
         * */
//        int min = findMin(nums);
//        int len = nums.length;
//        if (target == nums[len - 1]) {
//            return len - 1;
//        } else if (target > nums[len - 1]) {
//            int index = binarySearch(nums, -1, min, target);
//            return nums[index] == target ? index : -1;
//        } else {
//            int index = binarySearch(nums, -1, min, target);
//            return nums[index] == target ? index : -1;
//        }
        /*
         * 思路：一次二分，类似的，能否找到一个标准，将 nums[mid]与其比较一次就能排除一半的搜索空间，
         * nums[mid]之与target比由于数组不是单调的
         * 所以不能排除，所以还需要一个标准，同样的也是最后一个元素，他要么在target的右侧，要么就是target
         * 怎么判断左右？我们只需要关注一个状态，另一个状态就是else
         * */
        int len = nums.length;
        int left = -1, right = len - 1;
        int flag = nums[len - 1];
        while (left + 1 < right) {
            int mid = (left + right) >>> 1;
            if (nums[mid] > flag && flag > target) left = mid;
            else if (target > flag && flag > nums[mid]) right = mid;
            else if (nums[mid] >= target) right = mid;
            else left = mid;
        }
        return nums[right] == target ? right : -1;

    }


    public boolean searchMatrix(int[][] matrix, int target) {
        /*
         * 74. 搜索二维矩阵
         * 给你一个满足下述两条属性的 m x n 整数矩阵：
         * 每行中的整数从左到右按非严格递增顺序排列。
         * 每行的第一个整数大于前一行的最后一个整数。
         * 给你一个整数 target ，如果 target 在矩阵中，返回 true ；否则，返回 false 。
         * 思路：每行最后一个元素x与 target比较，如果target大于x，则说明它在后面的行，直接排除当前行
         * 如果target小于x说明target可能在当前行
         * */
        int m = matrix.length, n = matrix[0].length;
        for (int i = 0; i < m; i++) {
            int x = matrix[i][n - 1];
            if (target == x) return true;
            else if (target > x) continue;
            else {
                int index = binarySearch(matrix[i], -1, n - 1, target);
                return matrix[i][index] == target;
            }
        }
        return false;
    }

    public int[] findPeakGrid(int[][] mat) {
        /*
         * 1901. 寻找峰值 II
         * 一个 2D 网格中的 峰值 是指那些 严格大于 其相邻格子(上、下、左、右)的元素。
         * 给你一个 从 0 开始编号 的 m x n 矩阵 mat ，其中任意两个相邻格子的值都不相同 。找出任意一个峰值 mat[i][j] 并 返回其位置 [i,j] 。
         * 你可以假设整个矩阵周边环绕着一圈值为 -1 的格子。
         * 要求必须写出时间复杂度为 O(m log(n)) 或 O(n log(m)) 的算法
         * 思路：类比一维寻找峰值，峰值一定存在（由于周围围绕一圈-1）
         * 由于时间复杂度为O(n log(m))相当于在行或列上做二分查找
         * 由于峰值的性质，则它在他所在的列与行都是峰值；
         * 先在一行中找到峰值，然后需要在确认该列是否也是峰值，如果不是则在下一行中继续找
         * 这个思路有问题，因为找到的是该行的一个峰值，所以有可能出现全局的峰值出现在该行的另一个峰值中，而找出的当前行的峰值并不是全局的峰值
         * 思路2：利用大小的传递性，找到第一行的峰值，将它与下一行同列的元素比较，如果它大，则他就是峰值，如果它小，
         * 则向下传递，然后在左右判断该值是否在这行也是峰值，如果是，则继续向下判断，如果不是重新在这行找这行的峰值
         * 还是错误的，因为你找的这一行的峰值有可能不是全局的峰值，从而遗漏
         * 思路3：总和1、2，都是通过一维的找峰值可能不是全局的峰值，这会导致漏解
         * 思路找到中间行的最大值，比较该值与下一行同列元素的大小，
         * 如果小于，则下面的行中一定有峰值（大于的传递性），而这种传递不会穿过中间行向上，因为是从中间行的最大值传递下去的
         * 如果大于，则说明峰值在中间行或中间行的上面
         * */
        int m = mat.length, n = mat[0].length;
        int left = -1, right = m - 1;
        while (left + 1 < right) {
            int mid = (left + right) >>> 1; // 中间行号
            int j = findMax(mat[mid]);
            int x = mat[mid][j];
            if (x > mat[mid + 1][j]) right = mid;
            else left = mid;
        }
        return new int[]{right, findMax(mat[right])};


    }

    private int findMax(int[] ints) {
        int ans = 0;
        for (int i = 0; i < ints.length - 1; i++) {
            if (ints[i] > ints[i + 1])
                ans = i;
        }
        return ans;
    }

    public int findMin2(int[] nums) {
        /*
         * 154. 寻找旋转排序数组中的最小值 II
         * 在153寻找旋转排序数组中的最小值基础上，添加了数组中的元素有可能重复这一个条件
         * 思路：原来的思路是将中间元素与最后一个数作比较，如果比它小，最小在mid左边，比它大则在右边；
         * 不一样的地方在与mid可能与target相等，这是最小有可能在其左边，也有可能在其右边，这里需要考虑在左边的全部情况，然后else就是右边
         * */
        int len = nums.length;
        int left = -1, right = len - 1;
        while (left + 1 < right) {
            int mid = (left + right) >>> 1;
            if (nums[mid] < nums[right]) right = mid;
            else if (nums[mid] >nums[right]) left = mid;
            else right--;
        }
        return right;
    }

    private int binarySearch(int[] nums, int target) {
        int left = -1;
        int right = nums.length;
        while (left + 1 < right) {
            // 开区间
            int mid = left + (right - left) / 2;
            int x = nums[mid];
            if (x < target) left = mid;
            else right = mid;
        }
        return right;
    }
}
