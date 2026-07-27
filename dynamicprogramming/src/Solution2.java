import java.util.*;
import java.util.function.IntToLongFunction;


public class Solution2 {

//    public int rob(int[] nums) {
    /*
     * 198. 打家劫舍
     * 给定一个代表每个房屋存放金额的非负整数数组，计算你不触动警报装置的情况下 ，一夜之内能够偷窃到的最高金额。
     * 如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警。
     * 思路：从回溯到动态规划：
     * 回溯三问：倒着枚举当前操作：枚举第 i 个房子选不选
     * 子问题：前 i 个房子的最高金额
     * 下一个子问题：如果当前操作是选：则前 i-2 个房子的最高金额
     * 如果当前操作是不选，则前 i-1 个房子的最高金额
     * */
//        int n = nums.length;
//        int[] cache = new int[n];
//        Arrays.fill(cache, -1);
//        return dfs(n - 1, cache, nums);
    /*
     * 翻译为递推：省略往下递的过程，直接从低向上归，把递归改为循环
     * */
//        int n = nums.length;
//        int[] dp = new int[n + 2];
//        for (int i = 0; i < n; i++) {
//            dp[i + 2] = Math.max(dp[i + 1], dp[i] + nums[i]);
//        }
//        return dp[n + 1];
    // 优化空间复杂度，由于整个过程只涉及三个状态，当前状态、上一个状态、上上一个状态
//        int f0 = 0, f1 = 0;
//        for (int num : nums) {
//            int newF = Math.max(f0 + num, f1);
//            f0 = f1;
//            f1 = newF;
//        }
//        return f1;
//    }

//    private int dfs(int i, int[] cache, int[] nums) {
//        if (i < 0) return 0;
//        if (cache[i] != -1) return cache[i];
//        int res = Math.max(dfs(i - 1, cache, nums), dfs(i - 2, cache, nums) + nums[i]);
//        cache[i] = res;
//        return res;
//    }

    public int climbStairs(int n) {
        /*
         * 70. 爬楼梯
         * 假设你正在爬楼梯。需要 n 阶你才能到达楼顶。
         * 每次你可以爬 1 或 2 个台阶。你有多少种不同的方法可以爬到楼顶呢？
         * 思路：动态规划，需要 3 阶则 3 阶可以由 1 阶跨 2 阶来，也可以由 2 阶跨 1 阶来
         * */
//        int[] dp = new int[n + 2];
//        dp[0] = 0;
//        dp[1] = 1;
//        for (int i = 0; i < n; i++) {
//            dp[i + 2] = dp[i + 1] + dp[i];
//        }
//        return dp[n + 1];
        // 空间优化，实际只涉及三个状态
        int f0 = 0, f1 = 1;
        for (int i = 0; i < n; i++) {
            int newF = f0 + f1;
            f0 = f1;
            f1 = newF;
        }
        return f1;
    }

    public int minCostClimbingStairs(int[] cost) {
        /*
         * 746. 使用最小花费爬楼梯
         * 给你一个整数数组 cost ，其中 cost[i] 是从楼梯第 i 个台阶向上爬需要支付的费用。
         * 一旦你支付此费用，即可选择向上爬一个或者两个台阶。
         * 你可以选择从下标为 0 或下标为 1 的台阶开始爬楼梯。
         * 思路：动态规划：第 n 个台阶可以从第 n-1 个台阶来，也可以从第 n-2 个台阶来，以此类推
         * 最开始的第一个台阶与第 0 个台阶的消耗为 0
         * */
        int n = cost.length; // 要爬到第 n 个台阶
        int f0 = 0, f1 = 0;
        for (int i = 2; i <= n; i++) {
            int newF = Math.min(f0 + cost[i - 2], f1 + cost[i - 1]);
            f0 = f1;
            f1 = newF;
        }
        return f1;

    }

    public int climbStairs(int n, int[] costs) {
        /*
         * 3693. 爬楼梯 II
         * 你正在爬一个有 n + 1 级台阶的楼梯，台阶编号从 0 到 n。
         * 你还得到了一个长度为 n 的 下标从 1 开始 的整数数组 costs，其中 costs[i] 是第 i 级台阶的成本。
         * 从第 i 级台阶，你只能跳到第 i + 1、i + 2 或 i + 3 级台阶。
         * 从第 i 级台阶跳到第 j 级台阶的成本定义为： costs[j] + (j - i)2
         * 思路动态规划：当前状态 dp[n] 表示到达第 n 级台阶所需的最小总成本，
         * 该状态可以由 dp[n-1], dp[n-2], dp[n-3] 并加上各自的需要的成本得到
         * */
//        int[] dp = new int[n + 3];
//        for (int i = 0; i < n; i++) {
//            dp[i + 3] = Math.min(Math.min(dp[i] + 9, dp[i + 1] + 4), dp[i + 2] + 1) + costs[i];
//        }
//        return dp[n + 2];
        int f0 = 0, f1 = 0, f2 = 0;
        for (int i = 0; i < n; i++) {
            int newF = Math.min(Math.min(f0 + 9, f1 + 4), f2 + 1) + costs[i];
            f0 = f1;
            f1 = f2;
            f2 = newF;
        }
        return f2;

    }

    public int rob(int[] nums) {
        /*
         * 213. 打家劫舍 II
         * 给定一个代表每个房屋存放金额的非负整数数组，计算你在不触动警报装置的情况下，今晚能够偷窃到的最高金额。
         * 这个地方所有的房屋都围成一圈 ，这意味着第一个房屋和最后一个房屋是紧挨着的。
         * 同时，相邻的房屋装有相互连通的防盗系统，如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警。
         * 讨论是否偷 nums[0]：
         * 如果偷，则 nums[1] 与 nums[n-1] 不能偷，问题变为了 2~n-2 的非环形的版本
         * 如果不偷，则问题变味了 1~n-1 的版本
         * */
        int n = nums.length;
        return Math.max(nums[0] + rob2(nums, 2, n - 2), rob2(nums, 1, n - 1));
    }

    private int rob2(int[] nums, int start, int end) {
        int f0 = 0, f1 = 0;
        for (int i = start; i <= end; i++) {
            int newF = Math.max(f0 + nums[i], f1);
            f0 = f1;
            f1 = newF;
        }
        return f1;
    }

    public int deleteAndEarn(int[] nums) {
        /*
         * 740. 删除并获得点数
         * 给你一个整数数组 nums ，你可以对它进行一些操作。
         * 每次操作中，选择任意一个 nums[i] ，
         * 删除它并获得 nums[i] 的点数。之后，你必须删除 所有等于 nums[i] - 1 和 nums[i] + 1 的元素。
         * 开始你拥有 0 个点数。返回你能通过这些操作获得的最大点数。
         * 思路：将数组排序，删除所有等于 nums[i]-1 和 nums[i]+1的元素就可等价为删除 nums[i] - 1的元素即可（倒着枚举）
         * 回溯：当前操作枚举第 i 个元素选不选
         * 子问题：从前 i 个元素中获得点数
         * 下一个子问题：
         * 分类讨论：如果i-1个元素的值等于 nums[i] 或 等于 nums[i] -1,则需要删除则从前 i-2 个元素获取点数
         * */
//        int n = nums.length;
//        Arrays.sort(nums);
//        int[] cache = new int[n];
//        Arrays.fill(cache, -1);
//        return dfs(n - 1, nums, cache);
        /*
         * 思路：构建一个值域数组 a[i] 表示 nums 中元素值为 i 的所有和，则原问题转化为了打家劫舍问题
         *
         * */
        int mx = 0;
        for (int num : nums) {
            if (num > mx)
                mx = num;
        }
        int[] a = new int[mx + 1];
        for (int num : nums) {
            a[num] += num;
        }
        int f0 = 0, f1 = 0;
        for (int i = 0; i < a.length; i++) {
            int newF = Math.max(f0 + a[i], f1);
            f0 = f1;
            f1 = newF;
        }
        return f1;
    }

    //    private int dfs(int i, int[] nums, int[] cache) {
//        if (i < 0) return 0;
//        if (cache[i] != -1) return cache[i];
//        // 不选
//        int score1 = dfs(i - 1, nums, cache);
//        // 选
//        int val = nums[i];
//        int score2 = val;
//        int j = i - 1;
//        for (; j >= 0; j--) {
//            if (nums[j] == val) {
//                score2 = score2 + val;
//                continue;
//            } else if (nums[j] == val - 1) {
//                continue;
//            } else {
//                break;
//            }
//        }
//        score2 = score2 + dfs(j, nums, cache);
//        return cache[i] = Math.max(score2, score1);
//
//    }
    private static final int MOD = 1_000_000_007;

    public int countGoodStrings(int low, int high, int zero, int one) {
        /*
         * 2466. 统计构造好字符串的方案数
         * 给你整数 zero ，one ，low 和 high ，我们从空字符串开始构造一个字符串，每一步执行下面操作中的一种：
         * 将 '0' 在字符串末尾添加 zero  次。
         * 将 '1' 在字符串末尾添加 one 次。
         * 如果通过以上过程得到一个 长度 在 low 和 high 之间（包含上下边界）的字符串，那么这个字符串我们称为好字符串。
         * 请你返回满足以上要求的 不同 好字符串数目。由于答案可能很大，请将结果对 109 + 7 取余 后返回。
         * 思路：爬楼梯的变形：相当于每次可以爬 zero 或 one 阶台阶，求爬到 low-high 台阶的方案数总和
         * dfs(i) 表示爬到第 i 阶台阶的方案数
         * 当前操作，选 0 或 选 1
         * 子问题：前 i 阶台阶的方案数
         * 下一个子问题：分类讨论：选 0 ：爬到第 i-zero 阶的方案数
         * 选 1：爬到第 i-one 阶的方案数
         * */
//        int ans = 0;
//        int[] cache = new int[high + 1];
//        Arrays.fill(cache, -1);
//        for (int i = low; i <= high; i++) {
//            ans = (ans + dfs(i, zero, one, cache)) % MOD;
//        }
//        return ans;
        /*
         * 递推：f[i] 表示构造长为 i 字符串的方案数，如果 i >= zero，则状态可以来自 f[i-zero]；
         * 同理，如果 i >= one，则状态可以来自 f[i-one]
         * */
        int[] f = new int[high + 1];
        f[0] = 1;
        final int MOD = 1_000_000_007;
        for (int i = 1; i < high + 1; i++) {
            if (i >= zero) f[i] = (f[i] + f[i - zero]) % MOD;
            if (i >= one) f[i] = (f[i] + f[i - one]) % MOD;
        }
        int ans = 0;

        for (int i = low; i <= high; i++) {
            ans = (ans + f[i]) % MOD;
        }
        return ans;

    }

    //    private int dfs(int i, int zero, int one, int[] cache) {
//        if (i < 0) {
//            return 0;
//        }
//        if (i == 0) {
//            return 1;
//        }
//        if (cache[i] != -1)
//            return cache[i];
//        return cache[i] = (dfs(i - zero, zero, one, cache) + dfs(i - one, zero, one, cache))%MOD;
//    }
    public int combinationSum4(int[] nums, int target) {
        /*
         * 377. 组合总和 Ⅳ
         * 给你一个由 不同 整数组成的数组 nums ，和一个目标整数 target 。
         * 请你从 nums 中找出并返回总和为 target 的元素组合的个数。
         * 请注意，顺序不同的序列被视作不同的组合
         * 由于顺序不同的序列是不同的组合，如果枚举数组元素选不选
         * */
//        int n = nums.length;
//        int[][] cache = new int[n][target + 1];
//        for (int[] ints : cache) {
//            Arrays.fill(ints, -1);
//        }
//        return dfs(n - 1, nums, target, cache);
        /*
         * 思路：枚举当前元素选哪个
         * dfs(i) 表示前 i 个元素构成和为 target-sum 的组合数
         * 子问题：从 nums 中选出和为 target 的组合个数
         * 下一个子问题：从 nums 中选出和为 target - sum 的组合个数
         * */
//        int n = nums.length;
//        int[] cache = new int[target + 1];
//
//        Arrays.fill(cache, -1);
//
//        return dfs( nums, target, cache);
        /*
         * 翻译为递推
         * */
        // f[i] 表示和为 i 的组合个数
        int[] f = new int[target + 1];
        f[0] = 1;
        for (int i = 1; i <= target; i++) {
            for (int num : nums) {
                if (i >= num)
                    f[i] = f[i] + f[i - num];
            }
        }
        return f[target];
    }

//    private int dfs(int[] nums, int target, int[] cache) {
//        if (target <= 0) return target == 0 ? 1 : 0;
//        if (cache[target] != -1) return cache[target];
//        int res = 0;
//        for (int j = 0; j < nums.length; j++) {
//            if (target >= nums[j])
//                res = res + dfs(nums, target - nums[j], cache);
//        }
//        return cache[target] = res;
//
//    }

    // dfs(i) 表示：前i个元素总和为 target 的组合个数
//    private int dfs(int i, int[] nums, int target, int[][] cache) {
//        if (i < 0) return 0;
//        if (target == 0) return 1;
//        if (cache[i][target] != -1) return cache[i][target];
//        // 选
//        int val = nums[i];
//        int res = 0;
//        if (target >= val) {
//            res = dfs(nums.length - 1, nums, target - val, cache); // 由于顺序不同的序列是不同的组合，所以选了要从头选
//        }
//        // 不选
//        res = res + dfs(i - 1, nums, target, cache);
//        return cache[i][target] = res;
//    }

//    public int countTexts(String pressedKeys) {
    /*
     * 2266. 统计打字方案数
     * 22233：8
     * 思路：也是爬楼梯的变形，将字符串分组：例如 22233 分为 222 33
     * 阶梯长度也就是 target = 分组字符串长度，例如 2 和 3
     * 每次可以爬的阶梯数是 除了 '7' 和 '9' 对应 4，其他都是 3
     * 最后对的答案就是每个分组的方案数相乘
     * 所以构造 targets 数组，存储每个分组的字符串长度
     * nums 数组中 nums[i] 表示 target[i] 一次最多可以爬的阶梯数量，值只能是 3 或 4
     * */
//        char[] chars = pressedKeys.toCharArray();
//        int n = chars.length;
//        int count = 1;
//        char pre = chars[0];
//        List<AbstractMap.SimpleEntry<Integer, Integer>> list = new ArrayList<>();
//        for (int i = 1; i < chars.length; i++) {
//            char c = chars[i];
//            if (pre == c) count++;
//            else {
//                if (pre == '7' || pre == '9') {
//                    list.add(new AbstractMap.SimpleEntry<>(count, 4));
//                } else
//                    list.add(new AbstractMap.SimpleEntry<>(count, 3));
//                count = 1;
//                pre = c;
//            }
//        }
//        list.add(new AbstractMap.SimpleEntry<>(count, (chars[n - 1] == '7' || chars[n - 1] == '9') ? 4 : 3));
//        long ans = 1;
//        final int MOD = 1_000_000_007;
//        for (AbstractMap.SimpleEntry<Integer, Integer> pair : list) {
//            int target = pair.getKey();
//            int num = pair.getValue();
//            int[] f = new int[target + 1];
//            f[0] = 1;
//            for (int j = 0; j <= target; j++) {
//                for (int k = 1; k <= num; k++) {
//                    if (j >= k) {
//                        f[j] = (f[j] + f[j - k]) % MOD;
//                    }
//                }
//            }
//            ans = ans * f[target] % MOD;
//        }
//        return (int) ans;


//    }

    //    private static final int MOD = 1_000_000_007;
    private static final int MX = 100_001;
    private static final long[] f = new long[MX];
    //    private static final long[] g = new long[MX];
    private static boolean done = false;

    private void init() {
        if (done) return;
        done = true;
        f[0] = g[0] = 1;
        f[1] = g[1] = 1;
        f[2] = g[2] = 2;
        f[3] = g[3] = 4;
        for (int i = 4; i < MX; i++) {
            f[i] = (f[i - 1] + f[i - 2] + f[i - 3]) % MOD;
            g[i] = (g[i - 1] + g[i - 2] + g[i - 3] + g[i - 4]) % MOD;
        }
    }

    public int countTexts(String pressedKeys) {
        /*
         * 上述代码，每有一个分组就会创建进行一次动态规划的递推，而递推的过程是相同
         * 优化思路：利用静态代码块创建一个状态数组，全程只进行一次递推，分组后直接从状态数组拿答案
         * */
        init();
        char[] chars = pressedKeys.toCharArray();
        int n = chars.length;
        int count = 0;
        long ans = 1;
        for (int i = 0; i < n; i++) {
            char c = chars[i];
            count++;
            if (i == n - 1 || c != chars[i + 1]) {
                ans = (ans * ((c == '7' || c == '9') ? g[count] : f[count])) % MOD;
                count = 0;
            }
        }
        return (int) ans;
    }

    public int minPathSum(int[][] grid) {
        /*
         * 思路：f[i][j] 表示从 grid[0][0] 到 grid[i][j] 路径的最小总和
         * 该状态由f[i-1][j] 和 f[i][j-1] 的较小值加上 grid[i][j] 得来
         * */
        int n = grid.length;
        int m = grid[0].length;
        int[][] f = new int[n + 1][m + 1];
        Arrays.fill(f[0], Integer.MAX_VALUE);
        for (int i = 0; i <= n; i++) {
            f[i][0] = Integer.MAX_VALUE;
        }
        f[0][1] = 0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                f[i + 1][j + 1] = Math.min(f[i][j + 1], f[i + 1][j]) + grid[i][j];
            }
        }
        return f[n][m];
    }

    public int findTargetSumWays(int[] nums, int target) {
        /*
         * 494. 目标和
         * 给你一个非负整数数组 nums 和一个整数 target 。
         * 向数组中的每个整数前添加 '+' 或 '-' ，然后串联起所有整数，可以构造一个 表达式 ：
         * 例如，nums = [2, 1] ，可以在 2 之前添加 '+' ，在 1 之前添加 '-' ，然后串联起来得到表达式 "+2-1" 。
         * 返回可以通过上述方法构造的、运算结果等于 target 的不同 表达式 的数目。
         * 这道题怎么与动态规划联系起来？可以看成每个数选择 + 或者不选，不选则是 - 号
         * 同时还可以转换题目要求，设正数和为 p
         * 则负数的和为 sum-p
         * 根据题目要求 p - (sum-p) = target,
         * 即： p = (t+s)/2
         * 所以就是从 nums 中选择和为 p 的方案数目，p 满足(t+s)/2
         * */
        for (int num : nums) {
            target = target + num;
        }
        if (target < 0 || target % 2 == 1) return 0;
        target = target / 2;
        int[] f = new int[target + 1];
        f[0] = 1;
        for (int num : nums) {
            for (int j = target; j >= num; j--) {
                f[j] = f[j] + f[j - num];
            }
        }
        return f[target];
//        target = target / 2;
//        int[][] cache = new int[n][target + 1];
//        for (int[] ints : cache) {
//            Arrays.fill(ints, -1);
//        }
//        return dfs(n - 1, nums, target, cache);
    }

    // dfs(i) 表示前i个数和为target的方案数
    // 当前操作：枚举第 i 个数选不选
    // 子问题：前i个数和为 target 的方案数
    // 下一个子问题：选：前 i - 1 个数和为 target - nums[i] 的方案数
    // 不选：前 i-1 个数和为 target 的方案数
//    private int dfs(int i, int[] nums, int target, int[][] cache) {
//        if (i < 0) return target == 0 ? 1 : 0;
//        if (cache[i][target] != -1) return cache[i][target];
//        int res = 0;
//        if (nums[i] <= target) {
//            // 选
//            res = dfs(i - 1, nums, target - nums[i], cache);
//        }
//        res = res + dfs(i - 1, nums, target, cache);
//        return cache[i][target] = res;
//    }


    public int coinChange(int[] coins, int amount) {
        /*
         * 322. 零钱兑换
         * 给你一个整数数组 coins ，表示不同面额的硬币；以及一个整数 amount ，表示总金额。
         * 计算并返回可以凑成总金额所需的 最少的硬币个数 。如果没有任何一种硬币组合能组成总金额，返回 -1 。
         * 你可以认为每种硬币的数量是无限的。
         * 思路：完全背包，选了还能选
         * 从 coins 中选出和为 amount 的元素，coins 中每个元素选了还能再选
         * 求最小价值数，每个coins的价值为1
         * f[i][j] 表示前 i 个元素和为 j 的最少硬币数量
         * */
        int[] f = new int[amount + 1];
        Arrays.fill(f, Integer.MAX_VALUE / 2);
        f[0] = 0;
        for (int coin : coins) {
            for (int j = coin; j <= amount; j++) {
                f[j] = Math.min(f[j], f[j - coin] + 1);

            }
        }
        return f[amount] < Integer.MAX_VALUE / 2 ? f[amount] : -1;
    }

    public int lengthOfLongestSubsequence(List<Integer> nums, int target) {
        /*
         * 2915. 和为目标值的最长子序列的长度
         * 给你一个下标从 0 开始的整数数组 nums 和一个整数 target 。
         * 返回和为 target 的 nums 子序列中，子序列 长度的最大值 。如果不存在和为 target 的子序列，返回 -1 。
         * 子序列 指的是从原数组中删除一些或者不删除任何元素后，剩余元素保持原来的顺序构成的数组。
         * 0-1背包问题，容量是 target，体积是 nums[i]，价值是1 求最大价值
         * */
        int[] f = new int[target + 1];
        Arrays.fill(f, Integer.MIN_VALUE);
        f[0] = 0;
        int s = 0;
        for (Integer num : nums) {
            s = Math.min(s + num, target);
            for (int i = s; i >= num; i--) {
                f[i] = Math.max(f[i], f[i - num] + 1);
            }
        }
        int ans = f[target];
        return ans < 0 ? -1 : ans;
    }

    public boolean canPartition(int[] nums) {
        /*
         * 416. 分割等和子集
         * 给你一个只包含正整数的非空数组 nums。请你判断是否可以将这个数组分割成两个子集，使得两个子集的元素和相等。
         * 设选和为 p 的数作为子集1，则需要 sum-p = p，
         * 即2p = sum，p = sum/2
         * 即从nums中选择出和为 p 的子集，满足 p = sum/2
         * */
        int sum = 0;
        for (int num : nums) {
            sum = num + sum;
        }
        if (sum % 2 == 1) return false;
        int target = sum / 2;
        boolean[] f = new boolean[target + 1]; // f[i][j] 表示前i个数中是否存在和恰好为 j 的方案
        f[0] = true;
        int s2 = 0;
        for (int num : nums) {
            s2 = Math.min(s2 + num, target);
            for (int j = s2; j >= num; j--) {
                f[j] = f[j - num] || f[j];

            }
        }
        return f[target];
    }

    private static final int MOD = 1_000_000_007;

    public int numberOfWays(int n, int x) {
        /*
         * 2787. 将一个数字表示成幂的和的方案数
         * 给你两个 正 整数 n 和 x 。
         * 请你返回将 n 表示成一些 互不相同 正整数的 x 次幂之和的方案数。
         * 换句话说，你需要返回互不相同整数 [n1, n2, ..., nk] 的集合数目，满足 n = n1^x + n2^x + ... + nk^x 。
         * 由于答案可能非常大，请你将它对 109 + 7 取余后返回。
         * 比方说，n = 160 且 x = 3 ，一个表示 n 的方法是 n = 2^3 + 3^3 + 5^3 。
         * 容量相当于 n，每个物品的体积相当于 n1^x
         * 求恰巧装 n 的方案数
         * */
//        int nums = 1;
//        while (Math.pow(nums, x) <= n) {
//            nums++;
//        }
//        int[][] f = new int[nums+1][n + 1]; // f[i][j] 表示前 i 个数中和为 j
//        f[0][0] = 1;
//        f[1][0] = 1;
//        for (int i = 1; i < nums; i++) {
//            for (int j = 0; j <= n; j++) {
//                if (Math.pow(i, x) > j)
//                    f[i + 1][j] = f[i][j];
//                else {
//                    f[i + 1][j] = (f[i][j] + f[i][(int) (j - Math.pow(i, x))])%MOD;
//                }
//            }
//        }
//        return f[nums][n];
        long[] f = new long[n + 1];
        f[0] = 1;
        for (int i = 1; Math.pow(i, x) <= n; i++) {
            int v = (int) Math.pow(i, x);
            for (int j = n; j >= v; j--) {
                f[j] = f[j] + f[j - v];
            }
        }
        return (int) (f[n] % 1_000_000_007);
//        return dfs(nums - 1, n, x);
    }

    //    private int dfs(int i, int n, int x) {
//        if (i <= 0) return n == 0 ? 1 : 0;
//        int res = 0;
//        if (Math.pow(i, x) <= n) {
//            // 选
//            res = dfs(i - 1, (int) (n - Math.pow(i, x)), x);
//        }
//        res = res + dfs(i - 1, n, x);
//        return res;
//    }
    public int change(int amount, int[] coins) {
        /*
        * 518. 零钱兑换Ⅱ
        * 给你一个整数数组 coins 表示不同面额的硬币，另给一个整数 amount 表示总金额。
        请你计算并返回可以凑成总金额的硬币组合数。如果任何硬币组合都无法凑出总金额，返回 0 。
        假设每一种面额的硬币有无限个。
        题目数据 保证 最终 结果符合 32 位 带符号整数。
        * 方案数，完全背包，选了可以再选
        * */

        int[] f = new int[amount + 1];
        f[0] = 1;
        for (int coin : coins) {
            for (int j = coin; j <= amount; j++) {

                f[j] = f[j] + f[j - coin];

            }
        }
        return f[amount];
    }

    public int numSquares(int n) {
        /*
        * 279. 完全平方数
        * 给你一个整数 n ，返回 和为 n 的完全平方数的最少数量 。
        完全平方数 是一个整数，其值等于另一个整数的平方；换句话说，
        * 其值等于一个整数自乘的积。例如，1、4、9 和 16 都是完全平方数，而 3 和 11 不是。
        * 完全背包，容量为 n，体积为 1^2,2^2,3^2... ，每个物体的价值为1
        * 求恰好为装满 n 时，物体的最小价值
        * */
        int[] f = new int[n + 1];
        Arrays.fill(f, Integer.MAX_VALUE / 2);
        f[0] = 0;
        for (int i = 1; Math.pow(i, 2) <= n; i++) {
            int v = (int) Math.pow(i, 2);
            for (int j = v; j <= n; j++) {
                f[j] = Math.min(f[j], f[j - v] + 1);
            }
        }
        return f[n];
    }

    public int longestCommonSubsequence(String text1, String text2) {
        /*
         * 1143. 最长公共子序列
         * 给定两个字符串 text1 和 text2，返回这两个字符串的最长公共子序列的长度。如果不存在公共子序列，返回 0。
         * 思路：设两个字符串为 s、t，则对于每个字符串中的字符 s[i] 与 t[j] 都可以选作子序列也可以不选
         * 当前操作：枚举 s[i] 与 t[j] 选不选
         * 子问题：前i个字符与前j个字符构成的最长公共子序列
         * 下一个子问题：前i-1与前j-1个字符构成的最长公共子序列
         * 前 i 与前 j-1 个字符构成的最长公共子序列
         * 前 i-1 与 前 j 个字符构成的最长公共子序列
         * */
//        int n = text1.length();
//        int m = text2.length();
//        int[][] memo = new int[n][m];
//        for (int[] ints : memo) {
//            Arrays.fill(ints, -1);
//        }
//        return dfs(n - 1, m - 1, text1.toCharArray(), text2.toCharArray(), memo);
//        char[] s = text1.toCharArray();
//        char[] t = text2.toCharArray();
//        int n = s.length, m = t.length;
//        int[][] f = new int[2][m+1];
//        for (int i = 0; i < n; i++) {
//            for (int j = 0; j < m; j++) {
//                if (s[i] == t[j]) {
//                    f[(i + 1) % 2][j + 1] = f[i % 2][j] + 1;
//                } else {
//                    f[(i + 1) % 2][j + 1] = Math.max(f[i % 2][j + 1], f[(i + 1) % 2][j ]);
//                }
//            }
//        }
//        return f[n % 2][m];
        char[] s = text1.toCharArray();
        char[] t = text2.toCharArray();
        int n = s.length, m = t.length;
        int[] f = new int[m + 1];
        for (char c : s) {
            int pre = f[0];
            for (int j = 0; j < m; j++) {
                int temp = f[j + 1];
                if (c == t[j]) {
                    f[j + 1] = pre + 1;
                } else {
                    f[j + 1] = Math.max(f[j + 1], f[j]);
                }
                pre = temp;
            }
        }
        return f[m];

    }

//    private int dfs(int i, int j, char[] s, char[] t, int[][] memo) {
//        if (i < 0) return 0;
//        if (j < 0) return 0;
//        if (memo[i][j] != -1) return memo[i][j];
//        if (s[i] == t[j])
//            return memo[i][j] = dfs(i - 1, j - 1, s, t, memo) + 1; // 这里字符相等的情况下，只选一个的情况永远小于两个都选的情况
//        else
//            return memo[i][j] = Math.max(dfs(i - 1, j, s, t, memo), dfs(i, j - 1, s, t, memo)); // 字符不相等的情况下，两个都不选的情况包含在了之只选一个的情况
//    }

    public int minDistance(String word1, String word2) {
        /*
         * 72. 编辑距离
         * 给你两个单词 word1 和 word2， 请返回将 word1 转换成 word2 所使用的最少操作数  。
         * 你可以对一个单词进行如下三种操作：
         * 插入一个字符
         * 删除一个字符
         * 替换一个字符
         * 思路：两个字符串 s,t，枚举 s[i],t[j] 选不选：f[i][j] 表示前i个字符转换成前j个字符需要的操作数
         * 如果两个字符相等，则是选，状态由 f[i-1][j-1] 转移来
         * 如果两个字符不相等，可以插入一个字符，插入的字符一定等于 t[j]，此时需要前i个与前j-1个字符相等需要的操作数
         * 状态由 f[i][j-1] 转移来
         * 删除一个字符，状态由 f[i-1][j] 转移来，表示删除s[i]
         * 替换一个字符状态由 f[i-1][j-1] 转移来，对称的，插入的字符等于s[i]与删除t[j]
         * */
        char[] s = word1.toCharArray();
        char[] t = word2.toCharArray();
        int n = s.length, m = t.length;
        int[] f = new int[m + 1];
        for (int i = 0; i < m; i++) {
            f[i + 1] = i + 1;
        }
        for (int i = 0; i < n; i++) {
            f[0] = i + 1;
            int pre = f[0];
            for (int j = 0; j < m; j++) {
                int temp = f[j + 1];
                if (s[i] == t[j]) {
                    f[j + 1] = pre;
                } else {
                    f[j + 1] = Math.min(Math.min(f[j + 1], f[j]), pre) + 1;
                }
                pre = temp;
            }
        }
        return f[m];
    }

    public int minDistance(String word1, String word2) {
        /*
         * 583. 两个字符串的删除操作
         * 给定两个单词 word1 和 word2 ，返回使得 word1 和  word2 相同所需的最小步数。
         * 每步可以删除任意一个字符串中的一个字符。
         * */
        char[] s = word1.toCharArray();
        char[] t = word2.toCharArray();
        int n = s.length, m = t.length;
        int[] f = new int[m + 1];
        for (int i = 0; i < m; i++) {
            f[i + 1] = i + 1;
        }
        for (int i = 0; i < n; i++) {
            int pre = f[0];
            f[0] = i + 1;
            for (int j = 0; j < m; j++) {
                int temp = f[j + 1];
                if (s[i] == t[j]) {
                    f[j + 1] = pre;
                } else {
                    f[j + 1] = Math.min(f[j + 1], f[j]) + 1;
                }
                pre = temp;
            }
        }
        return f[m];
    }

    public int minimumDeleteSum(String s1, String s2) {
        /*
         * 712. 两个字符串的最小ASCII删除和
         * 给定两个字符串s1 和 s2，返回 使两个字符串相等所需删除字符的 ASCII 值的最小和 。
         * */
        char[] s = s1.toCharArray();
        char[] t = s2.toCharArray();
        int n = s.length, m = t.length;
        int[] f = new int[m + 1];
        int[] sum1 = new int[n], sum2 = new int[m];
        sum1[0] = s[0];
        for (int i = 1; i < n; i++) {
            sum1[i] = sum1[i - 1] + s[i];
        }
        sum2[0] = t[0];
        for (int i = 1; i < m; i++) {
            sum2[i] = sum2[i - 1] + t[i];
        }
        System.arraycopy(sum2, 0, f, 1, m);
        for (int i = 0; i < n; i++) {
            int pre = f[0];
            f[0] = sum1[i];
            for (int j = 0; j < m; j++) {
                int temp = f[j + 1];
                if (s[i] == t[j]) {
                    f[j + 1] = pre;
                } else {
                    f[j + 1] = Math.min(f[j + 1] + s[i], f[j] + t[j]);
                }
                pre = temp;
            }
        }
        return f[m];
    }

    public boolean isInterleave(String s1, String s2, String s3) {
        /*
         * 97. 交错字符串
         * 给定三个字符串 s1、s2、s3，请你帮忙验证 s3 是否是由 s1 和 s2 交错 组成的
         * 两个字符串 s 和 t 交错 的定义与过程如下，其中每个字符串都会被分割成若干非空子字符串
         * s = s1 + s2 + ... + sn
         * t = t1 + t2 + ... + tm
         * |n - m| <= 1 // 这个怎么实现?
         * 交错 是 s1 + t1 + s2 + t2 + s3 + t3 + ... 或者 t1 + s1 + t2 + s2 + t3 + s3 + ...
         * 思路：枚举 s3 第 n+m 个字符选 s[j] 还是 t[k]
         * 子问题：前 n+m 个字符是否能由 s 和 t 交错组成
         * 下一个子问题：前 i -n 个字符能否由 s 和 t 的子串组成
         * */
//        char[] s = s1.toCharArray();
//        char[] t = s2.toCharArray();
//        char[] target = s3.toCharArray();
//        int len = target.length, n = s.length, m = t.length;
//        if (len != n + m) return false;
//        int[][] cache = new int[n][m];
//        return dfs(n - 1, m - 1, s, t, target, cache);
        // 翻译为递推
        char[] s = s1.toCharArray();
        char[] t = s2.toCharArray();
        char[] target = s3.toCharArray();
        int len = target.length, n = s.length, m = t.length;
        if (len != n + m) return false;
        boolean[] f = new boolean[m + 1];
        f[0] = true;
        for (int j = 0; j < m; j++) {
            f[j + 1] = t[j] == target[j] && f[j];
        }
        for (int i = 0; i < n; i++) {
            f[0] = s[i] == target[i] && f[0];
            for (int j = 0; j < m; j++) {
                f[j + 1] = (s[i] == target[i + j + 1] && f[j + 1]) ||
                        (t[j] == target[i + j + 1] && f[j]);
            }
        }
        return f[m];
    }

//    private boolean dfs(int i, int j, char[] s, char[] t, char[] target, int[][] cache) {
//        int len = i + j + 1;
//        if (len < 0) return true;
//        if (i >= 0 && j >= 0 && cache[i][j] != 0) return cache[i][j] == 1;
//        char c = target[len];
//        boolean res = false;
//        if (i >= 0 && s[i] == c)
//            res = res || dfs(i - 1, j, s, t, target, cache);
//        if (j >= 0 && t[j] == c)
//            res = res || dfs(i, j - 1, s, t, target, cache);
//        if (i >= 0 && j >= 0)
//            cache[i][j] = res ? 1 : 2;
//        return res;
//    }

    public int maxDotProduct(int[] nums1, int[] nums2) {
        /*
         * 1458. 两个子序列的最大点积
         * 给你两个数组 nums1 和 nums2 。
         * 请你返回 nums1 和 nums2 中两个长度相同的非空子序列的最大点积
         * 思路：枚举第 i 个元素与第 j 个元素选不选
         * 子问题：前 i 个元素与前 j 个元素组成的非空子序列的最大点积
         * 下一个子问题：
         * 前 i 个元素与前 j-1 个元素组成的非空子序列的最大点积
         * 前 i-1 个元素与前 j 个元素组成的非空子序列的最大点积
         * 前 i-1 个元素与前 j-1 个元素组成的非空子序列的最大点积 + 当前点积 三者的最大值
         * */
//        int n = nums1.length, m = nums2.length;
//        int[][] cache = new int[n][m];
//        for (int[] ints : cache) {
//            Arrays.fill(ints, Integer.MAX_VALUE);
//        }
//        return dfs(n - 1, m - 1, nums1, nums2, cache);
        int n = nums1.length, m = nums2.length;
        int[] f = new int[m + 1];
        for (int j = 0; j < m; j++) {
            f[j + 1] = Integer.MIN_VALUE / 2;
        }
        for (int k : nums1) {
            int pre = f[0];
            f[0] = Integer.MIN_VALUE / 2;
            for (int j = 0; j < m; j++) {
                int temp = f[j + 1];
                int value = k * nums2[j];
                int res1 = Math.max(f[j + 1], f[j]);
                int res2 = Math.max(pre + value, value);
                f[j + 1] = Math.max(res2, res1);
                pre = temp;
            }
        }
        return f[m];
    }

    // 前i个元素与前j个元素组成的非空子序列的最大点积
//    private int dfs(int i, int j, int[] nums1, int[] nums2, int[][] cache) {
//        if (i < 0 && j < 0) return 0;
//        if (i < 0 || j < 0) return Integer.MIN_VALUE / 2;
//        if (cache[i][j] != Integer.MAX_VALUE) return cache[i][j];
//        int value = nums1[i] * nums2[j];
//
//        int res = Math.max(Math.max(Math.max(dfs(i - 1, j, nums1, nums2, cache), dfs(i, j - 1, nums1, nums2, cache)),
//                dfs(i - 1, j - 1, nums1, nums2, cache) + value), value);
//
//        return cache[i][j] = res;
//    }

    public String shortestCommonSupersequence(String str1, String str2) {
        /*
         * 1092. 最短公共超序列
         * 给你两个字符串 str1 和 str2，返回同时以 str1 和 str2 作为 子序列 的最短字符串。
         * 如果答案不止一个，则可以返回满足条件的 任意一个 答案。
         * 先求出最长公共子序列
         * s与t的长度对应m，n，假设答案的长度为len
         * 则 len = m+n-公共子序列
         * 要使 len 最短，m+n是固定的，就需要求最长公共子序列
         * 然后根据最长公共子序列来构造答案
         * 构造答案的过程：首先如果两个字符相同，直接加入答案
         * 如果不同，需要考虑选择哪个字符？如果选择 s[i]，则需要子问题s[:i-1]与t[:j]构造最短公共子序列，
         * 转换为s[:i-1]与t[:j]的最长公共子序列大于 s[:i]与t[:j-1]的长度，所以不选择t[i]
         * */
        char[] s = str1.toCharArray();
        char[] t = str2.toCharArray();
        int n = s.length, m = t.length;
        int[][] f = new int[n + 1][m + 1]; // f[i+1][j+1] 表示前i个元素与前j个元素构成的最长公共子序列长度
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                if (s[i] == t[j])
                    f[i + 1][j + 1] = f[i][j] + 1;
                else
                    f[i + 1][j + 1] = Math.max(f[i + 1][j], f[i][j + 1]);
            }
        }
        int len = n + m - f[n][m];
        char[] ans = new char[len];
        int i = n, j = m;
        while (i > 0 && j > 0) {
            if (s[i - 1] == t[j - 1]) {
                ans[len - 1] = s[i - 1];
                len--;
                i--;
                j--;
            } else if (f[i - 1][j] > f[i][j - 1]) {
                ans[len - 1] = s[i - 1];
                len--;
                i--;
            } else {
                ans[len - 1] = t[j - 1];
                len--;
                j--;
            }
        }
        if (len > 0) {
            while (i > 0) {
                ans[len - 1] = s[i - 1];
                i--;
                len--;
            }
            while (j > 0) {
                ans[len - 1] = t[j - 1];
                j--;
                len--;
            }
        }
        return new String(ans);

    }

    public int lengthOfLIS(int[] nums) {
        /*
         * 300. 最长递增子序列
         * 给你一个整数数组 nums ，找到其中最长严格递增子序列的长度。
         * 思路：子序列本质上也是子集
         * 采用枚举第 i 个答案选哪个
         * 子问题：前 i 个元素构成的最长严格递增子序列
         * 枚举前 i 个元素中小于 nums[i] 的 nums[j]
         * 下一个子问题：前j个元素构成的最长严格递增子序列
         * */
//        int n = nums.length;
//        int ans = 0;
//        int[] memo = new int[n];
//        Arrays.fill(memo, -1);
//        for (int i = 0; i < n; i++) {
//            ans = Math.max(ans, dfs(i, nums, memo));
//        }
//        return ans;
        // 递推
//        int n = nums.length;
//        int[] f = new int[n]; // f[i] 表示以 nums[i] 结尾的最长的LIS的长度
//        int ans = 0;
//        for (int i = 0; i < n; i++) {
//            for (int j = 0; j < i; j++) {
//                if (nums[j] < nums[i])
//                    f[i] = Math.max(f[j], f[i]);
//            }
//            f[i]++;
//            ans = Math.max(ans, f[i]);
//        }
//        return ans;
        // 二分+贪心：调换动态规划数组状态与状态值
        // g[i] 表示长度为 i+1 的 LIS 的最后一个元素的最小值，为什么是最小值？
        // 因为维护最小值，后面扩展 LIS 长度的可能性更大
        // 这里数组 g 是严格单调递增的，遍历 nums 的同时二分查找，当前 nums[i] 在g的为止
        int n = nums.length;
        List<Integer> g = new ArrayList<>(n);
        for (int num : nums) {
            int index = binarySearch(num, g);
            if (index == g.size()) {
                g.add(num);
            } else {
                g.set(index, num);
            }
        }
        return g.size();
    }

    private int binarySearch(int target, List<Integer> g) {
        int left = -1, right = g.size();
        while (left + 1 < right) {
            int mid = (left + right) >>> 1;
            if (g.get(mid) <= target) {
                left = mid;
            } else
                right = mid;
        }
        return right;
    }

    // dfs(i) 表示以 nums[i] 结尾的元素的最长递增子序列的长度
//    private int dfs(int i, int[] nums, int[] memo) {
//        if (i < 0) return 0;
//        if (memo[i] != -1) return memo[i];
//        int res = 0;
//        for (int j = 0; j < i; j++) {
//            if (nums[j] < nums[i])
//                res = Math.max(res, dfs(j, nums, memo));
//        }
//        return memo[i] = res + 1;
//    }

    public int minimumOperations(List<Integer> nums) {
        /*
         * 2826. 将三个组排序
         * 给你一个整数数组 nums 。nums 的每个元素是 1，2 或 3。
         * 在每次操作中，你可以删除 nums 中的一个元素。返回使 nums 成为 非递减顺序所需操作数的最小值。
         * 这道题就是求LIS的变形，求出 LIS的，再用原始的长度减去 LIS 的长度即可，
         * 严格来说这里求的是最长非递减自序列，也就是最长递增子序列的情况下，允许相同元素
         * */
//        int n = nums.size();
//        List<Integer> g = new ArrayList<>();
//        for (int num : nums) {
//            int index = binarySearch(num, g);
//            if (index == g.size())
//                g.add(num);
//            else g.set(index, num);
//        }
//        return n - g.size();
        /*
         * 由于值域很小，1，2，3：可以考虑二维 dp，将值域当作dp的一个维度
         * f[i+1][j] 表示前 i 个元素的最长非递减子序列，其中子序列最后一个元素 <= j
         * */
        int n = nums.size();
        int[] f = new int[4];
        for (int x : nums) {
            for (int j = 3; j >= x; j--) {
                f[j] = Math.max(f[j], f[x] + 1);
            }
        }
        return n - f[3];
    }

    public int[] longestObstacleCourseAtEachPosition(int[] obstacles) {
        /*
         *  找出到每个位置为止最长的有效障碍赛跑路线
         * 你打算构建一些障碍赛跑路线。给你一个下标从 0 开始的整数数组 obstacles
         * 数组长度为 n ，其中 obstacles[i] 表示第 i 个障碍的高度。
         * 对于每个介于 0 和 n - 1 之间（包含 0 和 n - 1）的下标  i ，
         * 在满足下述条件的前提下，请你找出 obstacles 能构成的最长障碍路线的长度：
         * 1. 你可以选择下标介于 0 到 i 之间（包含 0 和 i）的任意个障碍。
         * 2. 在这条路线中，必须包含第 i 个障碍。
         * 3. 你必须按障碍在 obstacles 中的 出现顺序 布置这些障碍
         * 4. 除第一个障碍外，路线中每个障碍的高度都必须和前一个障碍 相同 或者 更高 /
         * 这四个条件就是在说以 obstacles[i] 结尾的非递减子序列
         * 返回长度为 n 的答案数组 ans ，其中 ans[i] 是上面所述的下标 i 对应的最长障碍赛跑路线的长度。
         * */
//        int n = obstacles.length;
//        int[] f = new int[n];//f[i] 表示以 nums[i] 结尾的非递减子序列长度
//        for (int i = 0; i < n; i++) {
//            for (int j = 0; j < i; j++) {
//                if (obstacles[i] >= obstacles[j]) {
//                    f[i] = Math.max(f[i], f[j]);
//                }
//            }
//            f[i]++;
//        }
//        return f;
        int n = obstacles.length;
        int[] ans = new int[n];
        List<Integer> g = new ArrayList<>(); // g[i] 表示长度为 i+1 的LIS 的最后一个元素的最小值
        for (int i = 0; i < n; i++) {
            int num = obstacles[i];
            int index = binarySearch(num, g);
            if (g.size() == index) {
                g.add(num);
            } else {
                g.set(index, num);
            }
            ans[i] = index + 1;
        }
        return ans;
    }

    //    public int maxProfit(int[] prices) {
//        /*
//         * 122. 买卖股票的最佳时机 II
//         * 给你一个整数数组 prices ，其中 prices[i] 表示某支股票第 i 天的价格。
//         * 在每一天，你可以决定是否购买和/或出售股票。你在任何时候 最多 只能持有 一股 股票。
//         * 然而，你可以在 同一天 多次买卖该股票，但要确保你持有的股票不超过一股。
//         * 思路：对于第 i 天，有两个状态，持有股票或者未持有股票
//         * 当天持有股票的状态可以由前一天未持有的转移来，也可以由前一天持有的状态转移来
//         * 定义 f[i][0] 表示第 i 天未持有股票的状态下，前 i 天获取的最大利润
//         * */
//        int[] f = new int[2];
//        f[1] = Integer.MIN_VALUE;
//        for (int price : prices) {
//            int newF0 = Math.max(f[0], f[1] + price);
//            f[1] = Math.max(f[1], f[0] - price);
//            f[0] = newF0;
//        }
//        return f[0];
//    }
    public int maxProfit(int[] prices) {
        /*
        * 309. 买卖股票的最佳时机含冷冻期
        * 给定一个整数数组prices，其中第  prices[i] 表示第 i 天的股票价格
        * 设计一个算法计算出最大利润。在满足以下约束条件下，你可以尽可能地完成更多的交易（多次买卖一支股票）:
        * 卖出股票后，你无法在第二天买入股票 (即冷冻期为 1 天)。
        * 输入: prices = [1,2,3,0,2]
        输出: 3
        解释: 对应的交易状态为: [买入, 卖出, 冷冻期, 买入, 卖出]
        * 思路: 可以看到第 i 天买入的状态需要从第 i-2 天卖出的状态转移过来
        * 对应状态方程，定义 f[i][0] 表示第 i 天未持有股票的状态下，前 i 天的最大利润
        * f[i][1] 表示第 i 天持有股票的状态下，前 i 天的最大利润；
        * 对应f[i][i] 应该由 f[i-1][1] 或者 f[i-2][0] 转移来
        * */
//        int n = prices.length;
//        int[][] f = new int[n + 2][2];
//        f[1][1] = Integer.MIN_VALUE;
//        for (int i = 0; i < n; i++) {
//            f[i + 2][0] = Math.max(f[i + 1][0], f[i + 1][1] + prices[i]);
//            f[i + 2][1] = Math.max(f[i + 1][1], f[i][0] - prices[i]);
//        }
//        return f[n + 1][0];
        int[] f = new int[2];
        f[1] = Integer.MIN_VALUE;
        int pre = 0;
        for (int price : prices) {
            int newF0 = Math.max(f[0], f[1] + price);
            f[1] = Math.max(f[1], pre - price);
            pre = f[0];
            f[0] = newF0;
        }
        return f[0];

    }

    public int maxProfit(int k, int[] prices) {
        /*
        * 188. 买卖股票的最佳时机 IV
        * 给你一个整数数组 prices 和一个整数 k ，其中 prices[i] 是某支给定的股票在第 i 天的价格。
        设计一个算法来计算你所能获取的最大利润。你最多可以完成 k 笔交易。也就是说，你最多可以买 k 次，卖 k 次。
        * 思路：f[i][j][0] 表示第 i 天还剩下 j 次交易次数未持有股票的情况下，前 i 天的最大利润
        * f[i][j][0] 的状态由 f[i-1][j][0] 不操作 以及 f[i-1][j-1][1] + prices[i] 卖出前一天持有股票的状态转移而来
        * 由于买入卖出视为一次操作，所以只需在买入或卖出时，修改 j 即可
        * */
//        int n = prices.length;
//        int[][][] f = new int[n + 1][k + 2][2]; // 为什么是 k+2 因为在递归中 k 的值域是[-1,k]
//        // 难点初始化，这里有一个技巧，可以先将全部状态设置为非法的，然后设置正确的起点
//        for (int[][] ints : f) {
//            for (int[] anInt : ints) {
//                Arrays.fill(anInt, Integer.MIN_VALUE / 2);
//            }
//        }
//        // 合法的状态
//        for (int j = 1; j <= k + 1; j++) {
//            f[0][j][0] = 0;
//        }
//        for (int i = 0; i < n; i++) {
//            for (int j = 1; j <= k + 1; j++) {
//                f[i + 1][j][0] = Math.max(f[i][j][0], f[i][j - 1][1] + prices[i]);
//                f[i + 1][j][1] = Math.max(f[i][j][1], f[i][j][0] - prices[i]);
//            }
//        }
//        return f[n][k + 1][0];
        int n = prices.length;
        int[][] f = new int[k + 2][2]; // 为什么是 k+2 因为在递归中 k 的值域是[-1,k]
        // 难点初始化，这里有一个技巧，可以先将全部状态设置为非法的，然后设置正确的起点

        for (int[] anInt : f) {
            Arrays.fill(anInt, Integer.MIN_VALUE / 2);
        }

        // 合法的状态
        for (int j = 1; j <= k + 1; j++) {
            f[j][0] = 0;
        }
        for (int price : prices) {
            for (int j = 1; j <= k + 1; j++) {
                f[j][0] = Math.max(f[j][0], f[j - 1][1] + price);
                f[j][1] = Math.max(f[j][1], f[j][0] - price);
            }
        }
        return f[k + 1][0];

    }

    public int longestPalindromeSubseq(String s) {
        /*
         * 516. 最长回文子序列
         * 给你一个字符串 s ，找出其中最长的回文子序列，并返回该序列的长度。
         * 思路1：由于回文子序列正着读和反着读相同，
         * 可以先反转 s 为 t，再求 s 与 t 的最长公共子序列的长度即可
         * */
//        StringBuilder sb = new StringBuilder(s);
//        String t = sb.reverse().toString();
//        // f[i][j] 表示前 i 个字符与前 j 个字符的最长公共子序列
//        char[] sCharArray = s.toCharArray();
//        char[] tCharArray = t.toCharArray();
//        int n = sCharArray.length;
//        int[] f = new int[n + 1];
//        for (char c : sCharArray) {
//            int pre = f[0];
//            for (int j = 0; j < n; j++) {
//                int temp = f[j + 1];
//                if (c == tCharArray[j]) {
//                    f[j + 1] = pre + 1;
//                } else {
//                    f[j + 1] = Math.max(f[j + 1], f[j]);
//                }
//                pre = temp;
//            }
//        }
//        return f[n];
        /*
         * 思路2：区间dp：因为求的是回文子序列，由于回文串第一个元素与最后一个元素相同的特性
         * 正着枚举 i 与倒着枚举 j，如果s[i] == s[j] 则问题变为了 i+1 与 j-1 之间元素的最长回文子序列+1
         * 如果不等，不选 i 或 不选 j
         * f[i][j] 表示i到j构成的最长回文子序列
         * */
        char[] chars = s.toCharArray();
        int n = chars.length;
        int[] f = new int[n];
        for (int i = n - 1; i >= 0; i--) {
            f[i] = 1;
            int pre = 0;
            for (int j = i + 1; j < n; j++) {
                int temp = f[j];
                if (chars[i] == chars[j]) {
                    f[j] = pre + 2;
                } else {
                    f[j] = Math.max(f[j], f[j - 1]);
                }
                pre = temp;
            }
        }
        return f[n - 1];
//        return dfs(0, n - 1, chars);
    }

//    private int dfs(int i, int j, char[] chars) {
//        if (i == j) return 1;
//        if (i > j) return 0;
//        if (chars[i] == chars[j])
//            return dfs(i + 1, j - 1, chars) + 2;
//        else return Math.max(dfs(i + 1, j, chars), dfs(i, j - 1, chars));
//    }

    public int minScoreTriangulation(int[] values) {
        /*
         * 1039. 多边形三角剖分的最低得分
         * 你有一个凸的 n 边形，其每个顶点都有一个整数值。给定一个整数数组 values ，
         * 其中 values[i] 是按 顺时针顺序 第 i 个顶点的值。
         * 假设将多边形 剖分 为 n - 2 个三角形。
         * 对于每个三角形，该三角形的值是顶点标记的乘积，
         * 三角剖分的分数是进行三角剖分后所有 n - 2 个三角形的值之和。
         * 返回 多边形进行三角剖分后可以得到的最低分 。
         * 思路：这道题的关键点在于怎么找到子问题
         * 正着枚举 i 倒着枚举 j，dfs(i,j) 表示从 i 到 j 的多边形剖分的最低得分
         * 枚举 i 到 j 之间的顶点 k，原问题就变为了 从 i 到 k 的多边形剖分最低得分与 k 到 j 的最低得分 + i，j，k组成的三角形得分
         * */
        int n = values.length;
        int[][] f = new int[n][n];
        for (int i = n - 3; i >= 0; i--) {
            for (int j = i + 2; j < n; j++) {
                int res = Integer.MAX_VALUE;
                for (int k = j - 1; k > i; k--) {
                    res = Math.min(res, f[i][k] + f[k][j] + values[i] * values[j] * values[k]);
                }
                f[i][j] = res;
            }
        }
        return f[0][n - 1];
    }

//    private int ans = Integer.MIN_VALUE;

    public int diameterOfBinaryTree(TreeNode root) {
        /*
         *
         * 543. 二叉树的直径
         * 给你一棵二叉树的根节点，返回该树的 直径 。
         * 二叉树的 直径 是指树中任意两个节点之间最长路径的 长度 。这条路径可能经过也可能不经过根节点 root 。
         * 思路：直径一定经过叶子节点，因为如果不经过叶子节点，就还可以向下拓展
         * 定义 dfs(node) 返回以 node 为根节点的树的最大深度，则当前树的直径等与 左右子树的最大深度之和
         * */

//        dfs(root);
        return ans;
    }

//    private int dfs(TreeNode node) {
//        if (node == null) return 0;
//
//        int left = dfs(node.left);
//        int right = dfs(node.right);
//        ans = Math.max(ans, left + right);
//        return Math.max(left, right) + 1;
//    }

//    public int maxPathSum(TreeNode root) {
//        /*
//         * 124. 二叉树中的最大路径和
//         * 路径和 是路径中各节点值的总和。
//         * 给你一个二叉树的根节点 root ，返回其 最大路径和 。
//         * */
//        dfs(root);
//        return ans;
//    }

//    private int dfs(TreeNode node) {
//        if (node == null) return 0;
//        int left = dfs(node.left);
//        int right = dfs(node.right);
//        ans = Math.max(ans, left + right + node.val);
//        return Math.max(Math.max(left, right) + node.val, 0);
//    }

    private List<Integer>[] g;
    private char[] s;
    private int ans;

    public int longestPath(int[] parent, String s) {
        /*
         * 2246. 相邻字符不同的最长路径
         * 给你一棵 树（即一个连通、无向、无环图），根节点是节点 0 ，
         * 这棵树由编号从 0 到 n - 1 的 n 个节点组成。
         * 用下标从 0 开始、长度为 n 的数组 parent 来表示这棵树，
         * 其中 parent[i] 是节点 i 的父节点，由于节点 0 是根节点，所以 parent[0] == -1 。
         * 另给你一个字符串 s ，长度也是 n ，其中 s[i] 表示分配给节点 i 的字符。
         * 请你找出路径上任意一对相邻节点都没有分配到相同字符的 最长路径 ，并返回该路径的长度。
         * 相邻节点，对于二叉树，一个节点最多有三个相邻节点，左右子节点以及父节点
         * 对于一棵树，需要用for循环遍历其子节点
         * 思路：当前节点，需要知道它所有子树的深度（也就是链长），选择最大的两个，然后加上当前节点构成路径
         * 首先要构造一个列表，用来存储节点的相邻节点
         * */
        int n = parent.length;
        this.s = s.toCharArray();
        g = new ArrayList[n];
        Arrays.setAll(g, e -> new ArrayList<>());
        for (int i = 1; i < n; i++) {
            g[parent[i]].add(i);
        }
        dfs(0);
        return ans;
    }

    private int dfs(int x) {
        int maxLen = 0;
        for (int y : g[x]) {
            int len = dfs(y) + 1;
            if (s[x] != s[y]) {
                ans = Math.max(ans, maxLen + len);
                maxLen = Math.max(len, maxLen);
            }
        }
        return maxLen;
    }

    public int rob(TreeNode root) {
        /*
         * 337.打家劫舍 III
         * 小偷又发现了一个新的可行窃的地区。这个地区只有一个入口，我们称之为 root 。
         * 除了 root 之外，每栋房子有且只有一个“父“房子与之相连。
         * 一番侦察之后，聪明的小偷意识到“这个地方的所有房屋的排列类似于一棵二叉树”。
         * 如果 两个直接相连的房子在同一天晚上被打劫 ，房屋将自动报警。
         * 给定二叉树的 root 。返回 在不触动警报的情况下 ，小偷能够盗取的最高金额 。
         * 类比数组打家劫舍，枚举当前节点选不选，如果选，则递归到儿子的儿子
         * 如果不选，递归到儿子
         * 这个思路需要判断儿子是否为空，并且要分别递归到左儿子，右儿子
         * 代码十分复杂
         * 前面的思路，dfs的返回值是以 node 为根的二叉树的能偷的最高金额
         * 由于有选或不选两个情况，考虑分别返回再选的情况下的最高金额以及不选的情况下的最高金额
         * 这样如果选当前节点，则只需要从儿子中拿到不选儿子情况下的最高金额 + 当前节点值就构成了选情况下的最高金额
         * 如果不选，则取儿子节点中的选或不选的最高金额
         * 这个思路的目的是避免访问孙子节点，有点类似状态机，选只能从不选的状态转移来，不选则从较大者转移来
         * */
//        int[] nums = dfs(root);
//        return Math.max(nums[0], nums[1]);
        return 0;
    }

//    private int[] dfs(TreeNode node) {
//        if (node == null)
//            return new int[]{0, 0};
//        int[] left = dfs(node.left);
//        int[] right = dfs(node.right);
//        int rob = node.val + left[0] + right[0];
//        int notRob = Math.max(left[0], left[1]) + Math.max(right[0], right[1]);
//        return new int[]{notRob, rob};
//
//    }

    public int minCameraCover(TreeNode root) {
        /*
         * 968. 监控二叉树
         * 给定一个二叉树，我们在树的节点上安装摄像头。
         * 节点上的每个摄影头都可以监视其父对象、自身及其直接子对象。
         * 计算监控树的所有节点所需的最小摄像头数量。
         * 思路：找到原问题的子问题
         * 对于当前节点来说：有三种情况
         * 1. 安装摄像头
         * 2. 不安装摄像头，由其父节点安装摄像头
         * 3. 不安装摄像头，由其左右子节点安装摄像头
         * 如果 1，则原问题转化为了其左右子树所需最小摄像头的数量（当前节点安装了摄像头，其子树无论那种情况都可以） + 1
         * 如果 2，则原问题转化为了左右子树 1、3 的情况，因为当前节点不安装摄像头，所以左右子树的根节点不可能是情况 2
         * 如果 3，则原问题转化为了左右儿子节点至少有一个安装摄像头的情况下，并且要除情况2外子树所需最小摄像头数量
         * */

        int[] ans = dfs(root);
        return Math.min(ans[0], ans[2]);

    }

    private int[] dfs(TreeNode node) {
        if (node == null) return new int[]{Integer.MAX_VALUE / 2, 0, 0};
        int[] left = dfs(node.left);
        int[] right = dfs(node.right);
        int choose = Math.min(left[0], left[1])
                + Math.min(right[0], right[1]) + 1;
        int fa = Math.min(right[0], right[2]) + Math.min(left[0], left[2]);

        int children = fa + Math.clamp(Math.min(left[0] - left[2], right[0] - right[2]), 0, Integer.MAX_VALUE)；;
        return new int[]{choose, fa, children};
    }


}
