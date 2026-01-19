import javax.management.modelmbean.ModelMBean;
import javax.print.DocFlavor;
import javax.xml.stream.FactoryConfigurationError;
import java.lang.annotation.Target;
import java.util.*;
import java.util.function.DoublePredicate;

class Solution {


//    public int rob(int[] nums) {
//        /*198. 打家劫舍
//         * 给定一个代表每个房屋存放金额的非负整数数组，计算一夜之内能够偷窃到的最高金额。
//         * 必须间隔偷窃
//         * 思路：从回溯到记忆化搜索再到动态规划
//         * 回溯：枚举第 i 个元素选或不选；
//         * 子问题：前 i 个最大金额；
//         * 子子问题：前 i-2（选了第 i 个）最大金额和前 i-1 最大金额的最大值
//         * 记忆化搜索，对于计算过的 dfs(i) 将它记录到hash表中，再次计算时直接取
//         * 动态规划：状态数组记录 dfs(i)，循环代替递归，状态数组初始化代替递归边界条件
//         * */
////        int n = nums.length;
////        int[] cache = new int[n];
////        Arrays.fill(cache, -1);
////        return dfs(n - 1, nums, cache);
//        int n = nums.length;

    /// /        int[] f = new int[n+2];
    /// /        for (int i = 0; i < n; i++) {
    /// /            f[i+2] = Math.max(f[i+1], f[i] + nums[i]);
    /// /        }
    /// /        return f[n+1];
//        int f0 = 0, f1 = 0;
//        for (int i = 0; i < n; i++) {
//            int new_f = Math.max(f1, f0 + nums[i]);
//            f0 = f1;
//            f1 = new_f;
//        }
//        return f1;
//
//    }

    // 倒着枚举，第 i 个元素选不选
//    private int dfs(int i, int[] nums, int[] cache) {
//        if (i < 0) {
//            return 0;
//        }
//        if (cache[i] != -1)
//            return cache[i];
//        cache[i] = Math.max(dfs(i - 1, nums, cache), dfs(i - 2, nums, cache) + nums[i]);
//        return cache[i];
//    }
    public int climbStairs(int n) {
        /*70.爬楼梯
        * 假设你正在爬楼梯。需要 n 阶你才能到达楼顶。
          每次你可以爬 1 或 2 个台阶。你有多少种不同的方法可以爬到楼顶呢？
          * 回溯：枚举第n阶台阶，子问题：前n阶台阶的爬法 子子问题：前n-1阶与前n-2阶的爬法
          *
        * */
//        int[] cache = new int[n+1];
//        Arrays.fill(cache, -1);
//        return dfs(n, cache);

        int f0 = 1, f1 = 1;
        for (int i = 2; i <= n; i++) {
            int new_f = f1 + f0;
            f0 = f1;
            f1 = new_f;
        }
        return f1;

    }

//    private int dfs(int n, int[] cache) {
//        if (n < 0) {
//            return 0;
//        }
//        if (n == 1)
//            return 1;
//        if (n == 2)
//            return 2;
//        if(cache[n]!=-1)
//            return cache[n];
//        cache[n] = dfs(n - 1,cache) + dfs(n - 2,cache);
//        return cache[n];
//
//    }

    public int minCostClimbingStairs(int[] cost) {
        /*746.使用最小花费爬楼梯
        回溯：枚举第 i 个，爬到第 i 个最小花费，爬到 i -1 个最小花费，
         * 动态规划：当前状态可以由前一个状态来，或前前一个状态来
         * */
//        int n = cost.length;
//        int[] cache = new int[n + 1];
//        Arrays.fill(cache,-1);
//        return dfs(n, cost, cache);
//        int n = cost.length;
//        int[] f = new int[n + 1];
//        f[0] = f[1] = 0;
//        for (int i = 0; i < n-1; i++) {
//            f[i + 2] = Math.min(f[i + 1] + cost[i + 1], f[i] + cost[i]);
//        }
//        return f[n];
        int f0 = 0;
        int f1 = 0;
        for (int i = 0; i < cost.length - 1; i++) {
            int new_f = Math.min(f0 + cost[i], f1 + cost[i + 1]);
            f0 = f1;
            f1 = new_f;
        }
        return f1;

    }

//    private int dfs(int i, int[] cost, int[] cache) {
//        if (i <= 1)
//            return 0;
//        if (cache[i] != -1)
//            return cache[i];
//        cache[i] = Math.min(dfs(i - 1, cost, cache) + cost[i - 1], dfs(i - 2, cost, cache) + cost[i - 2]);
//        return cache[i];
//
//    }

    public int climbStairs(int n, int[] costs) {
        /*3693.爬楼梯
         *
         * */
        int f0 = 0, f1 = 0, f2 = 0;
        for (int i = 1; i <= n; i++) {
            int new_f = Math.min(Math.min(f0 + 9, f1 + 4), f2 + 1) + costs[i - 1];
            f0 = f1;
            f1 = f2;
            f2 = new_f;
        }
        return f2;


    }

    public int rob(int[] nums) {
        /*213.打家劫舍 Ⅱ
         *
         * */
        int n = nums.length;
        if (n == 1)
            return nums[0];
        if (n % 2 == 0) {
            int f0 = 0, f1 = 0;
            for (int i = 0; i < n; i++) {
                int new_f = Math.max(f0 + nums[i], f1);
                f0 = f1;
                f1 = new_f;
            }
            return f1;
        } else {
            int f0 = 0, f1 = 0;
            for (int i = 0; i < n - 1; i++) {
                int new_f = Math.max(f0 + nums[i], f1);
                f0 = f1;
                f1 = new_f;
            }
            int f00 = 0, f11 = 0;
            for (int i = 1; i < n; i++) {
                int new_f = Math.max(f00 + nums[i], f11);
                f00 = f11;
                f11 = new_f;
            }
            return Math.max(f1, f11);
        }

    }


    private int max;

    public int deleteAndEarn(int[] nums) {
        /*740.删除并获得点数
         * 每次操作选择任意一个 nums[i] 将它与所有nums[i]+-1一同删除，并获得 nums[i]的点数，返回最大点数
         * 回溯：倒着遍历，枚举当前元素选不选，选则将 nums[i]+1,nums[i]-1加入路劲表示不能选了，
         * 遍历到 nums[j] 时如果在路径中，直接不选,记忆化搜索并不能解决时间复杂度
         * */
//        int n = nums.length;
//        Map<Integer, Integer> path = new HashMap<>();
//        return dfs(n - 1, nums, path);

        /*思路2：将它转化为打家劫舍，
        根据题意选择了 nums[i] 之后不能选所有的 nums[i]+1,nums[i]-1，
        创建一个值域数组 sums，sums[i] 表示在nums 中所有值为 i 的数的和，
        例如nums = [2,2,3,3,3,4], sums=[0,0,4,9,4],这样就转化为了打家劫舍
        * */
        int max = 0;
        for (int num : nums) {
            max = Math.max(num, max);
        }
        int[] sums = new int[max + 1];
        for (int num : nums) {
            sums[num] += num;
        }

        int f0 = 0, f1 = 0;
        for (int i = 0; i < max + 1; i++) {
            int new_f = Math.max(f1, f0 + sums[i]);
            f0 = f1;
            f1 = new_f;
        }
        return f1;


    }

    // 返回值表示前i个数的最大值
//    private int dfs(int i, int[] nums, Map<Integer, Integer> path) {
//        int sum1 = 0;
//        int sum2 = 0;
//        if (i < 0)
//            return 0;
//        // 选：当前元素不在路劲中
//        if (!path.containsKey(nums[i]) || path.get(nums[i]) == 0) {
//            path.merge(nums[i] + 1, 1, Integer::sum);
//            path.merge(nums[i] - 1, 1, Integer::sum);
//            sum1 += nums[i] + dfs(i - 1, nums, path);
//            path.merge(nums[i] + 1, -1, Integer::sum);
//            path.merge(nums[i] - 1, -1, Integer::sum);
//        }
//        // 不选
//        sum2 = dfs(i - 1, nums, path);
//        return Math.max(sum1, sum2);
//    }

//    private static final int MOD = 1_000_000_007;

    public int countGoodStrings(int low, int high, int zero, int one) {
        /*2466.统计构造好字符串的方案数
        * 整数 zero ，one ，low 和 high ，从空字符串开始构造一个字符串，每一步执行下面操作中的一种：
        将 '0' 在字符串末尾添加 zero  次。
        将 '1' 在字符串末尾添加 one 次。
        以上操作可以执行任意次。
        如果通过以上过程得到一个 长度 在 low 和 high 之间（包含上下边界）的字符串，那么这个字符串我们称为好字符串。
        请你返回满足以上要求的不同好字符串数目。由于答案可能很大，请将结果对 10^9 + 7 取余后返回。
        * 枚举答案选哪个：
        * */
//        dfs(low, high, zero, one);
//        return ans;
        // 爬楼梯的变形：相当于每次可以爬 zero 或 one 个台阶，返回爬到 low 到 high 个台阶的方案数
        // 回溯:dfs(i) 表示爬到 i 的方案数，dfs(i) = dfs(i-zero) + dfs(i-one)
//        int ans = 0;
//        int[] cache = new int[high + 1];
//        Arrays.fill(cache, -1);
//        for (int i = low; i <= high; i++) {
//            ans = (ans + dfs(i, zero, one, cache)) % MOD;
//        }
//        return ans;
        // 动态规划
        final int MOD = 1_000_000_007;
        int[] dp = new int[high + 1];
        dp[0] = 1;
        int ans = 0;
        for (int i = 1; i < high + 1; i++) {
            if (i > zero) dp[i] = dp[i - zero];
            if (i > one) dp[i] = (dp[i] + dp[i - one]) % MOD;
            if (i >= low)
                ans = (ans + dp[i]) % MOD;
        }
        return ans;

    }

//    private int dfs(int i, int zero, int one, int[] cache) {
//        if (i < 0)
//            return 0;
//        if (i == 0)
//            return 1;
//        if (cache[i] != -1)
//            return cache[i];
//        return cache[i] = (dfs(i - zero, zero, one, cache) + dfs(i - one, zero, one, cache)) % MOD;
//    }


//    private void dfs(int low, int high, int zero, int one) {
//
//        if (high < 0) return;
//        if (low <= 0)
//            ans++;
//
//        // 选0
//        dfs(low - zero, high - zero, zero, one);
//
//
//        // 选 1
//        dfs(low - one, high - one, zero, one);
//
//    }

    public int combinationSum4(int[] nums, int target) {
        /*377.组合总和Ⅳ
         * 给你一个由不同整数组成的数组 nums ，和一个目标整数 target 。请你从 nums 中找出并返回总和为 target 的元素组合的个数。
         * 回溯 dfs(i) 表示前 i 个数总和为 target 的元素
         * */
//        int n = nums.length;
//        return dfs(n - 1, nums, target);
        /*
         * 本质是爬楼梯，爬 target 阶有多少种爬法，每次可以爬 nums[j] 阶
         * dfs(i) 表示爬 i 个台阶的方案，枚举最后一步爬了 nums[j] 阶
         * */
//        int[] cache = new int[target + 1];
//        Arrays.fill(cache, -1);
//        return dfs(target, nums, cache);
        int[] dp = new int[target + 1];
        dp[0] = 1;
        for (int i = 1; i < target + 1; i++) {
            for (int num : nums) {
                if (i >= num) dp[i] += dp[i - num];
            }
        }
        return dp[target];
    }

//    private int dfs(int target, int[] nums, int[] cache) {
//        if (target == 0)
//            return 1;
//        if (cache[target] != -1)
//            return cache[target];
//        int sum = 0;
//        for (int num : nums) {
//            if (num > target) continue;
//            sum += dfs(target - num, nums,cache);
//        }
//        return cache[target] = sum;
//
//    }

    //dfs(i) 表示前 i 个数总和为 target 的元素
//    private int dfs(int i, int[] nums, int target) {
//        if (target == 0)
//            return 1;
//        if (i < 0 || target < 0) return 0;
//
//        return dfs(i, nums, target - nums[i]) + dfs(i - 1, nums, target);
//    }
    private static final int MX = 100001;
    private static final int MOD = 1000000007;
    private static final long[] f = new long[MX];
    private static final long[] g = new long[MX];

//    private void init() {
//        f[0] = g[0] = 1;
//        f[1] = g[1] = 1;
//        f[2] = g[2] = 2;
//        f[3] = g[3] = 4;
//        for (int i = 4; i < MX; i++) {
//            f[i] = (f[i - 1] + f[i - 2] + f[i - 3]) % MOD;
//            g[i] = (g[i - 1] + g[i - 2] + g[i - 3] + g[i - 4]) % MOD;
//        }
//
//    }

    public int countTexts(String pressedKeys) {
        /*2266.统计打字方案数
        思路：分组+爬楼梯：例如“22233”
        分组 2 的target为3，可以爬一步、二步、三步，有 4 种爬法，
        分组 3 的target 为2，可以爬 1、2、3 步，有 2 种爬法，最终有 2*4种爬法
        * */
        init();
        long ans = 1L;
        char[] chars = pressedKeys.toCharArray();
        int cnt = 0;
        for (int i = 0; i < chars.length; i++) {
            char c = chars[i];
            cnt++;
            if (i == chars.length - 1 || c != chars[i + 1]) {
                ans = (ans * (c == '7' || c == '9' ? g[cnt] : f[cnt])) % MOD;
                cnt = 0;
            }
        }


        return (int) ans;
    }

    public int minPathSum(int[][] grid) {
        /*64.最小路径和
         * 给定一个包含非负整数的 m x n 网格 grid ，
         * 请找出一条从左上角到右下角的路径，使得路径上的数字总和为最小。
         * 说明：每次只能向下或者向右移动一步。
         * 从上到下逐个遍历，dp[i][j]状态的最小值只能来自dp[i][j-1] 或 dp[i-1][j]
         * */

        int m = grid.length;
        int n = grid[0].length;
        int[][] dp = new int[m + 1][n + 1];
        Arrays.fill(dp[0], Integer.MAX_VALUE);
        dp[0][1] = 0;
        for (int i = 0; i < m; i++) {
            dp[i + 1][0] = Integer.MAX_VALUE;
            for (int j = 0; j < n; j++) {

                dp[i + 1][j + 1] = Math.min(dp[i][j + 1], dp[i + 1][j]) + grid[i][j];
            }
        }


        return dp[m][n];

    }

    public int findTargetSumWays(int[] nums, int target) {
        /*494.目标和
         * 给你一个非负整数数组 nums 和一个整数 target 。
         * 数组元素前添加 + 或 -，然后所有元素求和为 target 的不同方案数
         * 思路：设 p 为选择添加 + 数的和，则添加 - 的元素的和为 s - p，s 为总和
         * 根据题意 target = p-(s-p) => p = (s+t)/2
         * 所以题目转化为选择 p 个数，使其和恰好为 (s+t)/2，这是零一背包的变形
         * 枚举每个数，都有选或不选，子问题 i 以前的数方案数
         * 子子问题：选 dfs(i,t-nums[i]) 不选(dfs,t)
         * */

//        for (int num : nums) {
//            target += num;
//        }
//        if (target < 0 || target % 2 == 1)
//            return 0;
//
//        target = target / 2;
//        int n = nums.length;
//        int[][] cache = new int[n][target + 1];
//        for (int[] ints : cache) {
//            Arrays.fill(ints, -1);
//        }
//        return dfs(n - 1, target, nums, cache);
        // 翻译为动态规划
        for (int num : nums) {
            target += num;
        }
        if (target < 0 || target % 2 == 1)
            return 0;

        target = target / 2;
        int n = nums.length;
//        int[][] dp = new int[n + 1][target + 1];
//        dp[0][0] = 1;
//        for (int i = 0; i < n ; i++) {
//            for (int j = 0; j < target + 1; j++) {
//                if (nums[i] > j)
//                    dp[i + 1][j] = dp[i][j];
//                else
//                    dp[i + 1][j] = dp[i][j] + dp[i][j - nums[i]];
//            }
//        }
//        return dp[n][target];
        // 空间复杂度优化
//        int[][] dp = new int[2][target + 1];
//        dp[0][0] = 1;
//        for (int i = 0; i < n; i++) {
//            for (int j = 0; j < target + 1; j++) {
//                if (nums[i] > j)
//                    dp[(i + 1) % 2][j] = dp[i % 2][j];
//                else
//                    dp[(i + 1) % 2][j] = dp[i % 2][j] + dp[i % 2][j - nums[i]];
//            }
//        }
//        return dp[n % 2][target];
        // 继续优化
        int[] dp = new int[target + 1];
        dp[0] = 1;
        for (int num : nums) {
            for (int j = target; j >= num; j--) {
                dp[j] = dp[j] + dp[j - num];
            }
        }
        return dp[target];

    }

//    private int dfs(int i, int target, int[] nums, int[][] cache) {
//        if (i < 0)
//            return target == 0 ? 1 : 0;
//        if (cache[i][target] != -1)
//            return cache[i][target];
//        if (nums[i] > target) {
//            cache[i][target] = dfs(i - 1, target, nums, cache);
//            return cache[i][target];
//        }
//
//        return cache[i][target] = dfs(i - 1, target, nums, cache) + dfs(i - 1, target - nums[i], nums, cache);
//    }

    public int coinChange(int[] coins, int amount) {
        /*322.零钱兑换
         * 给你一个整数数组 coins ，表示不同面额的硬币；以及一个整数 amount ，表示总金额。
         * 计算并返回可以凑成总金额所需的最少的硬币个数 。如果没有任何一种硬币组合能组成总金额，返回 -1 。
         * 你可以认为每种硬币的数量是无限的。
         * 完全背包的变形：回溯枚举每个元素，都可以选或者不选
         * 子问题：前 i 个元素和为 amount 的最少硬币数量
         * 子子问题：选了第 i 个元素 dfs(i,amount-coins[i]) 没选第 i 个元素 dfs(i-1,amount);
         * */
//        int n = coins.length;
//        long[][] cache = new long[n + 1][amount + 1];
//        for (long[] longs : cache) {
//            Arrays.fill(longs, Integer.MAX_VALUE);
//        }
//        cache[0][0] = 0;
//        long ans = dfs(n - 1, coins, amount, cache);
//        return ans == Integer.MAX_VALUE ? -1 : (int) ans;

        // 翻译为递推
//        int n = coins.length;
//        long[][] dp = new long[2][amount + 1];
//        Arrays.fill(dp[0], Integer.MAX_VALUE);
//        dp[0][0] = 0;
//        for (int i = 0; i < n; i++) {
//            for (int j = 0; j < amount + 1; j++) {
//                if (coins[i] > j)
//                    dp[(i + 1) % 2][j] = dp[i % 2][j];
//                else
//                    dp[(i + 1) % 2][j] = Math.min(dp[(i + 1) % 2][j - coins[i]] + 1, dp[i % 2][j]);
//            }
//        }
//        return dp[n%2][amount] == Integer.MAX_VALUE ? -1 : (int) dp[n%2][amount];

        // 进一步优化
        int n = coins.length;
        int[] dp = new int[amount + 1];
        Arrays.fill(dp, Integer.MAX_VALUE / 2);
        dp[0] = 0;
        for (int coin : coins) {
            for (int j = coin; j < amount + 1; j++) {
                dp[j] = Math.min(dp[j - coin] + 1, dp[j]);
            }
        }
        return dp[amount] >= Integer.MAX_VALUE / 2 ? -1 : dp[amount];
    }

    // 方法返回值表示：前 i 个元素和为 amount 的最少硬币数量
//    private long dfs(int i, int[] coins, int amount, long[][] cache) {
//        if (i < 0)
//            return amount == 0 ? 0 : Integer.MAX_VALUE;
//
//        if (cache[i][amount] != Integer.MAX_VALUE)
//            return cache[i][amount];
//
//        if (coins[i] > amount)
//            // 不选
//            return cache[i][amount] = dfs(i - 1, coins, amount, cache);
//
//        return cache[i][amount] = Math.min(dfs(i, coins, amount - coins[i], cache) + 1, dfs(i - 1, coins, amount, cache));
//    }

    public int lengthOfLongestSubsequence(List<Integer> nums, int target) {
        /*2915.和为目标值的最长子序列长度
        给你一个下标从 0 开始的整数数组 nums 和一个整数 target 。
        返回和为 target 的 nums 子序列中，子序列 长度的最大值 。如果不存在和为 target 的子序列，返回 -1 。
        回溯：对于每个元素是否加入子序列有选或不选，子问题：前 i 个元素和为 target 的最长子序列
        子子问题：对于当前元素，选了 dfs(i-1,target-nums[i]) 不选 dfs(i-1,target)
        * */
//        int n = nums.size();
//        int[][] cache = new int[n + 1][target + 1];
//        for (int[] ints : cache) {
//            Arrays.fill(ints, -1);
//        }
//        int ans = dfs(n - 1, target, nums, cache);
//        return ans < 0 ? -1 : ans;
        // 翻译为递推
//        int n = nums.size();
//        int[][] dp = new int[n + 1][target + 1];
//        Arrays.fill(dp[0], Integer.MIN_VALUE / 2);
//        dp[0][0] = 0;
//        for (int i = 0; i < n; i++) {
//            for (int j = 0; j < target + 1; j++) {
//                if (nums.get(i) > j)
//                    dp[i + 1][j] = dp[i][j];
//                else
//                    dp[i + 1][j] = Math.max(dp[i][j - nums.get(i)] + 1, dp[i][j]);
//            }
//        }
//        int ans = dp[n][target];
//        return ans < 0 ? -1 : ans;
        // 空间优化
//        int n = nums.size();
//        int[][] dp = new int[2][target + 1];
//        Arrays.fill(dp[0], Integer.MIN_VALUE / 2);
//        dp[0][0] = 0;
//        for (int i = 0; i < n; i++) {
//            for (int j = 0; j < target + 1; j++) {
//                if (nums.get(i) > j)
//                    dp[(i + 1) % 2][j] = dp[i % 2][j];
//                else
//                    dp[(i + 1) % 2][j] = Math.max(dp[i % 2][j - nums.get(i)] + 1, dp[i % 2][j]);
//            }
//        }
//        int ans = dp[n % 2][target];
//        return ans < 0 ? -1 : ans;
        // 进一步优化
        int n = nums.size();
        int[] dp = new int[target + 1];
        Arrays.fill(dp, Integer.MIN_VALUE);
        dp[0] = 0;
        int sum = 0;
        for (int num : nums) {
            sum = Math.min(num + sum, target);
            for (int j = sum; j >= num; j--) {
                dp[j] = Math.max(dp[j - num] + 1, dp[j]);
            }
        }

        int ans = dp[target];
        return ans < 0 ? -1 : ans;
    }

    // 返回值表示前i个元素中和为 target 的最长子序列
//    private int dfs(int i, int target, List<Integer> nums, int[][] cache) {
//        if (i < 0) {
//            // 枚举完了
//            return target == 0 ? 0 : Integer.MIN_VALUE / 2;
//        }
//        if (cache[i][target] != -1)
//            return cache[i][target];
//        if (nums.get(i) > target) {
//            return cache[i][target] = dfs(i - 1, target, nums, cache);
//        }
//        return cache[i][target] = Math.max(dfs(i - 1, target - nums.get(i), nums, cache) + 1, dfs(i - 1, target, nums, cache));
//    }

    public boolean canPartition(int[] nums) {
        /*416.分割等和子集
         * 给你一个只包含正整数的非空数组 nums 。请你判断是否可以将这个数组分割成两个子集，使得两个子集的元素和相等。
         * 思路：sum 为元素和，要为偶数，转化为 target = sum/2，是否能够分割为两个子集的和为 target
         * 01背包，每个元素选了不能重复选，capacity 为 target，恰好为装满的情况是否存在
//         * */
//        int target = 0;
//        for (int num : nums) {
//            target += num;
//        }
//        if (target % 2 != 0)
//            return false;
//        target = target / 2;
//        int n = nums.length;
//        int[][] cache = new int[n][target + 1];
//        for (int[] ints : cache) {
//            Arrays.fill(ints, -1);
//        }
//        return dfs(n - 1, target, nums, cache);
        // 递推
        int target = 0;
        for (int num : nums) {
            target += num;
        }
        if (target % 2 != 0)
            return false;
        target = target / 2;
        boolean[] dp = new boolean[target + 1];
        dp[0] = true;
        int sum = 0;
        for (int num : nums) {
            sum = Math.min(sum + num, target);
            for (int j = sum; j >= num; j--) {
                dp[j] = dp[j] || dp[j - num];
            }
            if (dp[target]) {
                return true;
            }
        }
        return false;


    }

    // dfs(i,j) 表示前i 个数和是否有恰好为j的方案数
//    private boolean dfs(int i, int target, int[] nums, int[][] cache) {
//        if (i < 0) {
//            return target == 0;
//        }
//
//        if (cache[i][target] != -1)
//            return cache[i][target] == 1;
//
//        boolean res;
//        if (nums[i] > target) {
//            res = dfs(i - 1, target, nums, cache);
//
//        } else res = dfs(i - 1, target, nums, cache) || dfs(i - 1, target - nums[i], nums, cache);
//
//        cache[i][target] = res ? 1 : 0;
//
//        return res;
//
//    }
//    private final int MOD = 1000000007;

    public int numberOfWays(int n, int x) {
        /*2787.将一个数字表示成幂的和的方案数
         * 给你两个正整数 n 和 x 。
         * 请你返回将 n 表示成一些 互不相同正整数的 x 次幂之和的方案数。
         * 即需要返回互不相同整数 [n1, n2, ..., nk] 的集合数目，满足 n = n1^x + n2^x + ... + nk^x 。
         * 由于答案可能非常大，请你将它对 109 + 7 取余后返回。
         * 思路：互不相同体现了这是 01背包，capacity = n，体积是 n1^x，数组是 1~n的x次方根向上取整
         * */
//        int nums = root(n, x);
//        int[][] cache = new int[nums + 1][n + 1];
//        for (int[] ints : cache) {
//            Arrays.fill(ints, -1);
//        }
//        return dfs(nums, n, x, cache);
        // 翻译为递推
//        int nums = root(n, x);
//        int[] dp = new int[n + 1];
//        dp[0] = 1;
//        int sum = 0;
//        for (int i = 1; i <= nums; i++) {
//            int num = pow(i, x);
//            sum = Math.min(sum + num, n);
//            for (int j = sum; j >= num; j--) {
//                dp[j] = (dp[j] + dp[j - num]) % MOD;
//            }
//        }
//        return dp[n];
        long[] dp = new long[n + 1];
        dp[0] = 1;
        for (int i = 1; Math.pow(i, x) <= n; i++) {
            int num = (int) Math.pow(i, x);
            for (int j = n; j >= num; j--) {
                dp[j] = dp[j] + dp[j - num];
            }
        }
        return (int) (dp[n] % 1000000007);

    }

    // 表示前 i 个数的x次方和为 target 的方案数
//    private int dfs(int i, int target, int x, int[][] cache) {
//        if (i < 1)
//            return target == 0 ? 1 : 0;
//
//        if (cache[i][target] != -1)
//            return cache[i][target];
//
//        int res;
//        if (pow(i, x) > target)
//            res = (dfs(i - 1, target, x, cache)) % MOD;
//        else
//            res = (dfs(i - 1, target, x, cache) + dfs(i - 1, target - pow(i, x), x, cache)) % MOD;
//        return cache[i][target] = res;
//
//    }

//    private int pow(int i, int x) {
//        int sum = i;
//        while (x > 1) {
//            sum = sum * i;
//            x--;
//        }
//        return sum;
//
//    }
//
//    private int root(int n, int x) {
//        for (int i = 1; i < n; i++) {
//            int s = i;
//            for (int j = 0; j < x - 1; j++) {
//                s = s * i;
//                if (s > n)
//                    return i - 1;
//            }
//        }
//        return n;
//
//    }


    public int change(int amount, int[] coins) {
        /*518.零钱兑换 Ⅱ
         * 整数数组 coins 表示不同面额的硬币，整数 amount 表示总金额。
         * 计算并返回可以凑成总金额的硬币组合数。如果任何硬币组合都无法凑出总金额，返回 0 。
         * 假设每一种面额的硬币有无限个。
         * 思路：完全背包，恰好为 amount 的方案数，体积 coins[i]
         * */
        int[] dp = new int[amount + 1];
        dp[0] = 1;
        for (int coin : coins) {
            for (int j = coin; j < amount + 1; j++) {
                dp[j] = dp[j] + dp[j - coin];
            }
        }
        return dp[amount];
    }

    private static boolean flag = false;
    private static final int[] dp = new int[10001];

    private void init() {
        if (flag)
            return;
        flag = true;
        Arrays.fill(dp, Integer.MAX_VALUE);
        dp[0] = 0;

        for (int i = 1; i < 101; i++) {
            int num = (int) Math.pow(i, 2);
            for (int j = num; j < 10001; j++) {
                dp[j] = Math.min(dp[j], dp[j - num] + 1);
            }
        }

    }

    public int numSquares(int n) {
        /*279.完全平方数
         * 整数 n ，返回和为 n 的完全平方数的最少数量 。
         * 完全平方数 是一个整数，其值等于另一个整数的平方；
         * 换句话说，其值等于一个整数自乘的积。
         * 例如，1、4、9 和 16 都是完全平方数，而 3 和 11 不是。
         * 思路：完全背包问题，容量恰好为 n 的最少价值和，体积 i^2，价值 1
         * */
//        int nums = 1;
//        while (Math.pow(nums, 2) <= n)
//            nums++;
//
//        int[] dp = new int[n + 1];
//
//        Arrays.fill(dp, Integer.MAX_VALUE / 2);
//        dp[0] = 0;
//
//        for (int i = 1; i < nums; i++) {
//            int num = (int) Math.pow(i, 2);
//            for (int j = num; j < n + 1; j++) {
//                dp[j] = Math.min(dp[j], dp[j - num] + 1);
//            }
//
//        }
//        return dp[n];
//        int nums = 1;
//        while (Math.pow(nums, 2) <= n)
//            nums++;
//        int[][] cache = new int[nums][n + 1];
//        for (int[] ints : cache) {
//            Arrays.fill(ints, -1);
//        }
//        return dfs(nums - 1, n, cache);

        // 时间优化，dp 数组预处理
        init();
        return dp[n];

    }

    // dfs(i,t) 表示前 i 个数的平方和为 target 最少数量
//    private int dfs(int i, int target, int[][] cache) {
//        if (i < 1)
//            return target == 0 ? 0 : Integer.MAX_VALUE / 2;
//
//        if (cache[i][target] != -1)
//            return cache[i][target];
//
//        int x = (int) Math.pow(i, 2);
//        int res;
//        if (x > target) {
//            // 不选
//            res = dfs(i - 1, target, cache);
//        } else res = Math.min(dfs(i - 1, target, cache), dfs(i, target - x, cache) + 1);
//        return cache[i][target] = res;
//
//    }
    public int longestCommonSubsequence(String text1, String text2) {
        /*1143.最长公共子序列
         * 给定两个字符串 text1 和 text2，返回这两个字符串的最长公共子序列的长度。如果不存在公共子序列 ，返回 0
         * 思路：回溯枚举当前字符 i,j 选或者不选；
         * 子问题：前i、j个字符串的最长子序列。
         * 下一个子问题：选了，不选
         * */
//        char[] chars1 = text1.toCharArray();
//        char[] chars2 = text2.toCharArray();
//        int n = chars1.length;
//        int m = chars2.length;
//        int[][] cache = new int[n][m];
//        for (int[] ints : cache) {
//            Arrays.fill(ints, -1);
//        }
//        return dfs(n - 1, m - 1, chars1, chars2, cache);

        // 翻译为递推
//        char[] chars1 = text1.toCharArray();
//        char[] chars2 = text2.toCharArray();
//        int n = chars1.length;
//        int m = chars2.length;
//        int[][] dp = new int[n + 1][m + 1];
//        for (int i = 0; i < n; i++) {
//            for (int j = 0; j < m; j++) {
//                if (chars1[i] == chars2[j]) dp[i + 1][j + 1] = dp[i][j] + 1;
//                else dp[i + 1][j + 1] = Math.max(dp[i][j + 1], dp[i + 1][j]);
//            }
//        }
//        return dp[n][m];
        // 进一步优化空间
        char[] chars1 = text1.toCharArray();
        char[] chars2 = text2.toCharArray();
        int m = chars2.length;
        int[] dp = new int[m + 1];
        for (char c : chars1) {
            int pre = 0;
            for (int j = 0; j < m; j++) {
                int temp = dp[j + 1];
                if (c == chars2[j]) dp[j + 1] = pre + 1;
                else dp[j + 1] = Math.max(dp[j + 1], dp[j]);
                pre = temp;
            }
        }
        return dp[m];
    }

    //    dfs(i,j) 表示前i个字符与前j个字符的最长公共子序列长度
//    private int dfs(int i, int j, char[] chars1, char[] chars2, int[][] cache) {
//        if (i < 0 || j < 0) {
//            // 枚举到头了
//            return 0;
//        }
//        if (cache[i][j] != -1)
//            return cache[i][j];
//
//
//        if (chars1[i] == chars2[j]) return cache[i][j] = dfs(i - 1, j - 1, chars1, chars2, cache) + 1;
//
//        return cache[i][j] = Math.max(dfs(i - 1, j, chars1, chars2, cache), dfs(i, j - 1, chars1, chars2, cache));
//
//    }
//    private char[] chars1;
//    private char[] chars2;

//    public int minDistance(String word1, String word2) {
//        /*72.编辑距离
//         * 给你两个单词 word1 和 word2，请返回将 word1 转换成 word2 所使用的最少操作数。
//         * 你可以对一个单词进行如下三种操作: 插入一个字符、删除一个字符、替换一个字符
//         * 思路：倒序枚举每个单词 i、j，如果 i 和 j 相等，直接进入子问题
//         * 如果 i ！= j，执行返回插入、删除、替换中较小值 +1
//         * 插入操作等价于 dfs(i,j-1) 删除操作等价于 dfs(i-1,j) 替换操作等价于 dfs(i-1,j-1)
//         * */
////        chars1 = word1.toCharArray();
////        chars2 = word2.toCharArray();
////        int n = chars1.length;
////        int m = chars2.length;
////        int[][] cache = new int[n][m];
////        for (int[] ints : cache) {
////            Arrays.fill(ints, -1);
////        }
////        return dfs(n - 1, m - 1, cache);
//        // 翻译为递推

    /// /        char[] chars1 = word1.toCharArray();
    /// /        char[] chars2 = word2.toCharArray();
    /// /        int n = chars1.length;
    /// /        int m = chars2.length;
    /// /        int[][] dp = new int[n + 1][m+1];
    /// /        for (int i = 0; i < m; i++) {
    /// /            dp[0][i+1] = i + 1;
    /// /        }
    /// /
    /// /        for (int i = 0; i < n; i++) {
    /// /            dp[i+1][0] = i + 1;
    /// /            for (int j = 0; j < m; j++) {
    /// /                if (chars2[j] == chars1[i]) dp[i + 1][j+1] = dp[i][j];
    /// /                else dp[i + 1][j+1] = Math.min(Math.min(dp[i + 1][j], dp[i][j+1]), dp[i][j]) + 1;
    /// /            }
    /// /        }
    /// /        return dp[n][m];
//        // 优化空间
//        char[] chars1 = word1.toCharArray();
//        char[] chars2 = word2.toCharArray();
//        int n = chars1.length;
//        int m = chars2.length;
//        int[] dp = new int[m + 1];
//        for (int j = 0; j < m; j++) {
//            dp[j + 1] = j + 1;
//        }
//
//        for (int i = 0; i < n; i++) {
//            int pre = dp[0];
//            dp[0] = i + 1;
//            for (int j = 0; j < m; j++) {
//                int temp = dp[j + 1];
//                if (chars2[j] == chars1[i]) dp[j + 1] = pre;
//                else dp[j + 1] = Math.min(Math.min(dp[j], dp[j + 1]), pre) + 1;
//                pre = temp;
//            }
//        }
//        return dp[m];
//
//    }

    //    dfs(i,j) 表示将前i个字符转化为前 j 个字符的最少操作数
//    private int dfs(int i, int j, int[][] cache) {
//        if (i < 0) {
//            // 表示word1字符比word2字符短，剩下的需要插入，短多少
//            return j + 1;
//        }
//        if (j < 0) return i + 1;
//
//        if (cache[i][j] != -1) return cache[i][j];
//
//        if (chars1[i] == chars2[j]) return cache[i][j] = dfs(i - 1, j - 1, cache);
//
//        return cache[i][j] = (Math.min(Math.min(dfs(i, j-1, cache), dfs(i - 1, j, cache)), dfs(i - 1, j - 1, cache)) + 1);
//
//    }
//    private char[] s;
//    private char[] t;
//    private int[][] cache;
    public int minDistance(String word1, String word2) {
        /*583.两个字符串的删除操作
         * 给定两个单词 word1 和 word2 ，返回使得 word1 和  word2 相同所需的最小步数。
         * 每步 可以删除任意一个字符串中的一个字符。
         * 思路：枚举每个字符 i、j，子问题前i、j个字符串相同的最小步数
         * 下一个子问题，删除i，j；删除i，不删除j；删除j，不删除i
         * */
//        s = word1.toCharArray();
//        t = word2.toCharArray();
//        int n = s.length;
//        int m = t.length;
//        cache = new int[n][m];
//        for (int[] ints : cache) {
//            Arrays.fill(ints, -1);
//        }
//        return dfs(n - 1, m - 1);

        // 翻译为递推
        char[] s = word1.toCharArray();
        char[] t = word2.toCharArray();
        int n = s.length;
        int m = t.length;
        int[] dp = new int[m + 1];
        for (int j = 0; j < m; j++) {
            dp[j + 1] = j + 1;
        }

        for (int i = 0; i < n; i++) {
            int pre = dp[0];
            dp[0] = i + 1;
            for (int j = 0; j < m; j++) {
                int temp = dp[j + 1];
                dp[j + 1] = s[i] == t[j] ? pre :
                        Math.min(Math.min(dp[j + 1], dp[j]) + 1, pre + 2);
                pre = temp;
            }
        }
        return dp[m];

    }

    // dfs(i,j) 表示前i个字符与前j个字符相同的最小部署
//    private int dfs(int i, int j) {
//        if (i < 0) return j + 1;
//        if (j < 0) return i + 1;
//
//        if (cache[i][j] != -1) return cache[i][j];
//
//        if (s[i] == t[j]) return cache[i][j] = dfs(i - 1, j - 1); // 不用删除操作
//
//        return cache[i][j] = Math.min(Math.min(dfs(i, j - 1), dfs(i - 1, j)) + 1, dfs(i - 1, j - 1) + 2);
//
//    }

    public int minimumDeleteSum(String s1, String s2) {
        /*712.两个字符串的最小 ASCII删除和
         * 给定两个字符串s1 和 s2，返回 使两个字符串相等所需删除字符的 ASCII 值的最小和 。
         * */
        char[] s = s1.toCharArray();
        char[] t = s2.toCharArray();
        int m = t.length;
        int[] dp = new int[m + 1];
        for (int j = 0; j < m; j++) {
            dp[j + 1] = t[j] + dp[j];
        }
        for (char c : s) {
            int pre = dp[0];
            dp[0] = c + dp[0];
            for (int j = 0; j < m; j++) {
                int temp = dp[j + 1];
                dp[j + 1] = c == t[j] ? pre : Math.min(
                        Math.min(dp[j + 1] + c, dp[j] + t[j]),
                        pre + c + t[j]);
                pre = temp;
            }
        }
        return dp[m];
    }


//    private char[] chars1, chars2, chars3;
//
//    private int[][] cache;

    public boolean isInterleave(String s1, String s2, String s3) {
        /*97.交错字符串
         * 给定三个字符串 s1、s2、s3，请你帮忙验证 s3 是否是由 s1 和 s2 交错 组成的。
         * 思路：回溯：枚举字符串s3的字符 i，
         * 如果字符c1与s（s1串中）字符相等，递归c1，如果s与字符串s2 中的c2相等递归c2
         * */
//        chars1 = s1.toCharArray();
//        chars2 = s2.toCharArray();
//        chars3 = s3.toCharArray();
//
//        int n = chars1.length;
//        int m = chars2.length;
//        int l = chars3.length;
//        if (n + m != l) return false;
//
//        return dfs(n - 1, m - 1, l - 1);

        // 回溯优化逻辑优化+记忆化搜索：
//        chars1 = s1.toCharArray();
//        chars2 = s2.toCharArray();
//        chars3 = s3.toCharArray();
//
//        int n = chars1.length;
//        int m = chars2.length;
//        int l = chars3.length;
//        if (n + m != l) return false;
//
//        cache = new int[n + 1][m + 1];
//        for (int[] ints : cache) {
//            Arrays.fill(ints, -1);
//        }
//        return dfs(n - 1, m - 1);
        // 翻译为递推
//        char[] chars1 = s1.toCharArray();
//        char[] chars2 = s2.toCharArray();
//        char[] chars3 = s3.toCharArray();
//
//        int n = chars1.length;
//        int m = chars2.length;
//        int l = chars3.length;
//        if (n + m != l) return false;
//
//        boolean[][] dp = new boolean[n + 1][m + 1];
//        dp[0][0] = true;
//        for (int j = 0; j < m; j++) {
//            dp[0][j + 1] = chars2[j] == chars3[j] && dp[0][j];
//        }
//        for (int i = 0; i < n; i++) {
//            dp[i+1][0] = chars1[i] == chars3[i] && dp[i][0];
//            for (int j = 0; j < m; j++) {
//                dp[i + 1][j + 1] = chars1[i] == chars3[i + j + 1] && dp[i][j + 1] ||
//                        chars2[j] == chars3[i + j + 1] && dp[i + 1][j];
//            }
//        }
//        return dp[n][m];
        // 进一步优化空间：
        char[] chars1 = s1.toCharArray();
        char[] chars2 = s2.toCharArray();
        char[] chars3 = s3.toCharArray();

        int n = chars1.length;
        int m = chars2.length;
        int l = chars3.length;
        if (n + m != l) return false;

        boolean[] dp = new boolean[m + 1];
        dp[0] = true;
        for (int j = 0; j < m; j++) {
            dp[j + 1] = chars2[j] == chars3[j] && dp[j];
        }
        for (int i = 0; i < n; i++) {
            dp[0] = chars1[i] == chars3[i] && dp[0];
            for (int j = 0; j < m; j++) {
                dp[j + 1] = chars1[i] == chars3[i + j + 1] && dp[j + 1] ||
                        chars2[j] == chars3[i + j + 1] && dp[j];
            }
        }
        return dp[m];

    }

//    private boolean dfs(int i, int j) {
//        if (i < 0 && j < 0) return true;
//
//        if (cache[i + 1][j + 1] != -1) return cache[i + 1][j + 1] == 1;
//
//        boolean res = (i >= 0 && chars1[i] == chars3[i + j + 1] && dfs(i - 1, j)) ||
//                (j >= 0 && chars2[j] == chars3[i + j + 1] && dfs(i, j - 1));
//        cache[i + 1][j + 1] = res ? 1 : 0;
//        return res;
//    }
//
//    private boolean dfs(int i, int j, int k) {
//        if (k < 0)
//            return true;
//        char c = chars3[k];
//        if (i >= 0 && j >= 0) {
//            if (chars1[i] != c && chars2[j] != c) {
//                return false;
//            }
//            if (chars1[i] == c && chars2[j] == c)
//                return dfs(i - 1, j, k - 1) || dfs(i, j - 1, k - 1);
//            else if (chars1[i] == c) return dfs(i - 1, j, k - 1);
//            else return dfs(i, j - 1, k - 1);
//        } else if (i >= 0) {
//            return chars1[i] == c && dfs(i - 1, j, k - 1);
//        } else if (j >= 0) return chars2[j] == c && dfs(i, j - 1, k - 1);
//        else return false;
//
//
//    }

    public int maxDotProduct(int[] nums1, int[] nums2) {
        /*1458.两个子序列的最大点积
         * 枚举第 i、j 个数，如果 i*j > 0 选，否则 i 不选，或 j 不选的最大值
         * */
//        int n = nums1.length;
//        int m = nums2.length;
//        int[][] cache = new int[n][m];
//        for (int[] ints : cache) {
//            Arrays.fill(ints, Integer.MIN_VALUE);
//        }
//        return dfs(n - 1, m - 1, nums1, nums2, cache);
        //翻译为递推
//        int n = nums1.length;
//        int m = nums2.length;
//        int[][] dp = new int[n + 1][m + 1];
//        for (int[] ints : dp) {
//            Arrays.fill(ints, Integer.MIN_VALUE / 2);
//        }
//        for (int i = 0; i < n; i++) {
//            for (int j = 0; j < m; j++) {
//                dp[i + 1][j + 1] = Math.max(
//                        Math.max(dp[i][j] + nums1[i] * nums2[j], nums1[i] * nums2[j]),
//                        Math.max(dp[i][j + 1], dp[i + 1][j]));
//            }
//        }
//        return dp[n][m];
        // 空间优化
        int m = nums2.length;
        int[] dp = new int[m + 1];

        Arrays.fill(dp, Integer.MIN_VALUE / 2);

        for (int x : nums1) {
            int pre = dp[0];
            for (int j = 0; j < m; j++) {
                int temp = dp[j + 1];
                dp[j + 1] = Math.max(
                        Math.max(pre + x * nums2[j], x * nums2[j]),
                        Math.max(dp[j + 1], dp[j]));
                pre = temp;
            }
        }
        return dp[m];
    }

    // dfs(i,j) 表示前i个数与前j个数的最大点积
//    private int dfs(int i, int j, int[] nums1, int[] nums2, int[][] cache) {
//        if (i < 0 || j < 0) return Integer.MIN_VALUE / 2; // 枚举完了
//        if (cache[i][j] != Integer.MIN_VALUE) return cache[i][j];
//
//        return cache[i][j] = Math.max(
//                Math.max(dfs(i - 1, j - 1, nums1, nums2, cache) + nums1[i] * nums2[j], nums1[i] * nums2[j]),
//                Math.max(dfs(i - 1, j, nums1, nums2, cache), dfs(i, j - 1, nums1, nums2, cache)));
//
//    }

    private char[] s, t;
    private int[][] cache;

    public String shortestCommonSupersequence(String str1, String str2) {
        /*1092. 最短公共超序列
         * 给你两个字符串 str1 和 str2，返回同时以 str1 和 str2 作为 子序列 的最短字符串。
         * 如果答案不止一个，则可以返回满足条件的 任意一个 答案。
         * 也就是要生成一个最短字符串，满足 str1与str2都是它的子序列
         *
         * */
//        s = str1.toCharArray();
//        t = str2.toCharArray();
//        int n = s.length;
//        int m = t.length;
//        String[][] cache = new String[n][m];
//
//        return dfs(n - 1, m - 1, cache);
        // 翻译为递推

//        char[] s = str1.toCharArray();
//        char[] t = str2.toCharArray();
//        int m = t.length;
//        String[] dp = new String[m + 1];
//        dp[0] = "";
//        for (int i = 0; i < m; i++) {
//            dp[i + 1] = dp[i] + t[i];
//        }
//
//        for (char c : s) {
//            String pre = dp[0];
//            dp[0] = dp[0] + c;
//            for (int j = 0; j < m; j++) {
//                String temp = dp[j + 1];
//                dp[j + 1] = c == t[j] ? pre + c :
//                        dp[j + 1].length() < dp[j].length() ? dp[j + 1] + c : dp[j] + t[j];
//                pre = temp;
//            }
//        }
//        return dp[m];

        // 递归长度与构造答案分开
        s = str1.toCharArray();
        t = str2.toCharArray();
        int n = s.length;
        int m = t.length;
        cache = new int[n][m];
        for (int[] ints : cache) {
            Arrays.fill(ints, -1);
        }
        return makeAns(n - 1, m - 1);


    }

    private String makeAns(int i, int j) {
        if (i < 0) return new String(t, 0, j + 1);
        if (j < 0) return new String(s, 0, i + 1);
        if (s[i] == t[j]) {
            return makeAns(i - 1, j - 1) + s[i];
        }
        if (dfs(i, j) == dfs(i - 1, j) + 1) return makeAns(i - 1, j) + s[i];
        else return makeAns(i, j - 1) + t[j];
    }

    // dfs(i,j) 表示前 i 个字符与前 j 个字符的最短超序列
    private int dfs(int i, int j) {
        if (i < 0) return j + 1;
        if (j < 0) return i + 1;
        if (cache[i][j] != -1) return cache[i][j];

        if (s[i] == t[j]) return cache[i][j] = dfs(i - 1, j - 1) + 1;
        else return cache[i][j] = Math.min(dfs(i - 1, j) + 1, dfs(i, j - 1) + 1);

    }

    // dfs(i,j) 表示前i字符与前j个字符的最短公共超序列序列的长度
//    private String dfs(int i, int j, String[][] cache) {
//        if (i < 0 && j < 0) {
//            return "";
//        }
//        if (i < 0) {
//            return new String(t, 0, j + 1);
//        }
//        if (j < 0) return new String(s, 0, i + 1);
//
//        if (cache[i][j] != null) return cache[i][j];
//
//        if (s[i] == t[j]) return dfs(i - 1, j - 1, cache) + s[i];
//        String str1 = dfs(i - 1, j, cache) + s[i];
//        String str2 = dfs(i, j - 1, cache) + t[j];
//        return cache[i][j] = str1.length() > str2.length() ? str2 : str1;
//    }

    public int lengthOfLIS(int[] nums) {
        /*300.最长递增子序列
         * 给你一个整数数组 nums ，找到其中最长严格递增子序列的长度。
         * 思路：子序列也是子集的一种，考虑枚举选不选与答案选哪个。
         * 因为严格递增，如果枚举选不选，需要维护上一个元素的下标，
         * 枚举答案选哪个，只用循环找到前一个数，递归参数只有一个。
         * */
//        int n = nums.length;
//        int[] cache = new int[n];
//        int ans = 0;
//        for (int i = 0; i < nums.length; i++) {
//            ans = Math.max(dfs(i, nums, cache), ans);
//        }
//        return ans;
        // 翻译为递推
//        int n = nums.length;
//        int[] dp = new int[n];
//        int ans = 0;
//        for (int i = 0; i < nums.length; i++) {
//            for (int j = i - 1; j >= 0; j--) {
//                if (nums[j] < nums[i]) dp[i] = Math.max(dp[j], dp[i]);
//            }
//            ans = Math.max(++dp[i], ans);
//        }
//
//        return ans;

        // 贪心加二分

//        List<Integer> g = new ArrayList<>(nums.length); // g[i] 表示 长为 i+1 的上升子序列的末尾元素的最小值
//        for (int num : nums) {
//            int j = binarySearch(num, g);
//            if (j == g.size()) {
//                // 元素不存在加入集合
//                g.add(num);
//            } else {
//                g.set(j, num);
//            }
//        }
//        return g.size();
        //原地修改
        int len = 0;
        for (int num : nums) {
            int j = binarySearch(num + 1, nums, len);
            nums[j] = num;
            if (j == len)
                len++;
        }
        return len;
    }

    private int binarySearch(int target, int[] nums, int right) {
        int left = 0;
        //左闭右开区间写法
        while (left < right) {
            int mid = (left + right) >>> 1;
            if (nums[mid] < target) left = mid + 1;
            else right = mid;
        }
        return left;
    }

    private int binarySearch(int num, List<Integer> g) {
        int left = -1;
        int right = g.size();

        while (left + 1 < right) {
            int mid = (left + right) >>> 1;
            if (g.get(mid) >= num) {
                right = mid;
            } else {
                left = mid;
            }

        }
        return left + 1;
    }

    // dfs(i) 表示以nums[i] 结尾的最长递增子序列长度
//    private int dfs(int i, int[] nums, int[] cache) {
//        if (cache[i] > 0) return cache[i];
//        int res = 0;
//        for (int j = i - 1; j >= 0; j--) {
//            if (nums[j] < nums[i]) {
//                // 符合要求加入
//                res = Math.max(dfs(j, nums, cache), res);
//            }
//
//        }
//        return cache[i] = (res + 1);
//    }

    public int minimumOperations(List<Integer> nums) {
        /*2826.将三个组排序
        整数数组 nums 。nums 的每个元素是 1，2 或 3。
        在每次操作中，你可以删除 nums 中的一个元素。
        返回使 nums 成为 非递减 顺序所需操作数的最小值。
        思路：构造最长非递减的子序列 a，用nums的长度减去子序列 a 的长度就是最小操作数
        * */
//        int len = nums.size();
//        int[] ints = new int[len];
//        for (int i = 0; i < len; i++) {
//            ints[i] = nums.get(i);
//        }
//        return len - lengthOfLIS(ints);
        // DP 枚举当前元素选不选
        // dp[i][j] 表示前i个元素的最大值小于等j的最长子序列的长度，x 表示当前元素
        // x>j 不能选当前元素 dp[i][j] = dp[i-1][j]
//        x<=j 选 dp[i-1][x] +1 和 dp[i-1][j] 的较大值
//        int len = nums.size();
//        int[][] dp = new int[len + 1][4];
//        for (int i = 0; i < len; i++) {
//            int x = nums.get(i);
//            for (int j = 1; j < 4; j++) {
//                if (x > j) dp[i + 1][j] = dp[i][j];
//                else dp[i + 1][j] = Math.max(dp[i][x] + 1, dp[i][j]);
//            }
//        }
//        return len - dp[len][3];
        int[] dp = new int[4];
        for (int x : nums) {
            dp[x]++;
            dp[2] = Math.max(dp[1], dp[2]);
            dp[3] = Math.max(dp[2], dp[3]);
        }
        return nums.size() - dp[3];
    }

    public int[] longestObstacleCourseAtEachPosition(int[] obstacles) {
        /*你打算构建一些障碍赛跑路线。
        给你一个 下标从 0 开始 的整数数组 obstacles ，数组长度为 n ，其中 obstacles[i] 表示第 i 个障碍的高度
        对于每个介于 0 和 n - 1 之间（包含 0 和 n - 1）的下标  i ，
        在满足下述条件的前提下，请你找出 obstacles 能构成的最长障碍路线的长度：
        你可以选择下标介于 0 到 i 之间（包含 0 和 i）的任意个障碍。
        在这条路线中，必须包含第 i 个障碍。
        你必须按障碍在 obstacles 中的 出现顺序 布置这些障碍。
        除第一个障碍外，路线中每个障碍的高度都必须和前一个障碍 相同 或者 更高 (非递减序列)。
        返回长度为 n 的答案数组 ans ，其中 ans[i] 是上面所述的下标 i 对应的最长障碍赛跑路线的长度。
        * 求i个数构成的最长非递减序列
        * dp[i][j] 表示i个数最大值不超过j的最长非递减序列, 该序列必须包含 nums[i],也就是以 nums[i] 为最大值
        nums[i] =x
        dp[i][x] = dp[i-1][x] +1 dp[i-1][1~x-1]

        * */
//        int len = obstacles.length;
//        int mx = 0;
//        for (int obstacle : obstacles) {
//            mx = Math.max(obstacle, mx);
//        }
//        int[] ans = new int[len];
//        int[][] dp = new int[len + 1][mx + 1];
//        for (int k = 0; k < len; k++) {
//            for (int i = 0; i <= k; i++) {
//                int x = obstacles[i];
//                for (int j = x; j > 0; j--) {
//                    dp[i + 1][x] = Math.max(dp[i][j] + 1, dp[i + 1][x]);
//                }
//            }
//            ans[k] = dp[k + 1][obstacles[k]];
//        }
//
//
//        return ans;
        // 回溯：枚举以第 i 个元素结尾的非递减子序列的最长长度
//        int len = obstacles.length;
//        int[] ans = new int[len];
//        int[] cache = new int[len];
//        for (int i = 0; i < len; i++) {
//            ans[i] = dfs(i, obstacles, cache);
//        }
//        return ans;
        // 动规 dp[i] 表示以第i个元素结尾的非递减子序列的最长长度
        // g[i] 表示长度为 i+1 的非递减子序列的尾元素最小值
//        int len = obstacles.length;
//        int[] dp = new int[len];
//
//        for (int i = 0; i < len; i++) {
//            for (int j = i - 1; j >= 0; j--) {
//                if (obstacles[j] <= obstacles[i]) dp[i] = Math.max(dp[j], dp[i]);
//            }
//            dp[i]++;
//        }
//        return dp;
        //贪心+二分
        int len = obstacles.length;
        int[] ans = new int[len];
        List<Integer> g = new ArrayList<>();
        int i = 0;
        for (int x : obstacles) {
            int j = binary(x, g);
            if (j == g.size()) {
                g.add(x);
                ans[i++] = g.size();
            } else {
                g.set(j, x);
                ans[i++] = j + 1;
            }
        }
        return ans;

    }

//    private int binary(int x, List<Integer> g) {
//        int left = -1;
//        int right = g.size();
//        while (left + 1 < right) {
//            int mid = (left + right) >>> 1;
//            if (g.get(mid) > x) right = mid;
//            else left = mid;
//        }
//        return right;
//    }

    // 返回以 i 结尾的非递减子序列的最长长度
//    private int dfs(int i, int[] obstacles, int[] cache) {
//        if (i < 0) return 0;
//        if (cache[i] != 0) return cache[i];
//        int res = 0;
//        for (int j = i - 1; j >= 0; j--) {
//            if (obstacles[j] <= obstacles[i]) res = Math.max(dfs(j, obstacles, cache), res);
//        }
//        return cache[i] = res + 1;
//    }


    public int minimumMountainRemovals(int[] nums) {
        /*1671.得到山形数组的最少删除次数
        * 我们定义 arr 是 山形数组 当且仅当它满足：
        * arr.length >= 3
        * 存在某个下标 i （从 0 开始） 满足 0 < i < arr.length - 1 且：
        arr[0] < arr[1] < ... < arr[i - 1] < arr[i]
        arr[i] > arr[i + 1] > ... > arr[arr.length - 1]
        给你整数数组 nums ，请你返回将 nums 变成 山形状数组 的最少 删除次数。
        * 思路：枚举 nums[i] 其左边的元素构成严格递增子序列的最长长度 len1
        * 其右边的元素构成严格递减的子序列的最长长度 len2
        * ans = nums.length - len1 -len2 +1;
        * */
//        int len = nums.length;
//        int ans = len;
//        int[] cache1 = new int[len];
//        int[] cache2 = new int[len];
//        for (int i = 1; i < len - 1; i++) {
//            int len1 = LIS(i, nums, cache1);
//            int len2 = LDS(i, nums, cache2);
//            if (len1 >= 2 && len2 >= 2)
//                ans = Math.min(ans, len - len1 - len2 + 1);
//        }
//        return ans;
        //翻译为递推
//        int len = nums.length;
//        int ans = len;
//        int[] f1 = new int[len];
//        int[] f2 = new int[len];
//        for (int i = len - 1; i >= 0; i--) {
//            for (int j = i + 1; j < len; j++) {
//                if (nums[i] > nums[j]) f1[i] = Math.max(f1[i], f1[j]);
//            }
//            f1[i]++;
//        }
//        for (int i = 0; i < len; i++) {
//            for (int j = i - 1; j >= 0; j--) {
//                if (nums[i] > nums[j]) f2[i] = Math.max(f2[i], f2[j]);
//            }
//            f2[i]++;
//            if (f1[i] >= 2 && f2[i] >= 2)
//                ans = Math.min(ans, len - f1[i] - f2[i] + 1);
//        }
//
//        return ans;
        //贪心+二分
        int len = nums.length;
        int[] suf = new int[len];
        List<Integer> g = new ArrayList<>();
        for (int i = len - 1; i > 0; i--) {
            int j = binary(nums[i], g);
            if (j == g.size())
                g.add(nums[i]);
            else
                g.set(j, nums[i]);
            suf[i] = j + 1;
        }
        int ans = len;
        g.clear();
        for (int i = 1; i < len - 1; i++) {
            int x = nums[i];
            int j = binary(x, g);
            if (j == g.size())
                g.add(x);
            else g.set(j, x);
            int pre = j + 1;
            if (pre >= 2 && suf[i] >= 2)
                ans = Math.min(ans, len - pre - suf[i] + 1);
        }
        return ans;

    }

//    private int binary(int num, List<Integer> g) {
//        int left = -1;
//        int right = g.size();
//        while (left + 1 < right) {
//            int mid = (left + right) >>> 1;
//            if (g.get(mid) >= num) right = mid;
//            else left = mid;
//        }
//        return right;
//    }

//    //LDS(i) 表示后i个数构成的LDS的最长长度
//    private int LDS(int i, int[] nums, int[] cache) {
//        if (cache[i] != 0) return cache[i];
//        int res = 0;
//        for (int j = i + 1; j < nums.length; j++) {
//            if (nums[i] > nums[j]) res = Math.max(LDS(j, nums, cache), res);
//        }
//        return cache[i] = res + 1;
//    }
//
//    // LIS(i) 前i个数构成的 LIS 的最长长度
//    private int LIS(int i, int[] nums, int[] cache) {
//        if (cache[i] != 0) return cache[i];
//        int res = 0;
//        for (int j = i - 1; j >= 0; j--) {
//            if (nums[i] > nums[j]) res = Math.max(LIS(j, nums, cache), res);
//        }
//        return cache[i] = res + 1;
//    }

    public int kIncreasing(int[] arr, int k) {
        /*2111. 使数组 K 递增的最少操作次数
        给你一个下标从 0 开始包含 n 个正整数的数组 arr ，和一个正整数 k 。
        如果对于每个满足 k <= i <= n-1 的下标 i ，都有 arr[i-k] <= arr[i] ，那么我们称 arr 是 K 递增 的。
        就是每隔k-1个数,非递减
        每一次 操作 中，你可以选择一个下标 i 并将 arr[i] 改成任意 正整数。
        请你返回对于给定的 k ，使数组变成 K 递增的 最少操作次数 。
        分为k组，求每组的最长的非递减子序列
        ans = arr.length - 每组的最长非递减子序列长度
        *
        * */
//        int len = arr.length;
//        int[] lens = new int[k];//lens[i] 表示第i组的最长非递减子序列长度
//        List<Integer> g = new ArrayList<>();
//        for (int i = 0; i < k; i++) {
//            g.clear();
//            for (int j = i; j < len; j = k + j) {
//                int x = arr[j];
//                int m = binary(x, g);
//                if (m == g.size())
//                    g.add(x);
//                else g.set(m, x);
//            }
//            lens[i] = g.size();
//        }
//        int ans = len;
//        for (int i : lens) {
//            ans = ans - i;
//        }
//        return ans;
        // 动态规划 dp[i] 表示以arr[i]元素结尾的最长非递减序列长度
        //        for (int i = 0; i < len; i++) {
//            for (int j = i - 1; j >= 0; j--) {
//                if (obstacles[j] <= obstacles[i]) dp[i] = Math.max(dp[j], dp[i]);
//            }
//            dp[i]++;
//        }
        int len = arr.length;
        int[] lens = new int[k];
        for (int i = 0; i < k; i++) {
            int[] dp = new int[len];
            int mx = 0;
            for (int j = i; j < len; j = j + k) {
                for (int m = j - k; m >= 0; m = m - k) {
                    if (arr[m] <= arr[j]) dp[j] = Math.max(dp[m], dp[j]);
                }
                dp[j]++;

            }

            lens[i] = mx;
        }
        int ans = len;
        for (int i : lens) {
            ans = ans - i;
        }
        return ans;
    }

    //找到第一个大于x的位置
//    private int binary(int x, List<Integer> g) {
//        int left = -1;
//        int right = g.size();
//        while (left + 1 < right) {
//            int mid = (left + right) >>> 1;
//            if (g.get(mid) <= x) left = mid;
//            else right = mid;
//        }
//        return right;
//    }

    public int maxEnvelopes(int[][] envelopes) {
        /*354.俄罗斯套娃信封问题
         * 给你一个二维整数数组 envelopes ，其中 envelopes[i] = [wi, hi] ，表示第 i 个信封的宽度和高度。
         * 当另一个信封的宽度和高度都比这个信封大的时候，这个信封就可以放进另一个信封里，如同俄罗斯套娃一样。
         * 请计算 最多能有多少个 信封能组成一组“俄罗斯套娃”信封（即可以把一个信封放到另一个信封里面）。
         * 注意：不允许旋转信封
         * 思路：先对envelopes按照宽w来排序，只需要求关于h的LIS了，
         * 枚举答案选哪个，只能选比当前元素大的元素，递归选
         * */
        int len = envelopes.length;
        // 双关键字排序：宽度降序，高度升序
        Arrays.sort(envelopes, (a, b) -> {
            if (a[0] == b[0]) {
                return a[1] - b[1];
            }
            return b[0] - a[0];
        });

//        int ans = 0;
//        int[] cache = new int[len];
//        for (int i = 0; i < len; i++) {
//            ans = Math.max(dfs(i, envelopes, cache), ans);
//        }
//        return ans;
        //翻译为递推
//        int[] dp = new int[len];
//        int ans = 0;
//        for (int i = 0; i < len; i++) {
//            for (int j = i - 1; j >= 0; j--) {
//                if (envelopes[j][0] > envelopes[i][0] && envelopes[j][1] > envelopes[i][1])
//                    dp[i] = Math.max(dp[j], dp[i]);
//            }
//            ans = Math.max(ans, ++dp[i]);
//        }
//
//        return ans;
        // 贪心+二分
        List<Integer> g = new ArrayList<>();
        for (int i = len - 1; i >= 0; i--) {
            int x = envelopes[i][1];
            int j = binary(x, g);
            if (j == g.size())
                g.add(x);
            else
                g.set(j, x);
        }
        return g.size();
    }

    private int binary(int x, List<Integer> g) {
        int left = -1;
        int right = g.size();
        while (left + 1 < right) {
            int mid = (left + right) >>> 1;
            if (g.get(mid) < x) left = mid;
            else right = mid;
        }
        return right;
    }
    // dfs(i) 表示前i个元素的最大套娃数
//    private int dfs(int i, int[][] envelopes, int[] cache) {
//        if (i < 0) return 0;
//        if (cache[i] != 0) return cache[i];
//        int w = envelopes[i][0];
//        int h = envelopes[i][1];
//        int res = 0;
//        for (int j = i - 1; j >= 0; j--) {
//            // 枚举选哪个
//            if (envelopes[j][0] > w && envelopes[j][1] > h) res = Math.max(dfs(j, envelopes, cache), res);
//        }
//        return cache[i] = res + 1;
//    }
    /*
    *     public int maxEnvelopes(int[][] envelopes) {
        int len = envelopes.length;
        int[] dp = new int[len];
        for (int i = 0; i < len; i++) {
            for (int j = i - 1; j >= 0; j--) {
                if (check(i, j, envelopes)) dp[i] = Math.max(dp[j], dp[i]);
            }
            dp[i]++;
        }
        return dp[len - 1];
    }
      private boolean check(int i, int j, int[][] envelopes) {
        return envelopes[i][0] < envelopes[j][0] && envelopes[i][1] < envelopes[j][1];
    }
    * */

    public int bestTeamScore(int[] scores, int[] ages) {
        /*1626.无矛盾的最佳球队
        * 假设你是球队的经理。对于即将到来的锦标赛，你想组合一支总体得分最高的球队。球队的得分是球队中所有球员的分数总和
        * 然而，球队中的矛盾会限制球员的发挥，所以必须选出一支 没有矛盾 的球队。
        * 如果一名年龄较小球员的分数 严格大于 一名年龄较大的球员，则存在矛盾。同龄球员之间不会发生矛盾。
        * 给你两个列表 scores 和 ages，其中每组 scores[i] 和 ages[i] 表示第 i 名球员的分数和年龄。
        * 请你返回 所有可能的无矛盾球队中得分最高那支的分数 。
         * */
    }

}

