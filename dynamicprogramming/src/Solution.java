import javax.management.modelmbean.ModelMBean;
import javax.xml.stream.FactoryConfigurationError;
import java.util.*;

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
}
