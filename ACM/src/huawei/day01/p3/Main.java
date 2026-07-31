package huawei.day01.p3;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Arrays;
import java.util.Map;
import java.util.StringTokenizer;

public class Main {
    /*
    * 大模型训练显存优化算法
    * 题目描述
    *
    * 在大模型训练过程中，NPU 显存不足。当前有 n 个候选张量，第 i 个张量占用的显存空间为 si
    * 对于每个张量，可以采用以下两种显存优化方式之一：
    * Swap：将张量换出到主存，释放 si 的显存，代价为 ai
    * 重计算：丢弃张量，在需要时重新计算，释放 si 的显存，代价为 bi 对于每个张量：

可以选择 Swap；
可以选择重计算；
也可以不对它进行优化；
同一个张量最多选择一种优化方式。

请选出若干张量进行优化，使释放的显存总量不少于目标值 m，同时让总代价最小。

如果无法释放至少 m 的显存，输出：

error
输入格式

第一行：

m

表示至少需要释放的显存空间。

第二行：

n

表示候选张量数量。

第三行包含 n 个整数：

s1 s2 ... sn

表示各张量占用的显存空间。

第四行包含 n 个整数：

a1 a2 ... an

表示对各张量执行 Swap 的代价。

第五行包含 n 个整数：

b1 b2 ... bn

表示对各张量执行重计算的代价。

数据范围
0<m<10000
0<n<10000
1≤si,ai,bi≤10000
输出格式

如果存在可行方案，输出一个整数，表示最小总代价。

否则输出：

error
样例输入
10
5
3 4 5 6 7
1 2 3 5 5
2 3 4 5 6
样例输出
6
样例说明

可以选择：

第 1 个张量：使用 Swap，释放 3，代价 1；
第 5 个张量：使用 Swap，释放 7，代价 5。

总释放空间：

3+7=10

总代价：

1+5=6

满足释放空间不少于 10，且最小总代价为 6。
    *
    * 思路：0-1背包的变形：
    * 对于每个张量有选或不选，选代表选择优化，优化方式有两种；不选则跳过
    * 求 target>=m 时，价值最小
    * */
    static class FastScanner {
        private final BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
        private StringTokenizer st;

        String next() throws IOException {
            while (st == null || !st.hasMoreTokens()) {
                String line = br.readLine();
                if (line == null) return null;
                st = new StringTokenizer(line);
            }
            return st.nextToken();
        }

        int nextInt() throws IOException {
            return Integer.parseInt(next());
        }
    }

    private static final int POS_INF = Integer.MAX_VALUE / 2;
    private static int[][] memo;

    public static void main(String[] args) throws IOException {
        FastScanner fs = new FastScanner();
        int target = fs.nextInt();
        int n = fs.nextInt();
        int[] s = new int[n];
        int[] a = new int[n];
        for (int i = 0; i < n; i++) {
            s[i] = fs.nextInt();
        }
        for (int i = 0; i < n; i++) {
            a[i] = fs.nextInt();
        }
        for (int i = 0; i < n; i++) {
            a[i] = Math.min(fs.nextInt(), a[i]);
        }
//        memo = new int[n][target + 1];
//        for (int[] ints : memo) {
//            Arrays.fill(ints, -1);
//        }
//        int ans = dfs(n - 1, target, s, a);
//        System.out.println(ans == POS_INF ? "error" : ans);
        // 翻译为递推
        int[] dp = new int[target + 1];
        Arrays.fill(dp, POS_INF);
        dp[0] = 0;
        for (int i = 0; i < n; i++) {
            for (int j = target; j >= 0; j--) {
                dp[j] = Math.min(dp[j], dp[Math.max(0, j - s[i])] + a[i]);
            }
        }

        int ans = dp[target];
        System.out.println(ans == POS_INF ? "error" : ans);

    }

    /*
     * dfs(i, remain) 表示从编号 0～i 的张量中选择，使得还需要释放至少 remain 的空间时，所需的最小代价。
     * 当前操作：枚举第 i 个张量优化或者不优化
     * 子问题：前 i 个张量的最小代价
     * 下一个子问题：优化 前 i 个张量和为 target - 当前张量优化的最优解的最小代价
     * 不优化 前 i 个张量和为 target  的最小代价
     * */
//    private static int dfs(int i, int target, int[] s, int[] a) {
//        if (target <= 0) {
//            return 0;
//        }
//
//        if (i < 0) {
//            return POS_INF;
//        }
//        if (memo[i][target] != -1) return memo[i][target];
//        // 不选
//        int res1 = dfs(i - 1, target, s, a);
//        // 选
//        int res2 = dfs(i - 1, target - s[i], s, a) + a[i];
//        return  memo[i][target] = Math.min(res1, res2);
//
//
//    }

}
