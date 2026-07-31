package huawei.day01.p0;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.DoubleBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.StringTokenizer;

public class Main {
    /*
     * 题目：模型推理量化加速优化问题
     * 描述：一个大模型共有 N 层，每一层都可以选择若干种不同的量化方案。
     * 每种量化方案包含：
     * 量化位数，例如 8bit、16bit
     * 精度损失 loss
     * 显存占用 memory
     * 每一层必须且只能选择一种量化方案
     * 要求所有层的累计精度损失不超过给定阈值 T，请计算模型所需的最小显存占用。
     * 输入格式：第一行包含：N T
     * N：模型层数    T：允许的最大累计精度损失
     * 接下来输入 N 行，每行描述一层的量化方案：
     * K bit1 loss1 memory1 bit2 loss2 memory2 ... bitK lossK memoryK
     * K：当前层可选方案数量
     * bit：量化方案名称
     * loss：选择该方案造成的精度损失
     * memory：选择该方案需要的显存
     * 输出格式
     * 输出满足精度损失限制时的最小总显存，保留两位小数。
     * 如果不存在可行方案，输出： -1
     * 3 0.3
     * 2 8bit 0.2 100.0 16bit 0.1 200.0
     * 2 8bit 0.3 150.0 16bit 0.1 300.0
     * 1 8bit 0.1 150.0
     * 输出：650.00
     * 思路：
     * 0-1 背包问题 更正分组背包问题
     * 精度损失就是体积，显存相当于价值
     * 求总体积小于 T 时的最小价值
     * */
    static class FastScanner {
        private final BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
        private StringTokenizer st;

        String next() throws IOException {
            while (st == null || !st.hasMoreTokens()) {
                String line = br.readLine();
                if (line == null)
                    return null;
                st = new StringTokenizer(line);
            }
            return st.nextToken();
        }

        double nextDouble() throws IOException {
            return Double.parseDouble(next());
        }

        int nextInt() throws IOException {
            return Integer.parseInt(next());
        }
    }

    private static final double EPS = 1e-9;


    public static void main(String[] args) throws IOException {
        FastScanner fs = new FastScanner();
        int n = fs.nextInt();
        int t = (int) Math.round(fs.nextDouble() * 100);
        List<Integer>[] lossNums = new ArrayList[n];
        Arrays.setAll(lossNums, e -> new ArrayList<>());
        List<Double>[] memNums = new ArrayList[n];
        Arrays.setAll(memNums, e -> new ArrayList<>());
        for (int i = 0; i < n; i++) {
            int k = fs.nextInt();
            for (int j = 0; j < k; j++) {
                fs.next();
                int loss = (int) Math.round(fs.nextDouble() * 100);
                double mem = fs.nextDouble();
                lossNums[i].add(loss);
                memNums[i].add(mem);
            }
        }
//        double ans = dfs(n - 1, t, lossNums, memNums);
//        if (Double.isInfinite(ans)) {
//            System.out.println("-1");
//        } else {
//            System.out.printf("%.2f%n", ans);
//        }
        // 翻译为递推
        double[][] dp = new double[n + 1][t + 1]; // dp[i][j] 表示前 i 层损失度和为 j 的最小价值
        for (double[] row : dp) {
            Arrays.fill(row, Double.POSITIVE_INFINITY);
        }

        dp[0][0] = 0;

        for (int i = 0; i < n; i++) {
            List<Integer> cap = lossNums[i];
            int k = cap.size();
            for (int j = 0; j <= t; j++) {
                for (int l = 0; l < k; l++) {
                    int loss = cap.get(l);
                    if (j >= loss && !Double.isInfinite(dp[i][j - loss])) {
                        dp[i + 1][j] = Math.min(
                                dp[i + 1][j],
                                dp[i][j - loss] + memNums[i].get(l)
                        );
                    }
                }
            }
        }
        double ans = Double.POSITIVE_INFINITY;
        for (int j = 0; j <= t; j++) {
            ans = Math.min(ans, dp[n][j]);
        }
        if (Double.isInfinite(ans)) {
            System.out.println("-1");
        } else {
            System.out.printf("%.2f%n", ans);
        }
    }

    /*
     * 原问题：n 层模型输出精度损失小于 t 时的最小显存
     * 当前操作：第i层的第j个方案选不选
     * 子问题：选：前 i 层的精度损失小于 t - loss[i][j] 下的最小显存 + 当前层的显存
     * 不选：：下一个方案
     *
     * */
    private static double dfs(int i, double target, List<Double>[] lossNums, List<Double>[] memNums) {
        if (target < -EPS) {
            return Double.POSITIVE_INFINITY;
        }

        if (i < 0) {
            return 0;
        }
        List<Double> nums = lossNums[i];
        List<Double> values = memNums[i];
        int k = lossNums[i].size();
        double res = Double.POSITIVE_INFINITY;
        for (int j = 0; j < k; j++) {
            // 当前方案选
            double next = dfs(i - 1, target - nums.get(j), lossNums, memNums);
            if (!Double.isInfinite(next)) {
                res = Math.min(res, next + values.get(j));
            }
        }
        return res;
    }
}
