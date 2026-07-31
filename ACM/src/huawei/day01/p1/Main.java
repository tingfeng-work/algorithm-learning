package huawei.day01.p1;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Arrays;
import java.util.StringTokenizer;

public class Main {
    /*
     * 第 2 题：最大能量路径
     * 自动驾驶系统需要从图像中识别一条由左向右延伸的车道线。
     * 现有：
     * 一幅大小为 H×W 的图像矩阵 I；
     * 一个大小为 K×K 的策略矩阵 S，其中 K 为奇数。
     * 首先，需要计算图像中每个位置的能量值。
     * p= K/2 向下取整
     * 位置 (r,c) 的能量为：E[r][c]=∑∑I[r+i−p][c+j−p]×S[i][j]
     * 如果图像坐标超出边界，则对应的图像信号值按 0 处理，也就是零填充。
     * 路径必须：
     * 从图像第一列的任意位置出发；
     * 到达图像最后一列；
     * 每次只能向以下三个方向移动：右上、正右、右下
     * 移动不能超出图像边界。
     * 路径能量为路径上所有位置能量值之和。
     * 请计算合法路径能够获得的最大能量。
     * 思路：先计算出每个位置的能量图，再从能量图触发求合法路径的最大能量
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
    public static void main(String[] args) throws IOException {
        FastScanner fs = new FastScanner();
        int h = fs.nextInt(), w = fs.nextInt(), k = fs.nextInt();
        fs.nextInt();
        int[][] I = new int[h][w];
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                I[i][j] = fs.nextInt();
            }
        }
        int[][] S = new int[k][k];
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < k; j++) {
                S[i][j] = fs.nextInt();
            }
        }
        // 构造能量矩阵
        // 策略矩阵与该位置周边信号值的乘积和包括该位置，超出边界的按0填充
        int[][] E = new int[h][w];
        int p = k / 2;
        for (int r = 0; r < h; r++) {
            for (int c = 0; c < w; c++) {
                for (int i = 0; i < k; i++) {
                    for (int j = 0; j < k; j++) {
                        if ((r + i - p >= 0) && (r + i - p) < h && (c + j - p) >= 0 && (c + j - p) < w)
                            E[r][c] += I[r + i - p][c + j - p] * S[i][j];
                    }
                }
            }
        }
        // 求合法路径的最大能量
        // 思路：从最后一列枚举以当前元素为终点的路径的最大能量，状态来自三个方向转移来取最大值
        int ans = Integer.MIN_VALUE;
        int[][] cache = new int[h][w];
        for (int[] ints : cache) {
            Arrays.fill(ints, -1);
        }
        for (int i = 0; i < h; i++) {
            ans = Math.max(dfs(i, w - 1, E, cache), ans);
        }
        System.out.printf("%.1f%n", (double)ans);
    }

    private static int dfs(int i, int j, int[][] e, int[][] cache) {
        // 非法路径
        if (i < 0 || i >= e.length) return 0;
        if (j < 0) return 0;
        if (cache[i][j] != -1)
            return cache[i][j];
        int value = e[i][j];
        int res1 = dfs(i + 1, j - 1, e, cache);
        int res2 = dfs(i, j - 1, e, cache);
        int res3 = dfs(i - 1, j - 1, e, cache);
        return cache[i][j] = Math.max(res1, Math.max(res2, res3)) + value;
    }
}
