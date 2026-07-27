package huawei.d20260423.p2_shortest_path;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Arrays;
import java.util.StringTokenizer;

public class Main {
    /*
     * 魔导传输网
     * 题目描述
     * 给定一个由 N 个节点和 M 条双向传输通道组成的网络。
     * 节点编号为：0 ～ N-1
     * 每条传输通道连接两个节点，并具有一个正整数传输延迟。同一对节点之间可能存在多条传输延迟不同的通道。
     * 现给出 Q 组查询。对于每组查询，计算两个指定节点之间的最小传输总延迟。
     * 如果两个节点之间不存在任何可达路径，输出 0。
     * 思路：弗洛伊德算法的实现
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
        int n = fs.nextInt(), m = fs.nextInt();
        final int INF = Integer.MAX_VALUE / 2;
        int[][] dis = new int[n][n];
        for (int i = 0; i < n; i++) {
            Arrays.fill(dis[i], INF);
            dis[i][i] = 0;
        }
        for (int i = 0; i < m; i++) {
            int x = fs.nextInt();
            int y = fs.nextInt();
            int w = fs.nextInt();
            dis[x][y] = Math.min(dis[x][y], w); // 重边的情况，记录最小值
            dis[y][x] = Math.min(dis[y][x], w); // 无向
        }
        for (int k = 0; k < n; k++) {
            // 这里表示依次以 k 节点为中间节点
            for (int i = 0; i < n; i++) {
                if (dis[i][k] == INF) continue; // 由下面的式子可以看到，当dis[i][k]为无穷时，距离不变

                for (int j = 0; j < n; j++) {
                    if (dis[k][j] == INF) {
                        continue;
                    }
                    dis[i][j] = Math.min(dis[i][j], dis[i][k] + dis[k][j]);
                }
            }
        }
        StringBuilder sb = new StringBuilder();
        int count = fs.nextInt();
        while (count-- > 0) {
            int x = fs.nextInt(), y = fs.nextInt();
            sb.append(dis[x][y] == INF ? 0 : dis[x][y]).append('\n');
        }
        System.out.print(sb);
    }
}
