package huawei.day03.p0;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.StringTokenizer;

public class Main {
    /*
     * 基于决策树预判资源调配优先级
     *
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

        double nextDouble() throws IOException {
            return Double.parseDouble(next());
        }
    }

    public static void main(String[] args) throws IOException {
        FastScanner fs = new FastScanner();
        // 读取数据
        int f = fs.nextInt(), m = fs.nextInt(), n = fs.nextInt();
        double[][] T = new double[m][5];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < 5; j++) {
                T[i][j] = fs.nextDouble();
            }
        }
        double[][] samples = new double[n][f];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < f; j++) {
                samples[i][j] = fs.nextDouble();
            }
        }
        // 推理决策
        int[] ans = new int[n];

        for (int i = 0; i < n; i++) {
            int row = 0;
            double[] features = samples[i];
            while (true) {
                // 当前节点是叶子节点
                if ((int) T[row][4] != -1) {
                    ans[i] = (int) T[row][4];
                    break;
                }

                int featureIndex = (int) T[row][0];
                double threshold = T[row][1];

                if (features[featureIndex] <= threshold) {
                    row = (int) T[row][2];
                } else {
                    row = (int) T[row][3];
                }
            }

        }
        for (int an : ans) {
            System.out.println(an);
        }
    }
}
