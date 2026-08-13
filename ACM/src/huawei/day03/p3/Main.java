package huawei.day03.p3;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.StringTokenizer;

public class Main {
    /*
     * 基于逻辑回归的意图分类器
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
        // 数据读取
        int n = fs.nextInt(), m = fs.nextInt();
        String[] features = new String[n];
        int[] labels = new int[n];
        for (int i = 0; i < n; i++) {
            features[i] = fs.next();
            labels[i] = fs.nextInt();
        }
        String[] samples = new String[m];
        for (int i = 0; i < m; i++) {
            samples[i] = fs.next();
        }
        // 数据预处理
        int[][] embeddings = new int[n][7];
        for (int i = 0; i < n; i++) {
            embeddings[i] = onehot(features[i]);
        }
        // 模型初始化
        double[] w = new double[7];
        double b = 0;

        double learningRate = 0.1;
        int epoch = 20;
        for (int i = 0; i < epoch; i++) {
            for (int j = 0; j < n; j++) {
                double p = sigmoid(w, embeddings[j], b);
                int y = labels[j];
                for (int l = 0; l < 7; l++) {
                    w[l] = w[l] - learningRate * (p - y) * embeddings[j][l];
                }
                b = b - learningRate * (p - y);
            }
        }
        int[] result = new int[m];
        for (int i = 0; i < m; i++) {
            double p = sigmoid(w, onehot(samples[i]), b);
            if (Double.compare(p, 0.5) > 0) {
                result[i] = 1;
            } else result[i] = 0;
        }
        for (int i : result) {
            System.out.println(i);
        }
    }

    private static double sigmoid(double[] w, int[] embedding, double b) {
        double z = 0;
        for (int i = 0; i < 7; i++) {
            z = z + w[i] * embedding[i];
        }
        z = z + b;

        return 1 / (1 + Math.exp(-z));
    }

    private static int[] onehot(String feature) {
        int[] result = new int[7];
        for (char c : feature.toCharArray()) {
            result[c - 'A'] = 1;
        }
        return result;
    }
}
