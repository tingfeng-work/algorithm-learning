package huawei.day03.p4;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.StringTokenizer;

public class Main {

    /*
     * 商品购买预测
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

    private static final double EPSILON = 1e-12;

    public static void main(String[] args) throws IOException {
        FastScanner fs = new FastScanner();
        int n = fs.nextInt(); // 样本数量
        int maxIter = fs.nextInt();
        double alpha = fs.nextDouble();
        double lambda = fs.nextDouble();
        double tol = fs.nextDouble();

        double[][] X = new double[n][3];
        int[] labels = new int[n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < 3; j++) {
                X[i][j] = fs.nextDouble();
            }
            labels[i] = fs.nextInt();
        }

        int m = fs.nextInt(); // 测试样本数量
        double[][] M = new double[m][3];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < 3; j++) {
                M[i][j] = fs.nextDouble();
            }
        }

        double[] w = new double[3];
        double b = 0.0;

        double preLoss = Double.NaN;

        for (int iteration = 0; iteration < maxIter; iteration++) {
            // 计算所有预测概率
            double[] probabilities = new double[n];

            double crossEntropySum = 0.0;

            for (int i = 0; i < n; i++) {
                double z = getZ(w, X[i], b);

                double p = sigmoid(z);
                probabilities[i] = p;

                int label = labels[i];


                /*
                 * 单个样本的二分类交叉熵：
                 *
                 * -[y*ln(p) + (1-y)*ln(1-p)]
                 */

                crossEntropySum -=
                        label * Math.log(p + EPSILON)
                                + (1 - label) * Math.log(1.0 - p + EPSILON);
            }
            double ce = crossEntropySum / n;

            /*
             * L2正则化损失：
             *
             * lambda / (2n) * ||w||²
             *
             * 注意：只正则化权重，不正则化偏置。
             */
            double weightSquareSum = 0.0;
            for (double weight : w) {
                weightSquareSum += weight * weight;
            }

            double l2Loss = lambda * weightSquareSum / (2.0 * n);

            double totalLoss = ce + l2Loss;

            /*
             * 提前停止：
             *
             * 第一轮没有previousLoss，不能判断。
             * 从第二轮开始比较相邻两轮损失。
             */
            if (!Double.isNaN(preLoss)
                    && Math.abs(totalLoss - preLoss) < tol
            ) break;

            preLoss = totalLoss;

            // ---------- 计算全部样本的批量梯度 ----------
            double[] gradW = new double[3];

            double gradB = 0.0;

            for (int i = 0; i < n; i++) {
                double error = probabilities[i] - labels[i];

                for (int j = 0; j < 3; j++) {
                    gradW[j] += error * X[i][j];
                }
                gradB += error;


            }


            /*
             * 求平均，并在权重梯度中加入L2梯度：
             *
             * dw_j =
             * 1/n * Σ((p_i-y_i)x_ij)
             * + lambda/n * w_j
             */

            for (int j = 0; j < 3; j++) {
                gradW[j] = gradW[j] / n + lambda * w[j] / n;
            }
            gradB = gradB / n;
            // ---------- 统一更新参数 ----------

            for (int j = 0; j < 3; j++) {
                w[j] = w[j] - alpha * gradW[j];
            }
            b = b - alpha * gradB;
        }

        for (int i = 0; i < m; i++) {
            double z = getZ(w, M[i], b);
            double p = sigmoid(z);
            int prediction =
                    p >= 0.5 ? 1 : 0;

            System.out.printf("%d %.4f%n", prediction, p);
        }

    }

    private static double getZ(double[] w, double[] x, double b) {
        double z = 0.0;
        for (int i = 0; i < 3; i++) {
            z += w[i] * x[i];
        }
        return z + b;
    }

    private static double sigmoid(double z) {
        if (z >= 0) {
            return 1.0 / (1.0 + Math.exp(-z));
        }

        double expZ = Math.exp(z);
        return expZ / (1.0 + expZ);
    }
}
