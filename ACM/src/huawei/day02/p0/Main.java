package huawei.day02.p0;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Arrays;
import java.util.StringTokenizer;

public class Main {
    /*
     * 题目：大模型 Attention 模块开发
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
        int n = fs.nextInt(), m = fs.nextInt(), h = fs.nextInt();
        int[][] x = new int[n][m];
        int[][] w = new int[m][h];
        for (int[] ints : x) {
            Arrays.fill(ints, 1);
        }
        for (int i = 0; i < m; i++) {
            for (int j = i; j < h; j++) {
                 w[i][j] = 1;
            }
        }
        int[][] qkv = matrix_multi(x, w);
        int[][] q_k_T = matrix_multi(qkv, matrix_T(qkv));
        double sqrt_h = Math.sqrt(h);
        double[][] M = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                M[i][j] = q_k_T[i][j] / sqrt_h;
            }
        }
        double[][] A = softmax(M);
        double[][] Y = matrix_multi(A, qkv);
        double ans = 0;
        for (double[] doubles : Y) {
            for (double aDouble : doubles) {
                ans += aDouble;
            }
        }
        System.out.println(Math.round(ans));
    }

    private static double[][] matrix_multi(double[][] matrix1, int[][] matrix2) {
        int n = matrix1.length, m = matrix2[0].length, h = matrix2.length;
        double[][] newMatrix = new double[n][m];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                for (int k = 0; k < h; k++) {
                    newMatrix[i][j] += matrix1[i][k] * matrix2[k][j];
                }
            }
        }
        return newMatrix;
    }

    private static double[][] softmax(double[][] matrix) {
        int n = matrix.length, m = matrix[0].length;
        double[] sum = new double[n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                sum[i] += matrix[i][j];
            }
        }
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                matrix[i][j] = matrix[i][j] / sum[i];
            }
        }
        return matrix;
    }

    private static int[][] matrix_T(int[][] matrix) {
        int n = matrix.length, m = matrix[0].length;
        int[][] newMatrix = new int[m][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                newMatrix[j][i] = matrix[i][j];
            }
        }
        return newMatrix;
    }

    private static int[][] matrix_multi(int[][] matrix1, int[][] matrix2) {
        if (matrix1[0].length != matrix2.length) return null;
        int n = matrix1.length, m = matrix2[0].length, h = matrix2.length;
        int[][] newMatrix = new int[n][m];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                for (int k = 0; k < h; k++) {
                    newMatrix[i][j] += matrix1[i][k] * matrix2[k][j];
                }
            }
        }
        return newMatrix;
    }
}
