package huawei.day03.p1;


import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Arrays;
import java.util.Locale;
import java.util.StringTokenizer;

public class Main {

    /*
     * F1 值最优的决策树剪枝
     *
     * 核心思路：
     *
     * 1. 先让每个验证样本沿原始决策树进行推理。
     *    统计每个节点会接收到多少正样本和负样本。 // 统计每个节点剪枝会影响多少样本
     *
     * 2. 对每个节点进行树形 DP。
     *
     *    对于非叶子节点有两种选择：
     *
     *    ① 剪枝：
     *       删除左右子树，让当前节点直接作为叶子节点，
     *       所有到达当前节点的样本都预测为 node.label。
     *
     *    ② 不剪枝：
     *       保留当前节点，分别优化左右子树，
     *       然后合并左右子树产生的 TP 和 FP。
     *
     * 3. dp[u][tp] 表示：
     *    以 u 为根的子树经过剪枝后，
     *    当产生 tp 个真正例时，最少产生多少个假正例。
     *
     * 4. 在根节点枚举所有 TP，计算：
     *
     *       F1 = 2TP / (2TP + FP + FN)
     *
     *    因为 FN = totalPositive - TP，所以：
     *
     *       F1 = 2TP / (totalPositive + TP + FP)
     */
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

    static class Node {
        int left;
        int right;
        int featureIndex;
        int threshold;
        int label;

        public Node(int left, int right, int featureIndex, int threshold, int label) {
            this.left = left;
            this.right = right;
            this.featureIndex = featureIndex;
            this.threshold = threshold;
            this.label = label;
        }

        boolean isLeaf() {
            return this.left == 0 && this.right == 0;
        }

    }

    private static Node[] tree;
    private static int[] posCount;
    private static int[] negCount;

    private static final int INF = 1_000_000_000;

    public static void main(String[] args) throws IOException {
        FastScanner fs = new FastScanner();
        int n = fs.nextInt(), m = fs.nextInt(), k = fs.nextInt();
        tree = new Node[n + 1];
        posCount = new int[n + 1]; // posCount[i] 表示树上经过节点 i 的正样本数量
        negCount = new int[n + 1];


        for (int i = 1; i <= n; i++) {
            int left = fs.nextInt();
            int right = fs.nextInt();
            int featureIndex = fs.nextInt();
            int threshold = fs.nextInt();
            int label = fs.nextInt();

            tree[i] = new Node(left, right, featureIndex, threshold, label);


        }
        int totalPositive = 0;

        for (int i = 0; i < m; i++) {
            int[] sample = new int[k];
            for (int j = 0; j < k; j++) {
                sample[j] = fs.nextInt();
            }
            int trueLabel = fs.nextInt();

            if (trueLabel == 1) totalPositive++;

            routeCount(sample, trueLabel); // 统计树中每个节点接收多少正样本与负样本
        }

        // 树形 dp
        /*
         * rootDp[tp] 表示：
         * 整棵树经过某种剪枝后，得到 tp 个真正例时，
         * 最少产生 rootDp[tp] 个假正例。
         */
        int[] rootDp = dfs(1);

        double ans = 0.0;

        /*
         * rootDp[tp] 表示：
         * 整棵树经过某种剪枝后，得到 tp 个真正例时，
         * 最少产生 rootDp[tp] 个假正例。
         */
        for (int tp = 0; tp < rootDp.length; tp++) {
            int fp = rootDp[tp];

            if (fp == INF) {
                continue;
            }

            /*
             * FN = 正样本总数 - TP
             *
             * F1 = 2TP / (2TP + FP + FN)
             *    = 2TP / (2TP + FP + totalPositive - TP)
             *    = 2TP / (totalPositive + TP + FP)
             */
            int denominator = totalPositive + tp + fp;

            double f1;

            if (denominator == 0) f1 = 0.0;
            else f1 = 2.0 * tp / denominator;

            ans = Math.max(ans, f1);
        }

        System.out.printf(Locale.US, "%.6f%n", ans);


    }

    /**
     * 树形 DP。
     * <p>
     * 返回数组 dp：
     * <p>
     * dp[tp] 表示以节点 u 为根的子树经过某种剪枝后，
     * 当产生 tp 个真正例时，最少产生多少个假正例。
     * <p>
     * 不可达状态为 INF。
     */
    private static int[] dfs(int u) {
        Node node = tree[u];

        int[] dp = new int[posCount[u] + 1]; // 最多产生 posCount[u] 个 tp
        Arrays.fill(dp, INF);

        // 分类讨论：
        // 当前节点作为叶子节点

        if (node.label == 0) {
            // 0 个 tp 产生 0 个 fp 产生
            dp[0] = 0;
        } else {
            int tp = posCount[u];
            int fp = negCount[u];

            dp[tp] = fp;
        }
        // 当前节点本来就是叶子节点，无需讨论子节点
        if (node.isLeaf())
            return dp;

        // 不剪当前节点
        int[] leftDp = dfs(node.left);
        int[] rightDp = dfs(node.right);

        for (int leftTp = 0; leftTp < leftDp.length; leftTp++) {
            if (leftDp[leftTp] == INF) {
                continue;
            }
            for (int rightTp = 0; rightTp < rightDp.length; rightTp++) {
                if (rightDp[rightTp] == INF) {
                    continue;
                }

                /*
                 * 相同 TP 下，只保留最小 FP。
                 *
                 * 因为 TP 相同时，FP 越小，最终 F1 一定越大。
                 */
                int totalTp = leftTp + rightTp;
                int totalFp = leftDp[leftTp] + rightDp[rightTp];
                dp[totalTp] = Math.min(dp[totalTp], totalFp);
            }
        }


        return dp;
    }

    private static void routeCount(int[] sample, int trueLabel) {
        int nodeIndex = 1; // 根节点从 1 标号开始
        while (true) {
            // 推理决策
            if (trueLabel == 1) {
                posCount[nodeIndex]++;
            } else {
                negCount[nodeIndex]++;
            }
            Node node = tree[nodeIndex];
            if (node.isLeaf()) {
                break;
            }
            int featureIndex = node.featureIndex;
            int feature = sample[featureIndex - 1];
            if (feature <= node.threshold) nodeIndex = node.left;
            else nodeIndex = node.right;
        }
    }


}