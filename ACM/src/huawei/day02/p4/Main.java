package huawei.day02.p4;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.*;

public class Main {
    /*
     * 大模型训练 MOE 场景路由优化算法
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
        int n = fs.nextInt(), m = fs.nextInt(), p = fs.nextInt(), k = fs.nextInt();
        double[] probabilities = new double[n];
        for (int i = 0; i < n; i++) {
            probabilities[i] = fs.nextDouble();
        }
        // 校验输入
        if (n % m != 0) {
            System.out.println("error");
            return;
        }
        if (p > m) {
            System.out.println("error");
            return;
        }
        if (k > p * n / m) {
            System.out.println("error");
            return;
        }
        int groupSize = n / m;
        // 每个组的代表概率
        double[] presentPro = new double[m];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < groupSize; j++) {
                int index = i * groupSize + j;
                presentPro[i] = Math.max(probabilities[index], presentPro[i]);
            }
        }
        // NPU 编号
        Integer[] groupId = new Integer[m];
        for (int i = 0; i < m; i++) {
            groupId[i] = i;
        }
        // 将编号按照代表概率排序
        Arrays.sort(groupId, new Comparator<Integer>() {
            @Override
            public int compare(Integer id1, Integer id2) {
                int compare = Double.compare(presentPro[id2], presentPro[id1]);
                if (compare != 0) return compare;
                return Integer.compare(id1, id2);
            }
        });
        // 选择 p 张 NPU 拿到候选专家
        List<Integer> candidates = new ArrayList<>();
        for (int i = 0; i < p; i++) {
            int start = groupId[i] * groupSize;
            int end = start + groupSize;
            for (int id = start; id < end; id++) {
                candidates.add(id);
            }
        }

        // 选择 k 个专家
        candidates.sort(new Comparator<Integer>() {
            @Override
            public int compare(Integer id1, Integer id2) {
                int compare = Double.compare(
                        probabilities[id2],
                        probabilities[id1]
                );

                if (compare != 0) {
                    return compare;
                }

                return Integer.compare(id1, id2);
            }
        });
        List<Integer> answer = new ArrayList<>();

        for (int i = 0; i < k; i++) {
            answer.add(candidates.get(i));
        }
        Collections.sort(answer);

        StringBuilder sb = new StringBuilder();

        for (int i = 0; i < answer.size(); i++) {
            if (i > 0) {
                sb.append(" ");
            }

            sb.append(answer.get(i));
        }

        System.out.println(sb);

    }
}
