package huawei.day02.p1;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.*;

public class Main {
    /*
     * 终端款型聚类识别
     * 某部门需要保障终端的漫游业务体验。
     * 不同型号的终端对网络配置的要求不同，因此需要根据终端的网络流量等特征识别终端型号。
     * 每台终端具有以下4个经过归一化的特征：
     * 1. 包间隔时长；
     * 2. 连接持续时长；
     * 3. 漫游前信号强度；
     * 4. 漫游后信号强度。
     * 已知终端共分为 K 种款型，请使用 K-Means 算法对终端进行聚类，并输出每种款型包含的终端数量。
     *
     * K-Means算法
     * 1. 初始化
     * 选择 K 个初始质心。本题为了消除随机性，规定：使用输入数据中的前 K 个终端作为初始质心。
     * 2. 分配
     * 对于每个终端，计算它与所有质心之间的欧氏距离，将其分配给距离最近的质心，形成 K 个簇。
     * 3. 更新
     * 对每个簇，将簇内所有终端在每个维度上的平均值作为新的质心
     * 4. 迭代
     * 不断重复“分配”和“更新”，直到满足以下任意条件：
     * 质心移动值小于 10−8；
     * 达到题目给定的最大迭代次数 n。
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

    static class Terminal {
        double[] features = new double[4];
    }

    static class Center {
        Terminal terminal;
        List<Terminal> cluster = new ArrayList<>();
    }


    public static void main(String[] args) throws IOException {
        FastScanner fs = new FastScanner();
        int k = fs.nextInt(), m = fs.nextInt(), n = fs.nextInt();
        // 读取数据
        Terminal[] terminals = new Terminal[m];
        for (int i = 0; i < m; i++) {
            Terminal terminal = new Terminal();
            for (int j = 0; j < 4; j++) {
                terminal.features[j] = fs.nextDouble();
            }
            terminals[i] = terminal;
        }
        // → 初始化质心
        Center[] centers = new Center[k];
        for (int i = 0; i < k; i++) {
            Center center = new Center();
            center.terminal = terminals[i];
            centers[i] = center;
        }
        while (n-- > 0) {
            // → 清空分组
            for (Center center : centers) {
                center.cluster.clear();
            }
            // → 分配终端
            for (Terminal terminal : terminals) {
                double minDis = Double.POSITIVE_INFINITY;
                int index = 0;
                for (int i = 0; i < k; i++) {
                    Center center = centers[i];
                    double dis = distance(terminal, center.terminal);
                    if (dis < minDis) {
                        index = i;
                        minDis = dis;
                    }
                }
                centers[index].cluster.add(terminal);
            }
            // → 更新质心
            double changeSquareSum = 0;
            for (Center center : centers) {
                List<Terminal> cluster = center.cluster;
                if (cluster.isEmpty()) {
                    continue;
                }
                double[] newFeatures = new double[4];
                for (Terminal terminal : cluster) {
                    double[] features = terminal.features;
                    for (int i = 0; i < 4; i++) {
                        newFeatures[i] += features[i];
                    }
                }
                for (int i = 0; i < 4; i++) {
                    newFeatures[i] = newFeatures[i] / cluster.size();
                    double diff = newFeatures[i] - center.terminal.features[i];
                    changeSquareSum += diff * diff;
                }
                Terminal newTerminal = new Terminal();
                newTerminal.features = newFeatures;
                center.terminal = newTerminal;
            }
            // → 判断是否停止
            double change = Math.sqrt(changeSquareSum);

            if (change < 1e-8) {
                break;
            }

        }
        // → 统计各簇数量
        Arrays.sort(centers, new Comparator<Center>() {
            @Override
            public int compare(Center o1, Center o2) {
                return Integer.compare(
                        o1.cluster.size(),
                        o2.cluster.size()
                );
            }
        });
        // → 排序输出
        StringBuilder sb = new StringBuilder();
        for (Center center : centers) {
            sb.append(center.cluster.size());
            sb.append(" ");
        }
        System.out.println(new String(sb).trim());


    }

    private static double distance(Terminal terminal1, Terminal terminal2) {
        double result = 0;
        for (int i = 0; i < 4; i++) {
            result += Math.pow(terminal1.features[i] - terminal2.features[i], 2);
        }
        return Math.sqrt(result);
    }
}
