package huawei.day02.p2;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;
import java.util.StringTokenizer;

public class Main {
    /*
     * 题目：网络流量分析
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

    static class Traffic {
        double[] features = new double[3];
    }

    static class Center {
        Traffic traffic;
        List<Traffic> cluster = new ArrayList<>();
    }

    public static void main(String[] args) throws IOException {
        FastScanner fs = new FastScanner();
        // 读取数据 + 初始化质心
        int k = fs.nextInt();
        Center[] centers = new Center[k];
        for (int i = 0; i < k; i++) {
            Traffic traffic = new Traffic();
            for (int j = 0; j < 3; j++) {
                traffic.features[j] = fs.nextDouble();
            }
            Center center = new Center();
            center.traffic = traffic;
            centers[i] = center;
        }
        int t = fs.nextInt(), m = fs.nextInt();
        Traffic[] traffics = new Traffic[m];
        for (int i = 0; i < m; i++) {
            Traffic traffic = new Traffic();
            for (int j = 0; j < 3; j++) {
                traffic.features[j] = fs.nextDouble();
            }
            traffics[i] = traffic;
        }
        while (t-- > 0) {
            // 清空簇
            for (Center center : centers) {
                center.cluster.clear();
            }

            // 分配样本
            for (Traffic traffic : traffics) {
                double minDis = Double.POSITIVE_INFINITY;
                int index = 0;
                for (int i = 0; i < k; i++) {
                    Center center = centers[i];
                    double dis = distance(traffic, center.traffic);
                    if (dis < minDis) {
                        index = i;
                        minDis = dis;
                    }
                }
                centers[index].cluster.add(traffic);
            }
            // 更新中心
            for (Center center : centers) {
                List<Traffic> cluster = center.cluster;
                if (cluster.isEmpty()) continue;
                double[] newFeatures = new double[3];
                for (Traffic traffic : cluster) {
                    for (int i = 0; i < 3; i++) {
                        newFeatures[i] += traffic.features[i];
                    }
                }
                for (int i = 0; i < 3; i++) {
                    newFeatures[i] /= cluster.size();
                }
                Traffic newTraffic = new Traffic();
                newTraffic.features = newFeatures;
                center.traffic = newTraffic;
            }
        }
        for (Center center : centers) {
            double[] features = center.traffic.features;
            System.out.printf(
                    "%.2f %.2f %.2f%n",
                    features[0],
                    features[1],
                    features[2]
            );
        }
    }

    private static double distance(Traffic traffic1, Traffic traffic2) {
        double result = 0;
        for (int i = 0; i < 3; i++) {
            result += Math.pow(traffic1.features[i] - traffic2.features[i], 2);
        }
        return Math.sqrt(result);
    }
}
