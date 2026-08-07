package huawei.day02.p3;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.*;

public class Main {
    /*
     * 无线网络优化中的基站聚类分析
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

    static class BaseStation {
        double x;
        double y;
        double s; // 轮廓系数
    }

    static class Center {
        BaseStation baseStation;
        List<BaseStation> cluster = new ArrayList<>();
    }

    public static void main(String[] args) throws IOException {
        FastScanner fs = new FastScanner();
        // 读取数据
        int n = fs.nextInt(), k = fs.nextInt();
        BaseStation[] baseStations = new BaseStation[n];
        for (int i = 0; i < n; i++) {
            BaseStation baseStation = new BaseStation();
            baseStation.x = fs.nextDouble();
            baseStation.y = fs.nextDouble();
            baseStations[i] = baseStation;
        }
        // 初始化中心
        Center[] centers = new Center[k];
        for (int i = 0; i < k; i++) {
            Center center = new Center();
            center.baseStation = baseStations[i];
            centers[i] = center;
        }
        int count = 100;
        while (count-- > 0) {
            // 清空簇
            for (Center center : centers) {
                center.cluster.clear();
            }
            // 分配基站
            for (BaseStation baseStation : baseStations) {
                double minDis = Double.POSITIVE_INFINITY;
                int index = 0;
                for (int i = 0; i < k; i++) {
                    double dis = distance(baseStation, centers[i].baseStation);
                    if (dis < minDis) {
                        index = i;
                        minDis = dis;
                    }
                }
                centers[index].cluster.add(baseStation);
            }
            // 更新
            boolean allStable = true; // 表示是否需要终止
            for (Center center : centers) {
                List<BaseStation> cluster = center.cluster;
                if (cluster.isEmpty()) continue;
                double newX = 0.0, newY = 0.0;
                for (BaseStation station : cluster) {
                    newX += station.x;
                    newY += station.y;
                }
                BaseStation newStation = new BaseStation();
                newStation.x = newX / cluster.size();
                newStation.y = newY / cluster.size();
                double change = distance(center.baseStation, newStation);
                if (change > 1e-6) {
                    allStable = false;
                }
                center.baseStation = newStation;
            }
            if (allStable) break;
        }
        for (Center center : centers) {
            center.cluster.clear();
        }

        for (BaseStation station : baseStations) {
            double minDistance = Double.POSITIVE_INFINITY;
            int bestCenter = 0;

            for (int i = 0; i < k; i++) {
                double currentDistance =
                        distance(station, centers[i].baseStation);

                if (currentDistance < minDistance) {
                    minDistance = currentDistance;
                    bestCenter = i;
                }
            }

            centers[bestCenter].cluster.add(station);
        }
        // 计算轮廓系数
        // 1. 簇内平均距离 a
        for (Center center : centers) {
            List<BaseStation> cluster = center.cluster;
            int size = cluster.size();
            if (size == 1) {
                cluster.getFirst().s = 1.0;
                continue;
            }
            for (int i = 0; i < size; i++) {
                double a = 0.0;
                for (int j = 0; j < size; j++) {
                    if (j == i) continue;
                    a += distance(cluster.get(i), cluster.get(j));
                }
                cluster.get(i).s = a / (size - 1);
            }
        }
        // 2. 最近其他簇平均距离 b
        for (int i = 0; i < k; i++) {
            List<BaseStation> curCluster = centers[i].cluster; // 当前簇
            if (curCluster.size() == 1) continue;
            for (BaseStation curStation : curCluster) {
                double b = Double.POSITIVE_INFINITY;
                for (int j = 0; j < k; j++) {
                    if (j == i) continue;
                    // 其他簇
                    List<BaseStation> cluster = centers[j].cluster;
                    double dis = 0.0;
                    for (BaseStation station : cluster) {
                        dis += distance(curStation, station);
                    }
                    dis = dis / cluster.size();
                    b = Math.min(b, dis);
                }
                double temp = Math.max(b, curStation.s);
                curStation.s = (b - curStation.s) / temp;
            }
        }
        // 一个簇的轮廓系数
        for (Center center : centers) {
            List<BaseStation> cluster = center.cluster;
            for (BaseStation station : cluster) {
                center.baseStation.s += station.s;
            }
            center.baseStation.s /= cluster.size();
        }
        Center answer = centers[0];

        for (int i = 1; i < k; i++) {
            if (centers[i].baseStation.s
                    < answer.baseStation.s) {
                answer = centers[i];
            }
        }
        System.out.println(
                format(answer.baseStation.x)
                        + ","
                        + format(answer.baseStation.y)
        );


    }

    private static String format(double value) {
        return BigDecimal.valueOf(value)
                .setScale(2, RoundingMode.HALF_EVEN)
                .toPlainString();
    }

    private static double distance(BaseStation station1, BaseStation station2) {
        double diff1 = station1.x - station2.x;
        double diff2 = station1.y - station2.y;
        return Math.sqrt(diff1 * diff1 + diff2 * diff2);
    }
}
