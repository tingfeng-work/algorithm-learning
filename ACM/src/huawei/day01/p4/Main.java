package huawei.day01.p4;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.*;

public class Main {
    /*
     * YOLO 检测器中的 Anchor 聚类
     * YOLO 目标检测算法通常会预先设置若干个不同尺寸的 Anchor 框，作为预测目标位置和尺寸的参考。
     * 为了使 Anchor 尺寸更符合训练数据中的真实检测框，
     * 可以对训练集中所有检测框的宽、高进行 K-Means 聚类，
     * 最终得到 K 个具有代表性的 Anchor 尺寸。
     * 给定 N 个检测框，每个检测框由宽度 w 和高度 h 表示。
     * 请按照题目规定的 K-Means 算法，将这些检测框聚类成 K 个 Anchor，并按照面积从大到小输出。
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

    static class Box {
        int wide;
        int high;

        public Box(int wide, int high) {
            this.wide = wide;
            this.high = high;
        }
    }

    static class Center {
        Box box;
        List<Box> cluster = new ArrayList<>();

        public Center(int wide, int high) {
            box = new Box(wide, high);
        }
    }

    public static void main(String[] args) throws IOException {
        FastScanner fs = new FastScanner();
        int n = fs.nextInt(), k = fs.nextInt(), t = fs.nextInt();
        // 初始化检测框
        Box[] boxes = new Box[n];
        for (int i = 0; i < n; i++) {
            int w = fs.nextInt();
            int h = fs.nextInt();
            boxes[i] = new Box(w, h);
        }
        // 初始化中心
        Center[] centers = new Center[k];
        for (int i = 0; i < k; i++) {
            Box box = boxes[i];
            centers[i] = new Center(box.wide, box.high);
        }
        while (t > 0) {
            // 分配
            for (Center center : centers) {
                center.cluster.clear();
            }
            for (Box box : boxes) {
                // 计算与聚类中心距离
                double minDis = Double.POSITIVE_INFINITY;
                int index = 0;
                for (int i = 0; i < k; i++) {
                    Center center = centers[i];
                    double distance = distance(box, center.box);
                    if (distance < minDis) {
                        index = i;
                        minDis = distance;
                    }
                }
                // 选择最小距离加入改中心的聚类
                centers[index].cluster.add(box);
            }
            double change = 0;
            // 更新
            for (Center center : centers) {
                List<Box> cluster = center.cluster;
                int count = cluster.size();
                if (count == 0) continue;
                int sumW = 0, sumH = 0;
                for (Box box : cluster) {
                    sumW += box.wide;
                    sumH += box.high;
                }
                int neww = sumW / count;
                int newH = sumH / count;
                Box newBox = new Box(neww, newH);
                change += distance(center.box, newBox);
                center.box = newBox;
            }
            if (change < 1e-4)
                break;
            t--;
        }
        Arrays.sort(centers, new Comparator<Center>() {
            @Override
            public int compare(Center o1, Center o2) {
                long area1 = (long) o1.box.wide * o1.box.high;
                long area2 = (long) o2.box.wide * o2.box.high;

                if (area1 != area2) {
                    return Long.compare(area2, area1);
                }

                if (o1.box.wide != o2.box.wide) {
                    return Integer.compare(o2.box.wide, o1.box.wide);
                }

                return Integer.compare(o2.box.high, o1.box.high);
            }
        });

        for (Center center : centers) {
            System.out.println(center.box.wide + " " + center.box.high);
        }
    }

    static double distance(Box box1, Box box2) {
        long intersection =
                (long) Math.min(box1.wide, box2.wide)
                        * Math.min(box1.high, box2.high);

        long area1 = (long) box1.wide * box1.high;
        long area2 = (long) box2.wide * box2.high;
        long union = area1 + area2 - intersection;

        double iou = intersection / (union + 1e-16);
        return 1.0 - iou;
    }
}
