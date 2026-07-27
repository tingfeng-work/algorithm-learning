import java.util.*;

public class Solution {
    public int compareVersion(String version1, String version2) {
        /*
         * 165. 比较版本号
         * 给你两个版本号字符串 version1 和 version2 ，请你比较它们。
         * 版本号由被点 '.' 分开的修订号组成。修订号的值是它转换为整数并忽略前导零。
         * 比较版本号时，请按 从左到右的顺序 依次比较它们的修订号。
         * 如果其中一个版本字符串的修订号较少，则将缺失的修订号视为 0。
         * */
        String[] s1 = version1.split("\\.");
        String[] s2 = version2.split("\\.");
        int n = s1.length, m = s2.length;
        for (int i = 0; i < n || i < m; i++) {
            int a = i < n ? Integer.parseInt(s1[i]) : 0;
            int b = i < m ? Integer.parseInt(s2[i]) : 0;
            if (a < b) return -1;
            if (a > b) return 1;
        }
        return 0;
    }

    public int networkDelayTime(int[][] times, int n, int k) {
        /*
         * 743. 网络延迟时间
         * 有 n 个网络节点，标记为 1 到 n。
         * 给你一个列表 times，表示信号经过有向边的传递时间。
         * times[i] = (ui, vi, wi)，其中 ui 是源节点，vi 是目标节点， wi 是一个信号从源节点传递到目标节点的时间。
         * 现在，从某个节点 K 发出一个信号。需要多久才能使所有节点都收到信号？如果不能使所有节点收到信号，返回 -1 。
         * 思路：求得是从 k 出发到其他节点的最短路径中的最大值，由于 wi 非负，所以使用迪杰斯特拉算法
         * */

        List<int[]>[] g = new ArrayList[n + 1]; // 邻接表
        Arrays.setAll(g, e -> new ArrayList<>());
        for (int[] edge : times) {
            int x = edge[0];
            int y = edge[1];
            int w = edge[2];
            g[x].add(new int[]{y, w});
        }

        PriorityQueue<int[]> queue = new PriorityQueue<>(Comparator.comparingInt(a -> a[0])); // 小堆顶，用来取每次的最短路径
        int[] dis = new int[n + 1];
        Arrays.fill(dis, Integer.MAX_VALUE);
        dis[k] = 0;
        queue.offer(new int[]{0, k});
        while (!queue.isEmpty()) {
            int[] ints = queue.poll();
            int disX = ints[0];
            int x = ints[1];
            if (disX > dis[x]) {
                continue; // 表示已经从堆中取出，并计算过
            }
            for (int[] ints2 : g[x]) {
                int disY = ints2[1];
                int y = ints2[0];
                int newDis = disX + disY;
                if (newDis < dis[y]) {
                    dis[y] = newDis;
                    queue.offer(new int[]{newDis, y});
                }
            }
        }
        int ans = 0;
        for (int i = 1; i < dis.length; i++) {
            if (dis[i] == Integer.MAX_VALUE) return -1;
            ans = Math.max(dis[i],ans);
        }
        return ans;
    }
}
