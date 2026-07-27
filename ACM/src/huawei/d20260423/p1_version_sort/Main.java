package huawei.d20260423.p1_version_sort;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.StringTokenizer;

public class Main {
    /*
     * 题目：版本号整理
     * 给定 N 个软件版本号，请按照规定从小到大排序。
     * 每个版本号由两部分组成：
     * 1. 主版本号 由 1～4 个非负整数组成，各部分之间使用 . 分隔，例如：
     * 1     1.0     1.0.2       1.0.2.3
     * 2. 可选的测试版本号
     * 测试版本号位于主版本号后，中间用一个空格分隔，格式为： betaX
     * 其中 X 为正整数，例如：1.0.2 beta3
     * 不包含 betaX 的版本称为正式版
     * 排序规则
     * 从左到右逐段比较主版本号中的整数。
     * 如果对应位置的数字不同，数字较小的版本排在前面。
     * 如果一个主版本号是另一个主版本号的前缀，较短的版本排在前面。
     * */
    static class Version {
        String original;
        int[] mainParts;
        Integer beta;

        Version(String original, int[] mainParts, Integer beta) {
            this.original = original;
            this.mainParts = mainParts;
            this.beta = beta;
        }
    }

    public static void main(String[] args) throws IOException {
        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
        int n = Integer.parseInt(br.readLine().trim());
        List<Version> list = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            String line = br.readLine().trim();
            String[] s = line.split(" ");
            String[] main = s[0].split("\\.");
            int[] mainParts = new int[main.length];
            for (int j = 0; j < main.length; j++) {
                mainParts[j] = Integer.parseInt(main[j]);
            }
            Integer beta = null;
            if (s.length == 2) {
                beta = Integer.parseInt(s[1].substring(4));
            }
            list.add(new Version(line, mainParts, beta));
        }
        list.sort(new Comparator<Version>() {
            @Override
            public int compare(Version v1, Version v2) {
                int[] nums1 = v1.mainParts;
                int[] nums2 = v2.mainParts;
                int n = nums1.length, m = nums2.length;
                for (int i = 0; i < n && i < m; i++) {
                    int a = nums1[i];
                    int b = nums2[i];
                    if (a < b) return -1;
                    if (a > b) return 1;
                }
                if (n < m) return -1;
                if (n > m) return 1;
                if (v1.beta == null && v2.beta == null) {
                    return 0;
                }

                if (v1.beta == null) {
                    return 1;
                }

                if (v2.beta == null) {
                    return -1;
                }

                return Integer.compare(v1.beta, v2.beta);
            }
        });
        for (Version version : list) {
            System.out.println(version.original);
        }

    }
}