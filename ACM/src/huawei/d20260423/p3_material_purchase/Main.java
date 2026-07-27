package huawei.d20260423.p3_material_purchase;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

public class Main {
    /*
     * 第三题：素材采购方案数（300 分）
     * 题目描述
     * 现在需要购买 N 种不同的素材，第 i 种素材的单价为 price[i]。
     * 你拥有恰好 budget 元预算，需要制定采购方案，并满足：
     * 1. 每种素材至少购买一个；
     * 2. 每种素材可以购买任意多个；
     * 3. 必须恰好花完全部预算；
     * 4. 即使两种素材价格相同，也属于不同的素材种类。
     * 求一共有多少种不同的采购方案。
     * 对于同一种素材，只关心购买数量，不考虑购买顺序。
     * 输入描述
     * 第一行输入一个整数：budget 表示总预算
     * 第二行输入若干个以空格分隔的整数：price[0] price[1] ... price[N-1]
     * 表示各类素材的单价，第二行整数的数量就是素材种类数 N
     * 输出一个整数，表示恰好使用全部预算的采购方案数量。
     * 答案可能超过 32 位整数范围，应使用 long。
     * 思路：完全背包问题：budget 是 target，求恰好选的物品和为 target 的方案数
     * 每种素材至少买一个怎么实现？就让target 先挨个减去 price，再去求方案数
     * */
    public static void main(String[] args) throws IOException {
        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
        int target = Integer.parseInt(br.readLine().trim());
        String[] strings = br.readLine().trim().split("\\s+");
        int n = strings.length;
        int[] nums = new int[n];
        for (int i = 0; i < n; i++) {
            nums[i] = Integer.parseInt(strings[i]);
            target = target - nums[i];
        }
        if (target < 0) {
            System.out.println(0);
            return;
        }
        long[] dp = new long[target + 1]; // dp[i][j] 表示前 i 个数和为 j 的方案数
        dp[0] = 1;
        for (int i = 0; i < n; i++) {
            for (int j = nums[i]; j <= target; j++) {
                dp[j] = dp[j - nums[i]] + dp[j];
            }
        }
        System.out.println(dp[target]);

    }
}
