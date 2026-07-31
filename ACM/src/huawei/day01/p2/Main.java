package huawei.day01.p2;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.HashMap;
import java.util.Map;
import java.util.StringTokenizer;


public class Main {
    /*
     * 第 3 题：大模型分词
     * 你正在为一种罕见语言构建专用大语言模型。由于训练样本不足，BPE 等常规分词器效果不佳。
     * 语言学家提供了：
     * 1. 一个已知词元表，每个词元具有一个置信度分数；
     * 2. 一个转移分数表，表示前一个词元对后一个词元的额外影响
     * 现在给定一个不含空格、只包含英文小写字母的字符串 text，请将其完整切分为若干个已知词元，使最终得分最大。
     * 假设切分结果为：token1, token2, ..., tokenK
     * ∑confidence(token[i])+∑transition(token[i],token[i+1])
     * 其中：
     * 每个词元都会贡献自己的置信度；
     * 相邻两个词元如果存在转移分数，则增加对应分数；
     * 如果转移表中不存在该词元对，则转移分数为 0；
     * 第一个词元没有前驱，因此没有转移加分；
     * 必须使用词汇表中的词元完整覆盖原字符串；
     * 不能跳过任何字符，也不能改变字符顺序。
     *
     * 如果字符串无法使用已知词元完整切分，输出 0。
     * 思路：这道题也是AI背景下的传统算法题，计算字符串分割的最高得分
     * 词元表可以用hash表记录，转移表也可以用hash表记录
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

    private static final int NEG_INF = Integer.MIN_VALUE / 4;
    private static Map<String, Integer> memo;

    public static void main(String[] args) throws IOException {
        FastScanner fs = new FastScanner();
        String text = fs.next();
        int n = fs.nextInt();
        Map<String, Integer> tokens = new HashMap<>(n);
        for (int i = 0; i < n; i++) {
            String token = fs.next();
            int confidence = fs.nextInt();
            tokens.put(token, confidence);
        }
        int m = fs.nextInt();
        Map<String, Map<String, Integer>> transitions = new HashMap<>();
        for (int i = 0; i < m; i++) {
            String pre = fs.next();
            String next = fs.next();
            int score = fs.nextInt();
            transitions.computeIfAbsent(pre, key -> new HashMap<>()).put(next, score);
        }
        // 当前操作：枚举text[0, end)中最后一个词元的起点start
        // 子问题：切分当前词元左侧的text[0, start)
        // 下一个子问题：前 i-当前词元长度个字符的划分最高分
        // 同时记录 next，枚举到当前词元时，查 transitions 表
        int len = text.length();
        memo = new HashMap<>();
        int ans = dfs(len, "", text, tokens, transitions);
        System.out.println(ans == NEG_INF ? 0 : ans);
    }


    /*
     * dfs(end, next)：
     * 将text[0, end)完整切分，并且切分结果右侧的词元为next时，
     * 可以取得的最高分。
     */
    private static int dfs(int end, String next, String text, Map<String, Integer> tokens, Map<String, Map<String, Integer>> transitions) {
        if (end == 0) {
            // 说明切分完了
            return 0;
        }
        String cacheKey = end + "#" + next;
        if (memo.containsKey(cacheKey))
            return memo.get(cacheKey);
        int res = NEG_INF;
        for (int start = end - 1; start >= 0; start--) {
            String current = text.substring(start, end); // 左闭右开
            if (!tokens.containsKey(current))
                continue;
            // 前面的得分
            int preScore = dfs(start, current, text, tokens, transitions);
            if (preScore == NEG_INF)
                continue;
            int score = 0;
            if (!next.isEmpty()) {
                Map<String, Integer> scores = transitions.get(current);
                if (scores != null) {
                    score = scores.getOrDefault(next, 0);
                }
            }
            int currentScore = preScore + tokens.get(current) + score;
            res = Math.max(res, currentScore);
        }
        memo.put(cacheKey, res);
        return res;
    }
}
