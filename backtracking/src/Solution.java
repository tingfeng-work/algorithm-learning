import com.sun.jdi.Value;

import java.util.*;

class Solution {
    public List<String> letterCombinations(String digits) {
        /*17.电话号码的字母组合
         * 给定一个仅包含数字 2-9 的字符串，返回所有它能表示的字母组合。答案可以按任意顺序返回。
         * 给出数字到字母的映射与电话按键相同
         * 由于答案的每个元素组合的长度根据输入字符串的长度不同会变化，循环嵌套的表达有限。
         * 怎么理解这句话？如果固定输入 digits 的长度为 2，则可以枚举第一个 digit 与第二个digit 映射的字符，
         * 将他们组合。由于长度是变的，不知道需要套几层循环，由此引入回溯
         * 怎么理解回溯，例如深度优先遍历中，当遍历了子节点，回到父节点的这个现象就叫回溯
         * 怎么与这道题结合：例如构造了答案 “ad” 然后归到父节点 “a”这个过程，再从“a”递到“ae”
         * */
        int n = digits.length();
        if (n == 0)
            return List.of();
        List<String> map = List.of("", "", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz");
        List<String> ans = new ArrayList<>();
        char[] path = new char[n];
//        dfs(0, n, ans, path, digits, map);
        return ans;
    }

    // 枚举下标大于等于i的剩余部分
    private void dfs(int i, List<String> ans, char[] path, String digits, List<String> map) {
        if (i == digits.length()) {
            ans.add(new String(path));
            return;
        }
        String str = map.get(digits.charAt(i) - '0');
        for (char c : str.toCharArray()) {
            path[i] = c; //通过直接覆盖，来实现回溯
            dfs(i + 1, ans, path, digits, map);
        }
    }

    public List<List<Integer>> subsets(int[] nums) {
        /*78.子集
         * 给你一个整数数组 nums ，数组中的元素互不相同 。返回该数组所有可能的子集（幂集）。
         * 解集不能包含重复的子集。你可以按任意顺序返回解集。
         * 由于解集不含重复，即[1,2] 与 [2,1] 相同，则我们可以规定枚举的顺序，
         * 枚举的下一个下标一定大于当前下标，这样就实现了去重，例如，枚举了[1,2]，在枚举[2]时，就只能枚举[2,3]
         * 这里主要关注的是下标，里面的值的大小无所谓
         * 还是采用回溯的思想，有两个思路：当前答案选谁；从输入的角度选元素
         * */
//        int len = nums.length;
//        if (len == 0) return List.of();
//        List<List<Integer>> ans = new ArrayList<>();
//        List<Integer> path = new Stack<>();
//        dfs(0, nums, ans, path);
//        return ans;
        // 选或不选的实现
        List<List<Integer>> ans = new ArrayList<>();
        List<Integer> path = new ArrayList<>();
        dfs(0, nums, path, ans);
        return ans;
    }

    // 从输入的角度枚举每个元素选与不选,叶子节点添加答案
//    private void dfs(int i, int[] nums, List<Integer> path, List<List<Integer>> ans) {
//        if (i == nums.length) {
//            ans.add(new ArrayList<>(path));
//            return;
//        }
//
//        //不选
//        dfs(i + 1, nums, path, ans);
//        // 选
//        path.add(nums[i]);
//        dfs(i + 1, nums, path, ans);
//        path.removeLast();
//    }

    // 当前答案选谁的实现
//    private void dfs(int i, int[] nums, List<List<Integer>> ans, List<Integer> path) {
//        ans.add(new ArrayList<>(path));
//        for (int j = i; j < nums.length; j++) {
//            path.add(nums[j]);
//            dfs(j+ 1, nums, ans, path);
//            path.removeLast();
//        }
//    }

    public List<List<String>> partition(String s) {
        /*131.分割回文串
         * 给你一个字符串 s，请你将 s 分割成一些子串，
         * 使每个子串都是回文串，返回 s 所有可能的分割方案。
         * 思路：先得到分割的结果集，再遍历结果集筛选是否回文加入答案
         * */
        List<List<String>> ans = new ArrayList<>();
        List<String> path = new ArrayList<>();
        dfs(0, 0, s, path, ans);
        return ans;
    }

    // 枚举第i个逗号选不选;也可以理解成：是否要在 i 和 i+1 处分割，
    private void dfs(int i, int start, String s, List<String> path, List<List<String>> ans) {
        if (i == s.length()) {
            ans.add(new ArrayList<>(path));
            return;
        }

        if (i < s.length() - 1) {
            //不选
            dfs(i + 1, start, s, path, ans);
        }

        //选
        if (isPalindrome(s, start, i)) {
            path.add(s.substring(start, i + 1));
            dfs(i + 1, i + 1, s, path, ans);
            path.removeLast();
        }
    }

    // 答案视角：考虑 s[i] ~ s[n-1] 怎么分割
//    private void dfs(int i, String s, List<String> path, List<List<String>> ans) {
//        if (i == s.length()) {
//            ans.add(new ArrayList<>(path));
//            return;
//        }
//        for (int j = i; j < s.length(); j++) { // 从 j 分割
//            if (isPalindrome(s, i, j)) {
//                path.add(s.substring(i, j+1));
//                dfs(j+1, s, path, ans);
//                path.removeLast();
//            }
//        }
//
//    }

    private boolean isPalindrome(String s, int left, int right) {
        while (left < right) {
            if (s.charAt(left) != s.charAt(right)) return false;
            left++;
            right--;
        }
        return true;
    }

    public List<String> binaryTreePaths(TreeNode root) {
        /*257.二叉树的所有路径
         * 给你一个二叉树的根节点 root ，按 任意顺序 ，返回所有从根节点到叶子节点的路径。
         *
         * */
        List<String> ans = new ArrayList<>();
        List<Integer> path = new ArrayList<>();
        dfs(root, path, ans);
        return ans;
    }

    private void dfs(TreeNode node, List<Integer> path, List<String> ans) {
        if (node == null)
            return;

        if (node.left == null && node.right == null) {
            // 叶子节点记录答案
            StringBuilder sb = new StringBuilder();
            for (Integer s : path) {
                sb.append(s + "->");
            }
            sb.append(node.val);
            ans.add(sb.toString());
        }
        path.add(node.val);
        dfs(node.left, path, ans);
        dfs(node.right, path, ans);
        path.removeLast();

    }

    public List<List<Integer>> pathSum(TreeNode root, int targetSum) {
        /*113.路径总和Ⅱ
         * 给你二叉树的根节点 root 和一个整数目标和 targetSum ，
         * 找出所有从根节点到叶子节点 路径总和等于给定目标和的路径。
         * */
        List<List<Integer>> ans = new ArrayList<>();
        List<Integer> path = new ArrayList<>();
        dfs(root, 0, targetSum, path, ans);
        return ans;
    }

    private void dfs(TreeNode node, int sum, int targetSum, List<Integer> path, List<List<Integer>> ans) {
        if (node == null)
            return;
        sum = sum + node.val;
        path.add(node.val);
        if (node.left == null && node.right == null && sum == targetSum) {
            ans.add(new ArrayList<>(path));
        } else {
            dfs(node.left, sum, targetSum, path, ans);
            dfs(node.right, sum, targetSum, path, ans);
        }
        path.removeLast();

    }

    public List<String> letterCasePermutation(String s) {
        /*784.字母大小写全排列
         * 给定一个字符串 s ，通过将字符串 s 中的每个字母转变大小写，我们可以获得一个新的字符串。
         * 返回所有可能得到的字符串集合 。以任意顺序返回输出。
         * */
        List<String> ans = new ArrayList<>();
        char[] path = new char[s.length()];
        dfs(0, s.toCharArray(), ans, path);
        return ans;
    }

    // 枚举 i 个字符，如果它是字母，它可以转变也可以不转变
    private void dfs(int i, char[] chars, List<String> ans, char[] path) {
        if (i == chars.length) {
            ans.add(new String(path));
            return;
        }
        char c = chars[i];
        if (!Character.isLetter(c)) {
            path[i] = c;
            dfs(i + 1, chars, ans, path);
        } else {
            // 大写
            path[i] = Character.toUpperCase(c);
            dfs(i + 1, chars, ans, path);
            // 小写
            path[i] = Character.toLowerCase(c);
            dfs(i + 1, chars, ans, path);
        }
    }

    //    private int max = -1;
//    private int path = 0;

    public int perfectMenu(int[] materials, int[][] cookbooks, int[][] attribute, int limit) {
        /*LCP 51.烹饪料理
         * 勇者背包内共有编号为 0 ~ 4 的五种食材，其中 materials[j] 表示第 j 种食材的数量。
         * 通过这些食材可以制作若干料理，cookbooks[i][j] 表示制作第 i 种料理需要第 j 种食材的数量，
         * 而 attribute[i] = [x,y] 表示第 i 道料理的美味度 x 和饱腹感 y。
         * 在饱腹感不小于 limit 的情况下，请返回勇者可获得的最大美味度。如果无法满足饱腹感要求，则返回 -1。
         * 注意：每种料理只能制作一次。
         * materials = [3,2,4,1,2] cookbooks = [[1,1,0,1,2],[2,1,4,0,0],[3,2,4,1,0]]
         * attribute = [[3,2],[2,4],[7,6]] limit = 5
         * 思路：回溯思想题解套路：枚举第 i 种料理选不选，选的话要能选，边界limit<=0且记录最大美味度
         * */
        // 输入的角度选与不选
        //dfs(0, materials, cookbooks, attribute, limit, 0);

        // 答案视角选哪个
        //   dfs(0, materials, cookbooks, attribute, limit);
        return max;
    }

    // 答案视角选哪个，每一个节点都有可能 limit<=0
    // 下标大于等于i的都可以选做答案
    // materials = [10,10,10,10,10] cookbooks = [[1,1,1,1,1],[3,3,3,3,3],[10,10,10,10,10]]
    // attribute = [[5,5],[6,6],[10,10]] limit = 1
//    private void dfs(int i, int[] materials, int[][] cookbooks, int[][] attribute, int limit) {
//        if (limit <= 0) {
//            max = Math.max(path, max);
//        }
//        if (i == cookbooks.length) return;
//        for (int j = i; j < cookbooks.length; j++) {
//            if (isAbleCook(j, materials, cookbooks)) {
//                for (int k = 0; k < materials.length; k++) {
//                    materials[k] -= cookbooks[j][k];
//                }
//                path = path + attribute[j][0];
//                dfs(j + 1, materials, cookbooks, attribute, limit - attribute[j][1]);
//                path = path - attribute[j][0];
//                for (int k = 0; k < materials.length; k++) {
//                    materials[k] += cookbooks[j][k];
//                }
//            }
//        }
//    }

    // 选与不选的实现：枚举到第 i 种料理做不做
    // materials = [10,10,10,10,10] cookbooks = [[1,1,1,1,1],[3,3,3,3,3],[10,10,10,10,10]]
    // attribute = [[5,5],[6,6],[10,10]] limit = 1
//    private void dfs(int i, int[] materials, int[][] cookbooks, int[][] attribute, int limit, int path) {
//        if (i == cookbooks.length) {
//            // 所有料理枚举完成，记录答案
//            if (limit <= 0) {
//                // path 表示遍历过程中，产生的美味度
//                max = Math.max(path, max);
//            }
//            return;
//        }
//        // 不选
//        dfs(i + 1, materials, cookbooks, attribute, limit, path);
//
//        // 选:还要看能不能选
//        if (isAbleCook(i, materials, cookbooks)) {
//            for (int j = 0; j < materials.length; j++) {
//                materials[j] -= cookbooks[i][j];
//            }
//            dfs(i + 1, materials, cookbooks, attribute, limit - attribute[i][1], path = path + attribute[i][0]);
//            for (int j = 0; j < materials.length; j++) {
//                materials[j] += cookbooks[i][j];
//            }
//        }
//    }

    private boolean isAbleCook(int i, int[] materials, int[][] cookbooks) {
        for (int j = 0; j < materials.length; j++) {
            if (materials[j] < cookbooks[i][j]) return false;
        }
        return true;
    }

    //  private int max = 0;

    public int maximumRows(int[][] matrix, int numSelect) {
        /*2397.被列覆盖的最多行数
         *一个 m x n 的二进制矩阵，numSelect 表示从矩阵中选择的列数，要求选择出的列使集合覆盖的行数最大
         * 覆盖指的是：如果矩阵一行中含 1 的列都被选择了，则这一行被覆盖了
         * 对于每一列有选或者不选，枚举第 i 列，path 路径记录选择的列。
         * 枚举完后，判断当前选择的列能够使集合覆盖的行数
         * 全局维护最大值
         * */
        int m = matrix.length;
        if (m == 0) return 0;
        int n = matrix[0].length;
        // path[i] 表示第 i 列选不选，1表示选
        boolean[] path = new boolean[n];

        dfs(0, path, matrix, numSelect);
        return max;
    }

    private void dfs(int i, boolean[] path, int[][] matrix, int numSelect) {
        if (i == matrix[0].length) {
            // 全部列枚举完成，计算答案
            max = Math.max(max, coverRow(path, matrix));
            return;
        }
        // 不选第 i 列
        if (matrix[0].length - i > numSelect)
            dfs(i + 1, path, matrix, numSelect);

        // 选第 i 列
        if (numSelect > 0) {
            path[i] = true;
            dfs(i + 1, path, matrix, --numSelect);
            path[i] = false;
        }
    }

    private int coverRow(boolean[] path, int[][] matrix) {
        int m = matrix.length;
        int n = matrix[0].length;
        int ans = m;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (matrix[i][j] == 1 && !path[j]) {
                    ans--;
                    break;
                }
            }
        }
        return ans;
    }

    private int max = 0;

    public int maxLength(List<String> arr) {
        /*1239.串联字符串的最大长度
         * 给定一个字符串数组 arr，
         * 对于每一个元素有选或者不选来构成满足没有重复字母的字符串 s
         * 返回s的最大长度
         * 枚举第 i 个字符串，有选或不选
         * 选的话要满足已经选的字符串与当前字符串没有重复字母
         * 枚举完后，统计 s 的长度
         * 维护全局变量记录 S 的长度
         * */
        // path 记录已经选择的字符串构成的字符串
        boolean[] path = new boolean[26];
        dfs(0, arr, path);
        return max;
    }

    private void dfs(int i, List<String> arr, boolean[] path) {
        if (i == arr.size()) {
            // 枚举完毕
            max = Math.max(max, count(path));
            return;
        }

        // 不选
        dfs(i + 1, arr, path);

        // 选
        if (check(i, arr, path)) {
            increase(i, arr, path);
            dfs(i + 1, arr, path);
            decrease(i, arr, path);
        }
    }

    private void decrease(int i, List<String> arr, boolean[] path) {
        for (char c : arr.get(i).toCharArray()) {
            path[c - 'a'] = false;
        }
    }

    private void increase(int i, List<String> arr, boolean[] path) {
        for (char c : arr.get(i).toCharArray()) {
            path[c - 'a'] = true;
        }
    }

    private boolean check(int i, List<String> arr, boolean[] path) {

        char[] chars = arr.get(i).toCharArray();
        boolean[] flag = new boolean[26];
        for (char c : chars) {
            if (path[c - 'a']) {
                return false;
            }
            if (flag[c - 'a']) return false;
            else flag[c - 'a'] = true;
        }
        return true;
    }

    private int count(boolean[] path) {
        int ans = 0;
        for (boolean b : path) {
            if (b) ans++;
        }
        return ans;
    }

    private int score = 0;
    // private int[] ans = new int[12];

    public int[] maximumBobPoints(int numArrows, int[] aliceArrows) {
        int[] path = new int[12];
        dfs(0, numArrows, aliceArrows, path);
        return ans;
    }

    private void dfs(int i, int numArrows, int[] aliceArrows, int[] path) {
        if (i == aliceArrows.length) {
            // 所有得分区间都枚举完毕
            // 计算得分
            int nowScore = 0;
            for (int j = 0; j < path.length; j++) {
                if (path[j] > 0) {
                    nowScore = nowScore + j;
                }
            }
            if (nowScore > score) {
                score = nowScore;
                ans = Arrays.copyOf(path, 12);
                if (numArrows > 0)
                    ans[0] += numArrows;
            }

            return;
        }
        // 不选当前得分
        dfs(i + 1, numArrows, aliceArrows, path);

        // 选当前得分
        if (numArrows > aliceArrows[i]) {
            path[i] = aliceArrows[i] + 1;
            dfs(i + 1, numArrows - aliceArrows[i] - 1, aliceArrows, path);
            path[i] = 0;
        }
    }

    boolean flag = false;

    public int punishmentNumber(int n) {
        /*2698.求一个整数的惩罚数
         * 回溯判断 i 是否满足条件，也就是 i*i 能否分割为和为 i 的子字符串
         * */
        int ans = 0;
        for (int i = 1; i <= n; i++) {
            if (check(i)) {
                ans += i * i;
                flag = false;
            }
        }
        return ans;
    }


    private boolean check(int i) {
        int num = i * i;
        String s = String.valueOf(num);
        List<String> path = new ArrayList<>();
        dfs(0, 0, s, i, path);
        return flag;
    }

    // i 表示第 i 与 i+1 是否分割
    private void dfs(int i, int start, String s, int target, List<String> path) {
        if (flag)
            return;
        if (i == s.length()) {
            // 枚举完了
            int sum = 0;
            for (String str : path) {
                sum += Integer.parseInt(str);
            }
            if (sum == target)
                flag = true;
            return;
        }
        // 不分割
        dfs(i + 1, start, s, target, path);

        //分割
        path.add(s.substring(start, i + 1));
        dfs(i + 1, i + 1, s, target, path);
        path.removeLast();
    }

    public List<String> restoreIpAddresses(String s) {
        /*93.复原 IP 地址
         * 一共只能加3个点，枚举第i个加点的位置*/
//        List<String> ans = new ArrayList<>();
//        List<String> path = new ArrayList<>();
//        f(0, 0, s, path, ans);
//        return ans;
        // 枚举答案选哪个，也就是枚举s[i]~s[n] 怎么分割
        List<String> ans = new ArrayList<>();
        String[] path = new String[s.length()];
        dfs(0, 0, s, s.length(), path, ans);
        return ans;
    }

    private void dfs(int i, int count, String s, int len, String[] path, List<String> ans) {
        // 剪枝: 还剩下 len - i个字符，段数 4 - count，每段至少一个字符，至多3个字符
        if (len - i < 4 - count || len - i > (4 - count) * 3) {
            return;
        }
        if (i == len) {
            ans.add(String.join(".", path));
        }
        // 枚举选第 i 个
        int value = 0;
        for (int j = i; j < len; j++) {
            value = value * 10 + (s.charAt(j) - '0');
            if (value > 255)
                break;
            path[count] = s.substring(i, j + 1);
            dfs(j + 1, count + 1, s, len, path, ans);
            if (value == 0)
                break;
        }
    }

    // 枚举第 i 个点加在字符串的位置
//    private void f(int i, int start, String s, List<String> path, List<String> ans) {
//        if (i == 4) {
//            ans.add(String.join(".", path));
//            return;
//        }
//        for (int j = start; j < s.length() - 2; j++) {
//            String str;
//            if (i < 3) {
//                str = s.substring(start, j + 1);
//                if (check(str)) {
//                    path.add(str);
//                    f(i + 1, j + 1, s, path, ans);
//                    path.removeLast();
//                } else break;
//            } else {
//                str = s.substring(start);
//                if (check(str)) {
//                    path.add(str);
//                    f(i + 1, j + 1, s, path, ans);
//                    path.removeLast();
//                } else break;
//                break;
//            }
//
//        }
//
//    }
//
//    private boolean check(String str) {
//        if (str.length() > 1) {
//            if (str.startsWith("0"))
//                return false;
//            if (Long.parseLong(str) > 255)
//                return false;
//        }
//        return true;
//    }


    public List<List<Integer>> combine(int n, int k) {
        /*77.组合
        * 给定两个整数 n 和 k，返回范围 [1, n] 中所有可能的 k 个数的组合。
          你可以按 任何顺序 返回答案。
        * 从答案的角度：枚举答案集合中第 i 个数选哪个，同样[1,2] 与[2,1]为一个集合，可以规定顺序
        * 剪枝：当答案集合长度等于k时，返回；当剩余元素个数小于k-答案元素个数时，返回
        * */
//        List<List<Integer>> ans = new ArrayList<>();
//        List<Integer> path = new ArrayList<>();
//        dfs(n, k, path, ans);
//        return ans;
        // 选不选的思路，枚举当前元素选不选
        List<List<Integer>> ans = new ArrayList<>();
        List<Integer> path = new ArrayList<>();
        dfs(n, k, path, ans);
        return ans;
    }


    //枚举第 i 个元素选不选
 /*   private void dfs(int i, int k, List<Integer> path, List<List<Integer>> ans) {

        int d = k - path.size();
        if (d == 0) {
            ans.add(new ArrayList<>(path));
            return;
        }
        if (d < i) {
            // 不选
            dfs(i - 1, k, path, ans);
        }

        // 选
        path.add(i);
        dfs(i - 1, k, path, ans);
        path.removeLast();
    }*/

//    private void dfs(int n, int k, List<Integer> path, List<List<Integer>> ans) {
//        int d = k - path.size();
//        // 记录答案
//        if (d == 0) {
//            ans.add(new ArrayList<>(path));
//            return;
//        }
//        // 剪枝
//        // 枚举第 1、2、.... k 个答案元素选哪个
//        for (int i = n; i >= d; i--) {
//            path.add(i);
//            dfs(i - 1, k, path, ans);
//            path.removeLast();
//        }
//
//
//    }

    public List<List<Integer>> combinationSum3(int k, int n) {
        /*216.组合总和 Ⅲ
        * 找出所有相加之和为 n 的 k 个数的组合，且满足下列条件：
        * 只使用数字1到9 每个数字最多使用一次
          返回所有可能的有效组合的列表 。该列表不能包含相同的组合两次，组合可以以任何顺序返回。
          枚举
          * 答案选哪个
          * 剪枝：当前和大于 n 时，直接返回
          * n 大于 d 个最大的数的和时，直接返回

        * */
//        List<List<Integer>> ans = new ArrayList<>();
//        List<Integer> path = new ArrayList<>();
//        dfs(9, n, k, path, ans);
//        return ans;
        // 枚举选不选
        List<List<Integer>> ans = new ArrayList<>();
        List<Integer> path = new ArrayList<>();
        dfs(9, n, k, path, ans);
        return ans;
    }

    // 枚举选不选
    private void dfs(int i, int target, int k, List<Integer> path, List<List<Integer>> ans) {
        int d = k - path.size();
        if (target < 0 | target > (2 * i + 1 - d) * d / 2)
            return;
        if (d == 0) {
            ans.add(new ArrayList<>(path));
            return;
        }

        // 不选
        if (i > d) {
            dfs(i - 1, target, k, path, ans);
        }

        // 选
        path.add(i);
        dfs(i - 1, target - i, k, path, ans);
        path.removeLast();
    }

//    private void dfs(int i, int target, int k, List<Integer> path, List<List<Integer>> ans) {
//        int d = k - path.size();
//        if (target < 0 | target > (2 * i + 1 - d) * d / 2)
//            return;
//        if (d == 0) {
//            ans.add(new ArrayList<>(path));
//            return;
//        }
//        for (int j = i; j >= d; j--) {
//            path.add(j);
//            dfs(j - 1, target - j, k, path, ans);
//            path.removeLast();
//        }
//
//    }

    public List<String> generateParenthesis(int n) {
        /*22.括号生成
         * 数字 n 代表生成括号的对数，请你设计一个函数，用于能够生成所有可能的并且 有效的 括号组合。
         * 思路：一共 2n 个括号，枚举第 i 个选不选，注意是有条件的选择：
         * 左括号必须大于等于右括号的个数
         * 左右括号数各自都需要小于n
         * 左右括号数相等时，只能选左括号
         * */
        List<String> ans = new ArrayList<>();
        char[] path = new char[n * 2];
        dfs(0, 0, n, path, ans);
        return ans;
    }

    private void dfs(int left, int right, int n, char[] path, List<String> ans) {
        if (right == n) {
            ans.add(new String(path));
            return;
        }
        // 选左
        if (left < n) {
            path[left + right] = '(';
            dfs(left, right + 1, n, path, ans);
        }

        // 选右
        if (left > right) {
            path[left + right] = ')';
            dfs(left + 1, right, n, path, ans);
        }
    }


    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        /*39.组合总和
         * 无重复数组中找出和为 target 的不同组合，可以重复选取数组中的元素
         * 2 <= candidates[i] <= 40
         * 枚举答案选哪个，注意可以重复选择
         * */
        Arrays.sort(candidates);
        List<List<Integer>> ans = new ArrayList<>(150);
        List<Integer> path = new ArrayList<>();
        dfs(0, target, candidates, path, ans);
        return ans;

    }

    // 第 i 个元素选或者不选
    private void dfs(int i, int target, int[] candidates, List<Integer> path, List<List<Integer>> ans) {
        if (target == 0) {
            ans.add(new ArrayList<>(path));
            return;
        }
        if (i == candidates.length || target < candidates[i])
            return;
        // 不选
        dfs(i + 1, target, candidates, path, ans);

        // 选
        path.add(candidates[i]);
        dfs(i, target - candidates[i], candidates, path, ans);
        path.removeLast();

    }


    //    private void dfs(int i, int target, int[] candidates, List<Integer> path, List<List<Integer>> ans) {
//        if (target < 0) {
//            return;
//        }
//
//        if (target == 0) {
//            ans.add(new ArrayList<>(path));
//            return;
//        }
//        // 枚举答案选哪个
//        for (int j = i; j < candidates.length; j++) {
//            path.add(candidates[j]);
//            dfs(j, target - candidates[j], candidates, path, ans);
//            path.removeLast();
//        }
//    }
    public List<List<Integer>> permute(int[] nums) {
        /*46.全排列
         * 给定一个不含重复数字的数组 nums ，返回其所有可能的全排列 。你可以按任意顺序返回答案。
         * 思路：[1,2] 与 [2,1] 不同，也就是枚举选过还能选，怎么实现？
         * 从元素选不选的角度不好实现：想象构造答案的搜索树，并不是每一个节点都有选或者不选两种情况，
         * 存在大量特殊情况，例如选了 1，后面只能选2，没选1，后面只能选2然后选1。
         * 从答案选哪个的角度好思考，需要维护一个数组，表示哪些元素能选，每次从能选的元素中选择一个元素
         * */
        int n = nums.length;
        List<List<Integer>> ans = new ArrayList<>();
        List<Integer> path = Arrays.asList(new Integer[n]);
        boolean[] onPath = new boolean[n];
        dfs(0, nums, onPath, path, ans);
        return ans;
    }

    // 枚举选哪个
    private void dfs(int i, int[] nums, boolean[] onPath, List<Integer> path, List<List<Integer>> ans) {
        if (i == nums.length) {
            ans.add(new ArrayList<>(path));
            return;
        }
        // 枚举第 i 个答案选第 j 个元素
        for (int j = 0; j < nums.length; j++) {
            if (!onPath[j]) {
                path.set(i, nums[j]);
                onPath[j] = true;
                dfs(i + 1, nums, onPath, path, ans);
                onPath[j] = false;
            }
        }

    }


    public List<List<String>> solveNQueens(int n) {
        /*51.N 皇后
         * 将n个皇后放在 nxn 的棋盘上，要求：皇后之间不同行不同列
         * 皇后不能位于同一斜线上
         * 返回所有不同解决方案
         * 每一种解法包含一个不同的 n 皇后问题 的棋子放置方案，该方案中 'Q' 和 '.' 分别代表了皇后和空位。
         * 思路：复杂问题的拆分：
         * 首先皇后位于不同行不同列，也就是每行每列只有一个皇后
         * 等价与数组 col[i] 表示第 i 行的皇后位于第 col[i] 列的全排列
         * 例如[1,3,0,2] 表示第 0 行的皇后在第1列，第 1行的皇后在第3列...
         * 皇后不能位于同一斜线上：
         * 等价于从上往下枚举，当前枚举到的行与列要放置皇后要求左上、右上方向上没有皇后
         * 等价于行号-列号 = 当前行号 - 当前列号；行号+列号 = 当前行号+当前列号
         * 将复杂问题的条件逐步拆分为熟悉的问题与解决方案，
         * 首先书写全排列，再在全排列的基础上筛选不同斜线
         *
         * */

        List<List<String>> ans = new ArrayList<>();
        boolean[] col = new boolean[n];
        int[] queens = new int[n];
        boolean[] diag1 = new boolean[2 * n - 1];
        boolean[] diag2 = new boolean[2 * n - 1];
        dfs(0, queens, col, diag1, diag2, ans);
        return ans;
    }

    private void dfs(int row, int[] queens, boolean[] col, boolean[] diag1, boolean[] diag2, List<List<String>> ans) {
        int n = queens.length;
        if (row == n) {
            // 构造答案
            List<String> path = new ArrayList<>(n);
            for (int i : queens) {
                char[] ansRow = new char[n];
                Arrays.fill(ansRow, '.');
                ansRow[i] = 'Q';
                path.add(new String(ansRow));
            }
            ans.add(path);
            return;
        }
        for (int c = 0; c < n; c++) {
            int rc = row - c + n - 1;
            if (!col[c] && !diag1[c + row] && !diag2[rc]) {
                // 第 i 列没有被选过
                // 合法
                queens[row] = c;
                col[c] = diag1[row + c] = diag2[rc] = true;
                dfs(row + 1, queens, col, diag1, diag2, ans);
                col[c] = diag1[row + c] = diag2[rc] = false;
            }

        }


    }

    //  private int ans = 0;

    public int totalNQueens(int n) {
        /*52.N 皇后Ⅱ
         * 给你一个整数 n ，返回 n 皇后问题 不同的解决方案的数量
         * */
        boolean[] col = new boolean[n];
        boolean[] diag1 = new boolean[n * 2 - 1];
        boolean[] diag2 = new boolean[n * 2 - 1];
        dfs(0, col, diag1, diag2);
        return ans;
    }

    private void dfs(int row, boolean[] col, boolean[] diag1, boolean[] diag2) {
        int n = col.length;
        if (row == n) {
            ans++;
            return;
        }
        // [r,c] 放皇后
        for (int c = 0; c < n; c++) {
            int rc = row - c + n - 1;
            if (!col[c] && !diag1[row + c] && !diag2[rc]) {
                // 合法可以放皇后
                col[c] = diag1[row + c] = diag2[rc] = true;
                dfs(row + 1, col, diag1, diag2);
                col[c] = diag1[row + c] = diag2[rc] = false;
            }
        }
    }


    public int countNumbersWithUniqueDigits(int n) {
        /*357.统计各位数字都不同的数字个数
         * 给你一个整数 n ，统计并返回各位数字都不同的数字 x 的个数，其中 0 <= x < 10^n 。
         * 枚举当前答案选哪个
         * */
        if (n == 0) return 1;
        boolean[] flag = new boolean[10];

        return dfs(0, n, flag);
    }

    // 枚举第 i 个数选哪个
    private int dfs(int i, int n, boolean[] flag) {
        int count = 0;
        if (i != n) {
            // 第 i 个数选j
            for (int j = 0; j < 10; j++) {
                // 剪枝：多位数时，第一位不为0
                if (j == 0 && n > 1 && i == 1) {
                    continue;
                }
                // 不能用用过的数字
                if (flag[j])
                    continue;
                flag[j] = true;
                count = count + dfs(i + 1, n, flag) + 1;
                flag[j] = false;
            }
        }
        return count;
    }

}

