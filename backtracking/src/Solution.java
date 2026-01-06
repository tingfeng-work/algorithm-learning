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
        dfs(0, n, ans, path, digits, map);
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
    private void dfs(int i, int[] nums, List<Integer> path, List<List<Integer>> ans) {
        if (i == nums.length) {
            ans.add(new ArrayList<>(path));
            return;
        }

        //不选
        dfs(i + 1, nums, path, ans);
        // 选
        path.add(nums[i]);
        dfs(i + 1, nums, path, ans);
        path.removeLast();
    }

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
}
