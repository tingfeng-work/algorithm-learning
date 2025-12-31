import java.util.*;
import java.util.function.Function;

public class Solution {


    public int maxDepth(TreeNode root) {
        /*104.二叉树的最大深度
        给定一个二叉树 root ，返回其最大深度。
        * */
        if (root == null) {
            return 0;
        }
        return Math.max(maxDepth(root.left), maxDepth(root.right)) + 1;
    }


    public int minDepth(TreeNode root) {
        /* 111. 二叉树的最小深度
         *  给定一个二叉树，找出其最小深度。
         *  原问题：找二叉树的最小深度
         *  子问题：左/右子树的最小深度
         * */
        if (root == null)
            return 0;
        if (root.left == null) {
            return minDepth(root.right) + 1;
        }
        if (root.right == null) {
            return minDepth(root.left) + 1;
        }
        return Math.min(minDepth(root.left), minDepth(root.right)) + 1;

    }

    public int sumOfLeftLeaves(TreeNode root) {
        /*404.左叶子之和
        给定二叉树的根节点 root ，返回所有左叶子之和。
        子问题：左/右子树的的左叶子之和
        边界条件：叶子节点返回
        * */
        if (root == null) {
            return 0;
        }
        // 深度优先
        int sum = sumOfLeftLeaves(root.left) + sumOfLeftLeaves(root.right);

        TreeNode left = root.left;
        if (left != null && left.left == null && left.right == null) {
            //左叶子记录答案
            sum = sum + left.val;
        }
        return sum;


    }

    public boolean hasPathSum(TreeNode root, int targetSum) {
        /*112. 路经总和
         *  判断该树中是否存在根节点到叶子节点的路径，
         *  这条路径上所有节点值相加等于目标和 targetSum
         *  子问题：是否存在左/右子树是否路径上所有节点值等于 targetSum-root.val
         * */
        // 边界条件
        if (root == null)
            return false;
        targetSum = targetSum - root.val;
        if (root.left == null && root.right == null) {
            return targetSum == 0;
        }
        //非边界条件
        return hasPathSum(root.left, targetSum)
                || hasPathSum(root.right, targetSum);
    }

    private int ans = 0;

    public int sumNumbers(TreeNode root) {
        /*129.求根节点到叶节点数字和
        *   给你一个二叉树的根节点 root ，树中每个节点都存放有一个 0 到 9 之间的数字。
            每条从根节点到叶节点的路径都代表一个数字：
            例如，从根节点到叶节点的路径 1 -> 2 -> 3 表示数字 123。
            计算从根节点到叶节点生成的 所有数字之和 。
        *
        * */
        dfs(root, 0);
        return ans;

    }

    //sumNumbers dfs
/*    private void dfs(TreeNode node, int val) {
        if (node == null)
            return;
        val = val * 10 + node.val;
        if (node.left == null && node.right == null) {
            //叶子节点
            ans = ans + val;
            return;
        }
        dfs(node.right, val);
        dfs(node.left, val);

    }*/

    public int goodNodes(TreeNode root) {

        /*返回二叉树中好节点的数目。
        「好节点」X 定义为：从根到该节点 X 所经过的节点中，没有任何节点的值大于 X 的值。
        *
        * */
        dfs(root, Integer.MIN_VALUE);
        return ans;
    }


    private void dfs(TreeNode root, int val) {
        if (root == null)
            return;
        if (root.val >= val) {
            ans++;
            val = root.val;
        }
        dfs(root.right, val);
        dfs(root.left, val);
    }

    private Map<Integer, List<int[]>> map = new TreeMap<>();


    public List<List<Integer>> verticalTraversal(TreeNode root) {
        /* 987. 二叉树的垂序遍历
         *
         * */
        dfs(root, 0, 0);
        List<List<Integer>> ans = new ArrayList<>(map.size());
        for (List<int[]> group : map.values()) {
            group.sort((a, b) -> a[0] != b[0] ? a[0] - b[0] : a[1] - b[1]);
            List<Integer> vals = new ArrayList<>(group.size());
            for (int[] ints : group) {
                vals.add(ints[1]);
            }
            ans.add(vals);
        }
        return ans;
    }

    private void dfs(TreeNode node, int row, int col) {
        if (node == null)
            return;
        map.computeIfAbsent(col, i -> new ArrayList<>()).add(new int[]{row, node.val});
        dfs(node.left, row + 1, col - 1);
        dfs(node.right, row + 1, col + 1);
    }


}
