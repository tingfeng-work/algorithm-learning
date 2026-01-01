import java.util.*;
import java.util.function.BiFunction;

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

    //private int ans = 0;

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


//    private void dfs(TreeNode root, int val) {
//        if (root == null)
//            return;
//        if (root.val >= val) {
//            ans++;
//            val = root.val;
//        }
//        dfs(root.right, val);
//        dfs(root.left, val);
//    }

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

//    private void dfs(TreeNode node, int row, int col) {
//        if (node == null)
//            return;
//        map.computeIfAbsent(col, i -> new ArrayList<>()).add(new int[]{row, node.val});
//        dfs(node.left, row + 1, col - 1);
//        dfs(node.right, row + 1, col + 1);
//    }

    private boolean isSameTree(TreeNode p, TreeNode q) {
        /* 100. 相同的树
            判断两棵树是否相同，相同指结构相同，节点有相同的值
            子问题：判断当前节点值是否相同，以及当前节点的左右子树是否相同
        * */
        if (p == null || q == null) {
            return p == q;
        }
        return p.val == q.val && isSameTree(p.left, q.right) && isSameTree(p.right, q.left);


    }

    public boolean isSymmetric(TreeNode root) {
        /* 101.对称二叉树
           判断一棵树是否轴对称
           其实是判断左子树与右子树是否镜像相等
        *
        * */
        if (root == null) {
            return true;
        }
        return isSameTree(root.left, root.right);

    }

    public boolean isBalanced(TreeNode root) {
        /*110.平衡二叉树
         * 判断一棵树是否是平衡二叉树
         * 思路：判断是否平衡 = 判断左右子树深度差是否小于等于 1
         *
         * */
        return getHeight(root) != -1;
    }

//    private int getHeight(TreeNode node) {
//        if (node == null)
//            return 0;
//        int left = getHeight(node.left);
//        if (left == -1)
//            return -1;
//        int right = getHeight(node.right);
//        if (right == -1 || Math.abs(left - right) > 1)
//            return -1;
//        return Math.max(left, right) + 1;
//
//    }

    public List<Integer> rightSideView(TreeNode root) {
        /*199.二叉树的右视图
         * 给定一个二叉树的根节点 root，想象自己站在它的右侧，
         * 按照从顶部到底部的顺序，返回从右侧所能看到的节点值
         * 思路：从右子树遍历走，往下遍历的过程中，判断当前节点的深度与答案的长度是否相等
         * 相等，说明是第一个看到的，需要加入答案
         * */
        List<Integer> ans = new ArrayList<>();
        rightDFS(root, 0, ans);
        return ans;
    }

    private void rightDFS(TreeNode node, int depth, List<Integer> ans) {
        if (node == null)
            return;
        //先记录再遍历
        if (ans.size() == depth)
            ans.add(node.val);
        // 优先遍历右子树
        rightDFS(node.right, depth + 1, ans);
        rightDFS(node.left, depth + 1, ans);
    }

    public boolean isUnivalTree(TreeNode root) {
        /*965.单值二叉树
         * 如果二叉树每个节点都具有相同的值，那么该二叉树就是单值二叉树。
         *
         * */
        if (root == null)
            return true;
        int left = root.left == null ? root.val : root.left.val;
        int right = root.right == null ? root.val : root.right.val;
        return (left == right && left == root.val) && isUnivalTree(root.left) && isUnivalTree(root.right);


    }

    public boolean flipEquiv(TreeNode root1, TreeNode root2) {
        /*951.翻转等价二叉树
          为二叉树 T 定义一个 翻转操作 ，如下所示：选择任意节点，然后交换它的左子树和右子树。
          只要经过一定次数的翻转操作后，能使 X 等于 Y，我们就称二叉树 X 翻转等价于二叉树 Y。
          思路：对于 root1 与 root2 其左右孩子有两种情况返回 true：相等、对称，否则返回false
                对于相等的两个节点，递归遍历其左右节点
        *
        * */
        if (root1 == null || root2 == null) {
            return root1 == root2;
        }
        if (root1.val != root2.val)
            return false;
        return (flipEquiv(root1.left, root2.left) && flipEquiv(root1.right, root2.right))
                || (flipEquiv(root1.left, root2.right) && flipEquiv(root1.right, root2.left));
    }


    public TreeNode invertTree(TreeNode root) {
        /*226. 反转二叉树
         * 给你一棵二叉树的根节点 root ，翻转这棵二叉树，并返回其根节点。
         * 反转一个树就是交换左右子树
         * 对于一个节点，交换左右子树后，递归交换左子节点的左右子树与右子节点的左右子树
         *
         * */
        if (root == null)
            return null;
        TreeNode temp = root.left;
        root.left = root.right;
        root.right = temp;
        invertTree(root.right);
        invertTree(root.left);
        return root;

    }

    public TreeNode mergeTrees(TreeNode root1, TreeNode root2) {
        /*617.合并二叉树
         * 你需要将这两棵树合并成一棵新二叉树。合并的规则是：
         * 如果两个节点重叠，那么将这两个节点的值相加作为合并后节点的新值；
         * 否则，不为 null 的节点将直接作为新二叉树的节点。
         * 返回合并后的二叉树。
         * 注意: 合并过程必须从两个树的根节点开始。
         * 思路：相当于把一棵树加到另一颗树上
         * 对于 node1、node2，合并其值，并递归合并左节点的的值，右节点的值
         * 边界条件：两个节点都为空
         * */
        if (root1 == null && root2 == null)
            return null;
        if (root1 == null) {
            root1 = new TreeNode(0);
        } else if (root2 == null) {
            root2 = new TreeNode(0);
        }
        // 合并值
        root1.val = root1.val + root2.val;
        root1.left = mergeTrees(root1.left, root2.left);
        root1.right = mergeTrees(root1.right, root2.right);
        return root1;
    }

    public boolean evaluateTree(TreeNode root) {
        /* 2331. 计算布尔二叉树的值
         *  给你一棵 完整二叉树 的根，这棵树有以下特征：
         *  叶子节点 要么值为 0 要么值为 1 ，其中 0 表示 False ，1 表示 True 。
         *  非叶子节点 要么值为 2 要么值为 3 ，其中 2 表示逻辑或 OR ，3 表示逻辑与 AND 。
         *  计算一个节点值的方式是：叶子节点的值就为其值本身，
         * 非叶子节点的值等于对左右孩子的值做该节点运算符的运算
         * 完整二叉树保证每个节点要么没有孩子，要么有两个孩子
         * 思路：对于每个节点，如果是叶子节点直接返回val
         * 对于非叶子节点递归计算左右子树的布尔值，做运算返回
         * */
        if (root.left == null && root.right == null) {
            //叶子节点 1:true,0:false
            return root.val == 1;
        }
        //非叶子节点
        boolean left = evaluateTree(root.left);
        boolean right = evaluateTree(root.right);
        if (root.val == 2) {
            return left || right;
        } else
            return left && right;
    }

    private int max = 0;

    public int[] findFrequentTreeSum(TreeNode root) {
        /*508. 出现次数最多的子树元素和
         * 请返回出现次数最多的子树元素和。如果有多个元素出现的次数相同，返回所有出现次数最多的子树元素和（不限顺序）。
         * 一个结点的 「子树元素和」 定义为以该结点为根的二叉树上所有结点的元素之和（包括结点本身）。
         * 思路：遍历二叉树。同时记录子树元素和以及出现次数，遍历结束后再添加答案
         * 求子树元素和 = 求左子树元素和 + 右子树元素和 + 根节点
         * */
        Map<Integer, Integer> map = new HashMap<>();
        dfs(root, map);
        List<Integer> list = new ArrayList<>(map.size());
        for (Integer i : map.keySet()) {
            if (map.get(i) == max) list.add(i);
        }
        int[] ans = new int[list.size()];
        for (int i = 0; i < list.size(); i++) {
            ans[i] = list.get(i);
        }
        return ans;
    }

    private int dfs(TreeNode node, Map<Integer, Integer> map) {
        if (node == null)
            return 0;
        int sum = node.val + dfs(node.left, map) + dfs(node.right, map);
        map.merge(sum, 1, Integer::sum);
        max = Math.max(max, map.get(sum));
        return sum;
    }

    private int ans;

    public int maxAncestorDiff(TreeNode root) {
        /*1026.节点与其祖先之间的最大差值
         * 给定二叉树的根节点 root，找出存在于不同节点 A 和 B 之间的最大值 V，其中 V = |A.val - B.val|，且 A 是 B 的祖先。
         * 思路：遍历时，将当前路径的最大值传下去，同时维护全局变量答案
         * 原问题：基于当前节点的树最大差值
         * 子问题：基于左右子节点的树最大差值
         * */
        dfs(root);
        return ans;
    }

//    private int[] dfs(TreeNode node) {
//        if (node == null)
//            return new int[]{Integer.MAX_VALUE, Integer.MIN_VALUE};
//        int[] left = dfs(node.left);
//        int[] right = dfs(node.right);
//
//        int min = Math.min(node.val, Math.min(left[0], right[0]));
//        int max = Math.max(node.val, Math.max(left[1], right[1]));
//        ans = Math.max(ans, Math.max(node.val - min, max - node.val));
//        return new int[]{min, max};
//
//    }

    public int longestZigZag(TreeNode node) {
        /*1372. 二叉树中最长交错路径
         * 交错路径：左右交替向下遍历，路径长度为节点个数减 1
         * 思路：自顶向下：枚举当前节点能向下走的深度，再枚举左孩子向下走的深度、右孩子向下走的深度
         * 遍历过程中需要知道该节点是右孩子还是左孩子
         * */
        getHeight(node, 0, true);
        return ans;
    }


    private void getHeight(TreeNode node, int len, boolean isLeft) {
        if (node == null)
            return;
        ans = Math.max(len, ans);
        getHeight(node.left, isLeft ? 1 : len + 1, true);
        getHeight(node.right, isLeft ? len + 1 : 1, false);
    }

    public TreeNode sufficientSubset(TreeNode root, int limit) {
        /*1080. 根到叶路径上的不足节点
         * 给你二叉树的根节点 root 和一个整数 limit ，
         * 请你同时删除树中所有不足节点 ，并返回最终二叉树的根节点。
         * 假如通过节点 node 的每种可能的 “根-叶” 路径上值的总和全都小于给定的 limit，则该节点被称之为不足节点 ，需要被删除。
         * */

        return dfs(root, limit, 0);
    }

    private TreeNode dfs(TreeNode root, int limit, int pathSum) {

        pathSum = pathSum + root.val;
        if (root.left == null && root.right == null) {
            //叶子
            return pathSum >= limit ? root : null;
        }
        if (root.left != null) root.left = dfs(root.left, limit, pathSum);
        if (root.right != null) root.right = dfs(root.right, limit, pathSum);

        return root.left == null && root.right == null ? null : root;

    }


}
