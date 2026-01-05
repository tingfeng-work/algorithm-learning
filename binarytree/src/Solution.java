
import java.lang.annotation.Target;
import java.nio.channels.NonReadableChannelException;
import java.security.cert.PKIXCertPathBuilderResult;
import java.util.*;
import java.util.function.BiFunction;
import java.util.jar.JarEntry;

public class Solution {


    public int maxDepth(TreeNode root) {
        /*104.二叉树的最大深度
        给定一个二叉树 root ，返回其最大深度。
        * */
//        if (root == null) {
//            return 0;
//        }
//        return Math.max(maxDepth(root.left), maxDepth(root.right)) + 1;
        //层序遍历实现
        return levelOrder(root).size();
    }


    public int minDepth(TreeNode root) {
        /* 111. 二叉树的最小深度
         *  给定一个二叉树，找出其最小深度。
         *  原问题：找二叉树的最小深度
         *  子问题：左/右子树的最小深度
         * */
//        if (root == null)
//            return 0;
//        if (root.left == null) {
//            return minDepth(root.right) + 1;
//        }
//        if (root.right == null) {
//            return minDepth(root.left) + 1;
//        }
//        return Math.min(minDepth(root.left), minDepth(root.right)) + 1;
        //层序遍历实现
        /*offer/poll 在失败时返回特殊值（false / null），更适合业务控制流；
        add/remove 在失败时抛异常（IllegalStateException / NoSuchElementException），
        更适合“失败即错误”的场景。
        * */
        if (root == null)
            return 0;
        Queue<TreeNode> q = new LinkedList<>();
        q.offer(root);
        int ans = Integer.MAX_VALUE;
        int depth = 0;
        while (!q.isEmpty()) {
            int len = q.size();
            depth++;
            while (len-- > 0) {
                TreeNode node = q.poll();
                if (node.left == null && node.right == null) {
                    ans = Math.min(ans, depth);
                }
                if (node.left != null)
                    q.offer(node.left);
                if (node.right != null) {
                    q.offer(node.right);
                }

            }
        }
        return ans;

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

//    public int sumNumbers(TreeNode root) {
//        /*129.求根节点到叶节点数字和
//        *   给你一个二叉树的根节点 root ，树中每个节点都存放有一个 0 到 9 之间的数字。
//            每条从根节点到叶节点的路径都代表一个数字：
//            例如，从根节点到叶节点的路径 1 -> 2 -> 3 表示数字 123。
//            计算从根节点到叶节点生成的 所有数字之和 。
//        *
//        * */
//        dfs(root, 0);
//        return ans;
//
//    }

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

//    public int goodNodes(TreeNode root) {
//
//        /*返回二叉树中好节点的数目。
//        「好节点」X 定义为：从根到该节点 X 所经过的节点中，没有任何节点的值大于 X 的值。
//        *
//        * */
//        dfs(root, Integer.MIN_VALUE);
//        return ans;
//    }


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


//    public List<List<Integer>> verticalTraversal(TreeNode root) {
//        /* 987. 二叉树的垂序遍历
//         *
//         * */
//        dfs(root, 0, 0);
//        List<List<Integer>> ans = new ArrayList<>(map.size());
//        for (List<int[]> group : map.values()) {
//            group.sort((a, b) -> a[0] != b[0] ? a[0] - b[0] : a[1] - b[1]);
//            List<Integer> vals = new ArrayList<>(group.size());
//            for (int[] ints : group) {
//                vals.add(ints[1]);
//            }
//            ans.add(vals);
//        }
//        return ans;
//    }

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

//    public boolean isBalanced(TreeNode root) {
//        /*110.平衡二叉树
//         * 判断一棵树是否是平衡二叉树
//         * 思路：判断是否平衡 = 判断左右子树深度差是否小于等于 1
//         *
//         * */
//        return getHeight(root) != -1;
//    }

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
//        List<Integer> ans = new ArrayList<>();
//        rightDFS(root, 0, ans);
//        return ans;
        // 层序遍历实现
        if (root == null)
            return List.of();

        List<Integer> ans = new ArrayList<>();
        Queue<TreeNode> q = new LinkedList<>();
        q.offer(root);
        while (!q.isEmpty()) {
            int len = q.size();
            for (int i = 0; i < len; i++) {
                TreeNode node = q.poll();
                if (i == 0) {
                    ans.add(node.val);
                }
                if (node.right != null) q.offer(node.right);
                if (node.left != null) q.offer(node.left);
            }
        }
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

    //private int max = 0;

//    public int[] findFrequentTreeSum(TreeNode root) {
//        /*508. 出现次数最多的子树元素和
//         * 请返回出现次数最多的子树元素和。如果有多个元素出现的次数相同，返回所有出现次数最多的子树元素和（不限顺序）。
//         * 一个结点的 「子树元素和」 定义为以该结点为根的二叉树上所有结点的元素之和（包括结点本身）。
//         * 思路：遍历二叉树。同时记录子树元素和以及出现次数，遍历结束后再添加答案
//         * 求子树元素和 = 求左子树元素和 + 右子树元素和 + 根节点
//         * */
//        Map<Integer, Integer> map = new HashMap<>();
//        dfs(root, map);
//        List<Integer> list = new ArrayList<>(map.size());
//        for (Integer i : map.keySet()) {
//            if (map.get(i) == max) list.add(i);
//        }
//        int[] ans = new int[list.size()];
//        for (int i = 0; i < list.size(); i++) {
//            ans[i] = list.get(i);
//        }
//        return ans;
//    }

//    private int dfs(TreeNode node, Map<Integer, Integer> map) {
//        if (node == null)
//            return 0;
//        int sum = node.val + dfs(node.left, map) + dfs(node.right, map);
//        map.merge(sum, 1, Integer::sum);
//        max = Math.max(max, map.get(sum));
//        return sum;
//    }

    //private int ans;

//    public int maxAncestorDiff(TreeNode root) {
//        /*1026.节点与其祖先之间的最大差值
//         * 给定二叉树的根节点 root，找出存在于不同节点 A 和 B 之间的最大值 V，其中 V = |A.val - B.val|，且 A 是 B 的祖先。
//         * 思路：遍历时，将当前路径的最大值传下去，同时维护全局变量答案
//         * 原问题：基于当前节点的树最大差值
//         * 子问题：基于左右子节点的树最大差值
//         * */
//        dfs(root);
//        return ans;
//    }

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
//
//    public int longestZigZag(TreeNode node) {
//        /*1372. 二叉树中最长交错路径
//         * 交错路径：左右交替向下遍历，路径长度为节点个数减 1
//         * 思路：自顶向下：枚举当前节点能向下走的深度，再枚举左孩子向下走的深度、右孩子向下走的深度
//         * 遍历过程中需要知道该节点是右孩子还是左孩子
//         * */
//        getHeight(node, 0, true);
//        return ans;
//    }


//    private void getHeight(TreeNode node, int len, boolean isLeft) {
//        if (node == null)
//            return;
//        ans = Math.max(len, ans);
//        getHeight(node.left, isLeft ? 1 : len + 1, true);
//        getHeight(node.right, isLeft ? len + 1 : 1, false);
//    }

//    public TreeNode sufficientSubset(TreeNode root, int limit) {
//        /*1080. 根到叶路径上的不足节点
//         * 给你二叉树的根节点 root 和一个整数 limit ，
//         * 请你同时删除树中所有不足节点 ，并返回最终二叉树的根节点。
//         * 假如通过节点 node 的每种可能的 “根-叶” 路径上值的总和全都小于给定的 limit，则该节点被称之为不足节点 ，需要被删除。
//         * */
//
//        return dfs(root, limit, 0);
//    }

//    private TreeNode dfs(TreeNode root, int limit, int pathSum) {
//
//        pathSum = pathSum + root.val;
//        if (root.left == null && root.right == null) {
//            //叶子
//            return pathSum >= limit ? root : null;
//        }
//        if (root.left != null) root.left = dfs(root.left, limit, pathSum);
//        if (root.right != null) root.right = dfs(root.right, limit, pathSum);
//
//        return root.left == null && root.right == null ? null : root;
//
//    }

    //private Long pre = Long.MIN_VALUE;

    public boolean isValidBST(TreeNode root) {
        /*98. 验证二叉搜索树
         * 先序、中序、后序遍历分别验证二叉搜索树
         * 先序：思路对于当前节点它的值要满足：大于最小值，小于最大值，并且左右子树都要满足，向下递的时候更改最值
         * 中序：严格递增，当前节点要大于之前节点的值
         * 后序：判断当前节点值是否大于左子树归上来的最大值，是否小于右子树归上来的最小值，
         * 再递归判断左右子树:抓住关键点当前节点值不在左右子树归上来的访问内，抓住这个点了就好写边界条件了
         * */
        return f(root)[1] != Long.MAX_VALUE;

    }

    private long[] f(TreeNode root) {
        if (root == null)
            return new long[]{Long.MAX_VALUE, Long.MIN_VALUE};
        long[] left = f(root.left);
        long[] right = f(root.right);
        long val = root.val;
        if (val > left[1] && val < right[0]) {
            //当前节点符合要求
            return new long[]{Math.min(val, left[0]), Math.max(val, right[1])};
        }
        //不符合要求
        return new long[]{Long.MIN_VALUE, Long.MAX_VALUE};
    }

    public TreeNode searchBST(TreeNode root, int val) {
        /*700.二叉搜索树中的搜索
        * 给定二叉搜索树（BST）的根节点 root 和一个整数值 val。
          你需要在 BST 中找到节点值等于 val 的节点。
          * 返回以该节点为根的子树。 如果节点不存在，则返回 null 。
        * */
        if (root == null)
            return null;

        TreeNode left = searchBST(root.left, val);
        if (left != null && left.val == val)
            return left;

        TreeNode right = searchBST(root.right, val);
        if (right != null && right.val == val)
            return right;

        return root.val == val ? root : null;

    }

    public int rangeSumBST(TreeNode root, int low, int high) {
        /*938.二叉搜索树的范围和
         * 给定二叉搜索树的根结点 root，返回值位于范围 [low, high] 之间的所有结点的值的和。
         * */
        if (root == null) {
            return 0;
        }
        if (root.val < low) {
            return rangeSumBST(root.right, low, high);
        } else if (root.val > high)
            return rangeSumBST(root.left, low, high);
        return rangeSumBST(root.left, low, high) + rangeSumBST(root.right, low, high) + root.val;


    }

//    private int ans = Integer.MAX_VALUE;
//    private int pre = Integer.MIN_VALUE / 2;

//    public int getMinimumDifference(TreeNode root) {
//        /*530.二叉搜索树的最小绝对差
//         * 给你一个二叉搜索树的根节点 root ，返回 树中任意两不同节点值之间的最小差值 。
//         * 差值是一个正数，其数值等于两值之差的绝对值。
//         * 思路：中序遍历：得到严格递增的数组
//         * 答案只能产生在相邻数组元素之中
//         * 进阶：再遍历的时候就更新答案
//         * */
//        dfs(root);
//        return ans;
//    }

//    private void dfs(TreeNode root) {
//        if (root == null)
//            return;
//        dfs(root.left);
//        ans = Math.min(ans, root.val - pre);
//        pre = root.val;
//        dfs(root.right);
//    }

//    private int pre = Integer.MIN_VALUE;
//    private int max = Integer.MIN_VALUE;

//    public List<List<Integer>> closestNodes(TreeNode root, List<Integer> queries) {
//        /*2476.二叉搜索树最近节点查询
//        * 给你一个 二叉搜索树 的根节点 root ，和一个由正整数组成、长度为 n 的数组 queries 。
//        * 请你找出一个长度为 n 的 二维 答案数组 answer ，其中 answer[i] = [mini, maxi] ：
//        * mini 是树中小于等于 queries[i] 的 最大值 。如果不存在这样的值，则使用 -1 代替。
//          maxi 是树中大于等于 queries[i] 的 最小值 。如果不存在这样的值，则使用 -1 代替。
//          * 思路：先把他排成有序数组，然后用二分查找搜索目标值
//          * 进阶：在中序遍历时来找答案。
//            思路：中序遍历树，记录当前节点的前一个值，满足 前一个值 < query < 当前节点，就是答案
//        * */
//
//        List<Integer> list = new ArrayList<>();
//        dfs(root, list);
//        int len = list.size();
//        int[] sortArray = new int[len];
//        for (int i = 0; i < len; i++) {
//            sortArray[i] = list.get(i);
//        }
//        List<List<Integer>> ans = new ArrayList<>(queries.size());
//        for (int query : queries) {
//            int idx = binarySearch(sortArray, query);
//            int min = idx == -1 ? -1 : sortArray[idx];
//            if (idx == -1 || sortArray[idx] != query) {
//                idx++;
//            }
//            int max = idx > len ? -1 : sortArray[idx];
//            ans.add(List.of(min, max));
//        }
//        return ans;
//
//    }


    private int binarySearch(int[] sortArray, int target) {
        int left = -1;
        int right = sortArray.length;
        while (left + 1 < right) {
            // 开区间
            int mid = (left + right) >>> 1;
            if (sortArray[mid] <= target) {
                left = mid;
            } else
                right = mid;
        }
        return left;
    }

//    private void dfs(TreeNode root, List<Integer> list) {
//        if (root == null)
//            return;
//        dfs(root.left, list);
//        list.add(root.val);
//        dfs(root.right, list);
//    }

    private int maxCnt = 0;
    //private int cnt = 0;
//    private int pre = Integer.MIN_VALUE;

//    public int[] findMode(TreeNode root) {
//        /*501. 二叉搜索树中的众数
//        找出并返回 BST 中的所有众数，不止一个时可以任意顺序返回
//        * 思路：遍历节点时，对它进行计数，遍历结束后将次数最多的加入
//          进阶：不适用递归以外额外的空间
//          思路：没有利用到二叉搜索树的性质，根据中序遍历得到有序序列，众数一定是连续出现的
//          中序遍历时维护一个变量记录前一个节点的值，如果值相同，CNT++，不相同CNT重置为1
//          这样只能找到最高频率，怎么记录答案？
//          同时维护一个遍历到当前位置出现最多频率的 maxCnt，当 cnt == maxCnt 时记录答案
//          但是当 cnt > maxCnt 后要清空答案，同时更新 maxCnt
//        * */
//        Queue<Integer> stack = new LinkedList<>();
//        dfs(root, stack);
//        int[] ans = new int[stack.size()];
//        for (int i = 0; i < ans.length; i++) {
//            ans[i] = stack.poll();
//        }
//        return ans;
//    }

//    private void dfs(TreeNode node, Queue<Integer> stack) {
//        if (node == null)
//            return;
//        dfs(node.left, stack);
//        if (node.val == pre) {
//            cnt++;
//        } else {
//            cnt = 1;
//        }
//        if (cnt > maxCnt) {
//            maxCnt = cnt;
//            stack.clear();
//            stack.offer(node.val);
//        } else if (cnt == maxCnt) {
//            stack.offer(node.val);
//        }
//        pre = node.val;
//        dfs(node.right, stack);
//    }

    private int cnt;
    private int ans;

//    public int kthSmallest(TreeNode root, int k) {
//        /*230.二叉搜索树中第 k 小的元素
//         * 给定一个二叉搜索树的根节点 root ，
//         * 和一个整数 k ，找其中第 k 小的元素（k 从 1 开始计数）。
//         * */
//        cnt = k;
//        return dfs(root);
//    }

    //    private int dfs(TreeNode node) {
//
//        if(node==null)
//            return -1;
//        int left= dfs(node.left);
//        if(left !=-1)
//            return left;
//        if(--cnt == 0)
//            return node.val;
//        return dfs(node.right);
//    }
    private int maxSum = Integer.MIN_VALUE;

    public int maxSumBST(TreeNode root) {
        /*1373.二叉搜索子树的最大键值和
         * root 是二叉树，找到其中的二叉搜索子树，并返回二叉搜索子树中的最大键值和
         * 思路：判断左子树是否是二叉搜索树，右子树是否是二叉搜索树，如果都是，在判断当前树是否是二叉搜索树
         * 同时还需要返回子树的最大键值和，这样看后序遍历似乎更好
         * 注意：负数时，返回零，因为空的树是任意树的子树，也算二叉搜索树，其“最大和”为0
         * */
        dfs(root);
        return Math.max(maxSum, 0);

    }

//    private int[] dfs(TreeNode node) {
//        if (node == null)
//            return new int[]{Integer.MAX_VALUE, Integer.MIN_VALUE, 0};
//        int[] left = dfs(node.left);
//        int[] right = dfs(node.right);
//        int val = node.val;
//        int sum = left[2] + right[2] + val;
//        if (val > left[1] && val < right[0]) {
//            maxSum = Math.max(maxSum, sum);
//            return new int[]{Math.min(val, left[0]), Math.max(val, right[1]), sum};
//        }
//        return new int[]{Integer.MIN_VALUE, Integer.MAX_VALUE, sum};
//
//    }

    //    public TreeNode buildTree(int[] preorder, int[] inorder) {
//        /*105.从前序与中序序列构造二叉树
//         *
//         * */
//        int n = preorder.length;
//        Map<Integer, Integer> map = new HashMap<>(n);
//        for (int i = 0; i < inorder.length; i++) {
//            map.put(inorder[i], i);
//        }
//        return dfs(preorder, 0, n, 0, n, map);
//    }
//
//    private TreeNode dfs(int[] preorder, int preL, int preR, int inL, int inR, Map<Integer, Integer> map) {
//        if (preL == preR) {
//            return null;
//        }
//        int leftLen = map.get(preorder[preL]) - inL;
//        TreeNode left = dfs(preorder,preL+1,preL+1+leftLen,inL,inL+leftLen,map);
//        TreeNode right = dfs(preorder,preL+1+leftLen,preR,inL+1+leftLen,inR,map);
//        return new TreeNode(preorder[preL],left,right);
//
//    }
//    public TreeNode buildTree(int[] inorder, int[] postorder) {
//        /*106.从中序与后序构造二叉树
//         *
//         * */
//        int n = inorder.length;
//        Map<Integer, Integer> map = new HashMap<>(n);
//        for (int i = 0; i < inorder.length; i++) {
//            map.put(inorder[i], i);
//        }
//        return dfs(postorder, 0, n, 0, n, map);
//    }

//    private TreeNode dfs(int[] postorder, int posL, int posR, int inL, int inR, Map<Integer, Integer> map) {
//        if (posL == posR)
//            return null;
//
//        int leftLen = map.get(postorder[posR - 1]) - inL;
//        TreeNode left = dfs(postorder, posL, posL + leftLen, inL, inL + leftLen, map);
//        TreeNode right = dfs(postorder, posL + leftLen, posR - 1, inL + 1 + leftLen, inR, map);
//        return new TreeNode(postorder[posR-1],left,right);
//    }

    public TreeNode constructFromPrePost(int[] preorder, int[] postorder) {
        // 889.根据前序和后序构造二叉树，存在多个答案返回其中一个即可
        int n = preorder.length;
        int[] map = new int[n + 1];
        for (int i = 0; i < n; i++) {
            map[postorder[i]] = i;
        }
        return dfs(preorder, 0, 0, n, map);
    }

    private TreeNode dfs(int[] preorder, int posL, int preL, int preR, int[] map) {
        if (preL == preR)
            return null;
        if (preL + 1 == preR) { // 叶子节点
            return new TreeNode(preorder[preL]);
        }

        int leftLen = map[preorder[preL + 1]] - posL + 1;
        TreeNode left = dfs(preorder, posL, preL + 1, preL + 1 + leftLen, map);
        TreeNode right = dfs(preorder, posL + leftLen, preL + 1 + leftLen, preR, map);
        return new TreeNode(preorder[preL], left, right);
    }

    public List<TreeNode> delNodes(TreeNode root, int[] to_delete) {
        /*1110.删点成林
        *给出二叉树的根节点 root，树上每个节点都有一个不同的值。
        如果节点值在 to_delete 中出现，我们就把该节点从树上删去，最后得到一个森林（一些不相交的树构成的集合）。
        返回森林中的每棵树。你可以按任意顺序组织答案。
        * 思路：显然这道题后序遍历比较合适，先序如果删了当前节点，左右子树找不到了，中序右子树找不到了
        * 后序遍历：左右子树都删除完成后，判断当前节点是否需要删除，是的话，将左右加入答案
        * */
        int length = to_delete.length;
        Set<Integer> set = new HashSet<>(length);
        List<TreeNode> ans = new ArrayList<>();
        for (int i : to_delete) {
            set.add(i);
        }

        root = dfs(root, set, ans);
        if (root != null)
            ans.add(root);
        return ans;
    }

    private TreeNode dfs(TreeNode node, Set<Integer> map, List<TreeNode> ans) {
        if (node == null)
            return null;
        node.left = dfs(node.left, map, ans);
        node.right = dfs(node.right, map, ans);
        if (!map.contains(node.val)) {
            return node;
        }
        if (node.left != null)
            ans.add(node.left);
        if (node.right != null)
            ans.add(node.right);
        return null;

    }


//    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
//        /*236.二叉树的最近公共祖先
//         * 给定一个二叉树, 找到该树中两个指定节点的最近公共祖先。
//         * 注意：一个节点也可以是它自己的祖先
//         * 思路：如果直观做题，而不是写代码，我们是通过定位两个节点的位置，向上看哪个节点是他们公共的祖先
//         * 而通过代码实现，要有代码思维：讨论答案出现的情况与当前节点有什么特征，
//         * 同样也是找 p、q，遍历到的当前节点，与 p、q 的关系决定了当前节点是否是答案
//         * 分类讨论：如果当前节点 = p/q 这时它们的公共祖先不会在下面了，直接返回当前节点
//         * 同样如果当前节点为空，直接返回空也就是当前节点
//         * 如果p、q分别在左右子树上，当前节点就是公共祖先
//         * 如果右子树为空，说明 p、q 都在左子树中
//         * */
//        if (root == null || root == p || root == q)
//            return root;
//        // q p 在左子树中
//        TreeNode left = lowestCommonAncestor(root.left, p, q);
//        // p、q 在右子树中
//        TreeNode right = lowestCommonAncestor(root.right, p, q);
//
//        if (left != null && right != null)
//            return root;
//
//        if (right == null)
//            return left;
//        return right;
//
//
//    }


    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        /*235. 二叉搜索树的最近公共祖先
         * 如果 p，q 的值都比 root 的值小，说明在左子树中
         * 如果 p，q 的值都比 root 大，说明在右子树中
         * 总结一个规律，写这种题就想一个节点的答案怎么计算就好了。
         * */
        if (root == null || root == p || root == q)
            return root;
        int val = root.val;
        if (p.val < val && q.val < val) {
            return lowestCommonAncestor(root.left, p, q);
        } else if (p.val > val && q.val > val) {
            return lowestCommonAncestor(root.right, p, q);
        } else {
            return root;
        }
    }


//    private TreeNode ans;
//    private int maxDepth = -1;

    public TreeNode lcaDeepestLeaves(TreeNode root) {
        /*1123. 所有最深叶节点的最近公共祖先
         * 思路：找到最深叶节点
         * 利用之前找最深公共祖先的思路
         * 优化：在找最深叶节点时，如果最深叶节点在当前节点左子树中，则最近公共祖先也在其中
         * 如果左右子树最深叶节点深度一样则当前节点可能为答案
         * 全局维护整个树的最大深度，当左右子树叶节点的深度相等且与当前节点的最大深度相同时，记录答案
         * 当左右子树叶节点的深度大于全局最大深度时，更新
         * 这里有个优化就是将最深叶节点转化为最深空节点，因为最深空姐点的父节点一定是最深叶节点
         * 在上面的基础上，采用自底向上的思路：
         * 分治：对于每个子树，需要知道它的深度（不是全局的深度，而是子树的深度），以及这颗子树的最深叶子的最近公共祖先
         * 如果左子树的深度 > 右子树的深度
         * 说明左子树的最近公共祖先就是这颗树的最近公共祖先，这颗树的深度就是左子树深度 +1
         * 如果左子树深度 = 右子树深度，说明这棵树的最近公共祖先就是当前节点，且深度+1
         * */
        // 自己实现
//        List<TreeNode> list = new ArrayList<>();
//        dfs(root, list, 0);
//        return lowestCommonAncestor(root, list);
        //----------------------------------------------
        //优化后的实现
//        dfs(root, 0);
//        return ans;
        /*
         * ----------------------------------------------
         * 自底向上的实现*/
        return dfs(root).getValue();
    }

    static class Pair<K, V> {
        K x;
        V y;

        Pair(K x, V y) {
            this.x = x;
            this.y = y;
        }

        public K getKey() {
            return x;
        }

        public V getValue() {
            return y;
        }
    }

    private Pair<Integer, TreeNode> dfs(TreeNode root) {
        if (root == null) {
            return new Pair<>(0, null);
        }
        Pair<Integer, TreeNode> left = dfs(root.left);
        Pair<Integer, TreeNode> right = dfs(root.right);
        if (left.getKey() > right.getKey()) {
            return new Pair<>(left.getKey() + 1, left.getValue());
        }
        if (right.getKey() > left.getKey()) {
            return new Pair<>(right.getKey() + 1, right.getValue());
        }
        return new Pair<>(left.getKey() + 1, root);

    }

    // 返回值返回以当前节点为根的树的最大深度
//    private int dfs(TreeNode node, int depth) {
//        if (node == null) {
//            maxDepth = Math.max(maxDepth, depth);
//            return depth;
//        }
//        int leftMaxDepth = dfs(node.left, depth + 1);
//        int rightMaxDepth = dfs(node.right, depth + 1);
//        if (leftMaxDepth == rightMaxDepth && leftMaxDepth == maxDepth) {
//            ans = node;
//        }
//        return Math.max(leftMaxDepth, rightMaxDepth);
//    }

    private TreeNode lowestCommonAncestor(TreeNode root, List<TreeNode> list) {
        if (root == null || list.contains(root))
            return root;
        TreeNode left = lowestCommonAncestor(root.left, list);
        TreeNode right = lowestCommonAncestor(root.right, list);
        if (left == null) return right;
        if (right == null) return left;
        return root;
    }

    //    private void dfs(TreeNode node, List<TreeNode> list, int depth) {
//        if (node == null)
//            return;
//        if (node.right == null && node.left == null) {
//            // 叶子节点
//            if (depth == maxDepth) {
//                list.add(node);
//            }
//            if (depth > maxDepth) {
//                maxDepth = depth;
//                list.clear();
//                list.add(node);
//            }
//        }
//        dfs(node.left, list, depth + 1);
//        dfs(node.right, list, depth + 1);
//    }
    public List<List<Integer>> levelOrder(TreeNode root) {
        // 102.二叉树的层序遍历
        /* 思路：利用两个数组，一个记录当前层节点，一个记录下一层节点，
        遍历当前层时，将当前节点的左右儿子加入下一层节点，
        为了优化，减少两个数组的使用，引入了队列，怎么判断这一层的循环结束？
        进入循环时，记录队列长度，该长度就是当前层的节点
        *
        * */
        if (root == null) {
            return List.of();
        }
        List<List<Integer>> ans = new ArrayList<>();
        Queue<TreeNode> que = new LinkedList<>();
        que.offer(root);
        while (!que.isEmpty()) {
            int len = que.size();
            List<Integer> list = new ArrayList<>(len);
            for (int i = 0; i < len; i++) {
                TreeNode node = que.poll();
                list.add(node.val);
                if (node.left != null) que.offer(node.left);
                if (node.right != null) que.offer(node.right);
            }
            ans.add(list);
        }
        return ans;
    }

    public List<List<Integer>> zigzagLevelOrder(TreeNode root) {
        /*102.二叉树的锯齿形层序遍历
         * 给你二叉树的根节点 root ，返回其节点值的 锯齿形层序遍历 。
         * （即先从左往右，再从右往左进行下一层遍历，以此类推，层与层之间交替进行）。
         * 思路：再层序遍历的基础上，判断当前上一层是否反向，是的话这一层就不反
         * */
        if (root == null) {
            return List.of();
        }
        List<List<Integer>> ans = new ArrayList<>();
        Queue<TreeNode> que = new LinkedList<>();
        que.offer(root);
        boolean flag = false;
        while (!que.isEmpty()) {
            int len = que.size();
            List<Integer> list = new ArrayList<>(len);
            for (int i = 0; i < len; i++) {
                TreeNode node = que.poll();
                list.add(node.val);
                if (node.left != null) que.offer(node.left);
                if (node.right != null) que.offer(node.right);
            }
            if (!flag) {
                ans.add(list);
            } else
                ans.add(list.reversed());
            flag = !flag;
        }
        return ans;
    }

    public int findBottomLeftValue(TreeNode root) {
        /*513.找树左下角的值
         * 给定一个二叉树的 根节点 root，请找出该二叉树的 最底层 最左边 节点的值。
         * 假设二叉树中至少有一个节点。
         * 1.直接返回层序遍历结果最后一层第一个树
         * 2.反向层序，返回最后一个节点的值
         * */
        Queue<TreeNode> que = new LinkedList<>();
        que.offer(root);
        TreeNode node = root;
        while (!que.isEmpty()) {
            node = que.poll();
            if (node.right != null) que.offer(node.right);
            if (node.left != null) que.offer(node.left);
        }
        return node.val;
    }

    public List<List<Integer>> levelOrderBottom(TreeNode root) {
        /*107.二叉树的层序遍历Ⅱ
         * 给你二叉树的根节点 root ，返回其节点值 自底向上的层序遍历 。
         * （即按从叶子节点所在层到根节点所在的层，逐层从左向右遍历
         * 思路：层序遍历后反转答案
         * list.reversed()返回的是“倒序视图（view）”，不改变原列表；
         * Collections.reverse(list) 是“就地反转”，直接把原列表的元素顺序翻转掉。
         * */
        if (root == null) {
            return List.of();
        }
        List<List<Integer>> ans = new ArrayList<>();
        Queue<TreeNode> que = new LinkedList<>();
        que.offer(root);
        while (!que.isEmpty()) {
            int len = que.size();
            List<Integer> list = new ArrayList<>(len);
            for (int i = 0; i < len; i++) {
                TreeNode node = que.poll();
                list.add(node.val);
                if (node.left != null) que.offer(node.left);
                if (node.right != null) que.offer(node.right);
            }
            ans.add(list);
        }
        return ans.reversed();
    }

    public long kthLargestLevelSum(TreeNode root, int k) {
        /*2583.二叉树中的第 K 大层和
        * 给你一棵二叉树的根节点 root 和一个正整数 k 。
        树中的 层和 是指 同一层 上节点值的总和。
        返回树中第 k 大的层和（不一定不同）。如果树少于 k 层，则返回 -1 。
        * 就是按层和排序，返回第 k 大的层和
        * */
        if (root == null)
            return -1;
        Queue<TreeNode> q = new LinkedList<>();
        q.offer(root);
        List<Long> ans = new ArrayList<>();
        while (!q.isEmpty()) {
            int len = q.size();
            Long sum = 0L;
            while (len-- > 0) {
                TreeNode node = q.poll();
                sum += node.val;
                if (node.left != null) q.offer(node.left);
                if (node.right != null) q.offer(node.right);
            }
            ans.add(sum);
        }
        Collections.sort(ans);
        int len = ans.size();
        if (len < k) {
            return -1;
        } else
            return ans.get(len - k);
    }

    public Node connect(Node root) {
        /*116.填充每个节点的下一个右侧节点指针
        * 给定一个 完美二叉树 ，其所有叶子节点都在同一层，每个父节点都有两个子节点。二叉树定义如下：
        * struct Node {
          int val;
          Node *left;
          Node *right;
          Node *next;
        }
        * 填充它的每个 next 指针，让这个指针指向其下一个右侧节点。
        * 如果找不到下一个右侧节点，则将 next 指针设置为 NULL。
        * 初始状态下，所有 next 指针都被设置为 NULL。
        * */
        Node dummy = new Node();
        Node cur = root;
        while (cur != null) {
            dummy.next = null;
            Node nxt = dummy;//下一层的链表
            while (cur != null) {
                if (cur.left != null) {
                    nxt.next = cur.left;
                    nxt = cur.left;
                }
                if (cur.right != null) {
                    nxt.next = cur.right;
                    nxt = cur.right;
                }
                cur = cur.next;
            }
            cur = dummy.next;
        }

        return root;
    }

    public int deepestLeavesSum(TreeNode root) {
        /*1302. 层数最深叶子节点和
        给你一棵二叉树的根节点 root ，请你返回层数最深的叶子节点的和 。
        * */
        if (root == null)
            return -1;
        Queue<TreeNode> q = new LinkedList<>();
        q.offer(root);
        int sum;
        while (!q.isEmpty()) {
            int len = q.size();
            sum = 0;
            while (len-- > 0) {
                TreeNode node = q.poll();
                sum += node.val;
                if (node.left != null) q.offer(node.left);
                if (node.right != null) q.offer(node.right);
            }
        }
        return sum;
    }

    public boolean isEvenOddTree(TreeNode root) {
        /*1609.奇偶数
         * 如果一棵二叉树满足下述几个条件，则可以称为 奇偶树 ：
         * 从第 0 层开始，偶数层上的节点值都是严格递增的奇数
         * 奇数层上的节点值是严格递减的偶数
         * */
        Queue<TreeNode> q = new LinkedList<>();
        q.offer(root);
        boolean flag = true;
        while (!q.isEmpty()) {
            int len = q.size();
            while (len-- > 0) {
                TreeNode node = q.poll();
                int val = node.val;
                if (flag) {
                    if (val % 2 == 0)
                        return false;
                } else if (val % 2 == 1)
                    return false;
                if (len > 0) {
                    if (flag) {
                        // 偶数层
                        if (q.peek() != null) {
                            if (val >= q.peek().val) return false;
                        }
                    } else {
                        // 奇数层
                        if (q.peek() != null) {
                            if (val <= q.peek().val) return false;
                        }
                    }
                }

                if (node.left != null) q.offer(node.left);
                if (node.right != null) q.offer(node.right);
            }
            flag = !flag;
        }
        return true;

    }

    public TreeNode reverseOddLevels(TreeNode root) {
        /*2415.反转完美二叉树的奇数层
         * 更改值
         * */
        Queue<TreeNode> q = new LinkedList<>();
        q.offer(root);
        boolean flag = false;
        while (!q.isEmpty()) {
            int len = q.size();
            List<TreeNode> list = new ArrayList<>(len);
            while (len-- > 0) {
                TreeNode node = q.poll();
                list.add(node);
                if (node.left != null) q.offer(node.left);
                if (node.right != null) q.offer(node.right);
            }
            if (flag) {
                for (int i = 0; i < list.size() / 2; i++) {
                    TreeNode node1 = list.get(i);
                    TreeNode node2 = list.get(list.size() - 1 - i);
                    int temp = node1.val;
                    node1.val = node2.val;
                    node2.val = temp;
                }
            }
            flag = !flag;
        }
        return root;
    }
}

public TreeNode replaceValueInTree(TreeNode root) {
        /*2641.二叉树的堂兄弟节点Ⅱ
        给你一棵二叉树的根 root ，请你将每个节点的值替换成该节点的所有 堂兄弟节点值的和 。
        两个节点处于同一层，但是父节点不同
        思路：先一次遍历，拿到每一层的和，再遍历一次更改值
        * */
    Queue<TreeNode> q = new LinkedList<>();
    q.offer(root);
    List<Integer> sumList = new ArrayList<>();
    // 拿到每一层的所有节点和
    while (!q.isEmpty()) {
        int len = q.size();
        int sum = 0;
        while (len-- > 0) {
            TreeNode node = q.poll();
            sum += node.val;
            if (node.left != null) q.offer(node.left);
            if (node.right != null) q.offer(node.right);
        }
        sumList.add(sum);
    }
    int level = 0;
    q.offer(root);
    while (level < sumList.size() - 1) {
        int len = q.size();
        int allSum = sumList.get(level + 1);
        while (len-- > 0) {
            TreeNode node = q.poll();
            int sum = 0;
            if (node.left != null) {
                sum += node.left.val;
                q.offer(node.left);
            }
            if (node.right != null) {
                sum += node.right.val;
                q.offer(node.right);
            }
            if (node.left != null) node.left.val = allSum - sum;
            if (node.right != null) node.right.val = allSum - sum;
        }
        level++;
    }
    root.val = 0;
    return root;
}
