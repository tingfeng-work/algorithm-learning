import java.util.*;

/**
 * Example 1:
 * root = [5,8,9,2,1,3,7,4,6], k = 2
 * level sums: [5, 17, 13, 10] -> 2nd largest = 13
 */
public class Test {

    // Definition for a binary tree node.
    public static class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;

        TreeNode() {
        }

        TreeNode(int val) {
            this.val = val;
        }

        TreeNode(int val, TreeNode left, TreeNode right) {
            this.val = val;
            this.left = left;
            this.right = right;
        }
    }

    // ===== 你的题解方法（你把自己的 kthLargestLevelSum 放这里即可）=====
    public static long kthLargestLevelSum(TreeNode root, int k) {
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

    // ===== 用层序数组构建二叉树（支持 null）=====
    public static TreeNode buildTree(Integer[] levelOrder) {
        if (levelOrder == null || levelOrder.length == 0 || levelOrder[0] == null) return null;

        TreeNode root = new TreeNode(levelOrder[0]);
        Queue<TreeNode> q = new ArrayDeque<>();
        q.offer(root);

        int i = 1;
        while (i < levelOrder.length && !q.isEmpty()) {
            TreeNode cur = q.poll();

            // left
            if (i < levelOrder.length && levelOrder[i] != null) {
                cur.left = new TreeNode(levelOrder[i]);
                q.offer(cur.left);
            }
            i++;

            // right
            if (i < levelOrder.length && levelOrder[i] != null) {
                cur.right = new TreeNode(levelOrder[i]);
                q.offer(cur.right);
            }
            i++;
        }
        return root;
    }

    public static void main(String[] args) {
        // 示例 1：root = [5,8,9,2,1,3,7,4,6], k = 2
        Integer[] arr = {5, 8, 9, 2, 1, 3, 7, 4, 6};
        int k = 2;

        TreeNode root = buildTree(arr);
        long ans = kthLargestLevelSum(root, k);

        System.out.println("Input: root=" + Arrays.toString(arr) + ", k=" + k);
        System.out.println("Output: " + ans);
        System.out.println("Expected: 13");
    }
}
