import java.util.HashSet;
import java.util.Set;

public class Solution {
    public ListNode middleNode(ListNode head) {
        /*876. 链表的中间节点*/
        ListNode slow = head;
        ListNode fast = head;
        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
        }
        return slow;
    }

    public boolean hasCycle(ListNode head) {
        /*141. 环形链表
         * 判断链表是否有环
         * 思路：快慢指针：如果有环，则比相遇，否则无环
         * */
        ListNode slow = head;
        ListNode fast = head;
        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
            if (slow == fast)
                return true;
        }
        return false;
    }

    public ListNode detectCycle(ListNode head) {
        /* 142.环形链表Ⅱ
           返回环的入口，没有则返回 null
           a:头节点到环入口的距离
           b：相遇时距离环入口的距离
           c：相遇时完成一个环剩下的距离
           2(a+b)=a+b+k(b+c) => a-c = (k-1)(b+c)
           这个式子的意义在于：
           两个节点分别从距离入口 a-c 的距离以及环入口开始走，两节点比在环入口相遇
        * */
        ListNode slow = head;
        ListNode fast = head;
        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
            if (slow == fast) {
                // 快慢指针相遇
                while (head != slow) {
                    head = head.next;
                    slow = slow.next;
                }
                return head;
            }
        }
        return null;
    }


    public void reorderList(ListNode head) {
        /*143.重排链表
        *   给定一个单链表 L 的头节点 head ，单链表 L 表示为：
        *    L0 → L1 → … → Ln - 1 → Ln
        *   请将其重新排列后变为：
            L0 → Ln → L1 → Ln - 1 → L2 → Ln - 2 → …
        *   不能只是单纯的改变节点内部的值，而是需要实际的进行节点交换。
        * */
        ListNode head2 = middleNode(head);
        head2 = reverseList(head2);
        while (head2.next != null) {
            ListNode nxt = head.next;
            ListNode nxt2 = head2.next;
            head.next = head2;
            head2.next = nxt;
            head = nxt;
            head2 = nxt2;
        }
    }

    private ListNode reverseList(ListNode head) {
        ListNode pre = null;
        ListNode cur = head;
        while (cur != null) {
            ListNode nxt = cur.next;
            cur.next = pre;
            pre = cur;
            cur = nxt;
        }
        return pre;
    }

    public boolean isPalindrome(ListNode head) {
        /*234.回文链表
         * 思路：找到中间节点，将它反转，然后依次比较值
         * */
        ListNode head2 = middleNode(head);
        head2 = reverseList(head2);
        while (head2 != null) {
            if (head.val != head2.val) {
                return false;
            }
            head2 = head2.next;
            head = head.next;
        }
        return true;
    }

    public int pairSum(ListNode head) {
        /* 2130.链表最大孪生和
         * 给定一个大小为偶数的链表
         * 孪生节点表示该节点对称节点。
         * 例如长为4的链表，第0个节点的孪生节点为最后一个节点，也就是第3个节点
         * 求一个节点和他孪生节点和的最大值
         * */
        int max = Integer.MIN_VALUE;
        ListNode head2 = middleNode(head);
        head2 = reverseList(head2);
        while (head2 != null) {
            max = Math.max(max, head.val + head2.val);
            head = head.next;
            head2 = head2.next;
        }
        return max;
    }

    public void deleteNode(ListNode node) {
        /*237. 删除链表中的节点
         * 链表值唯一，保证所给的node不是最后一个节点，
         * 这里的删除指给定节点的只不存在链表中
         * */
        node.val = node.next.val;
        node.next = node.next.next;
    }

    public ListNode removeNthFromEnd(ListNode head, int n) {
        /*19.删除链表的倒数第 N 个节点
         * 给你一个链表，删除链表的倒数第 n 个结点，并且返回链表的头结点。
         * 思路：因为头节点可以被删除，引入dummy node 简化删除逻辑
         * */
        ListNode dummy = new ListNode(0, head);
        ListNode right = dummy;
        for (int i = 0; i < n; i++) {
            right = right.next;
        }
        ListNode left = dummy;
        while (right.next != null) {
            left = left.next;
            right = right.next;
        }
        left.next = left.next.next;
        return dummy.next;
    }

    /*    public ListNode deleteDuplicates(ListNode head) {
     *//*83.删除排序链表中的重复元素
     * 给定一个已排序的链表的头 head ，
     * 删除所有重复的元素，使每个元素只出现一次 。返回已排序的链表 。
     * 思路：判断当前节点的值与下一个节点的值是否相同，
     * 相同就删除下一个节点，直到不同时才移动当前节点
     * *//*
        if(head == null)
            return head;
        ListNode cur = head;
        while (cur.next != null) {
            if (cur.val == cur.next.val) {
                cur.next = cur.next.next;
            } else cur = cur.next;
        }
        return head;
    }*/
    public ListNode deleteDuplicates(ListNode head) {
        /*82.删除排序链表中的重复元素
         *  给定一个已排序的链表的头 head ，
         * 删除原始链表中所有重复数字的节点，只留下不同的数字 。返回已排序的链表 。
         * 思路：这题不同的是，只要有重复数字出现，全部删除，可能删除到头节点
         * 引入 dummy node，同时当有重复值出现时，需要循环删除
         * */
        if (head == null) {
            return head;
        }
        ListNode dummy = new ListNode(0, head);
        ListNode cur = dummy;
        // 循环内要用到next,next.next 所以两个都要判断
        while (cur.next != null && cur.next.next != null) {
            int val = cur.next.next.val;
            if (cur.next.val == val) {
                while (cur.next != null && cur.next.val == val) {
                    cur.next = cur.next.next;
                }
            } else cur = cur.next;
        }
        return dummy.next;

    }

    public ListNode removeElements(ListNode head, int val) {
        /*203.移除链表元素
         * 给你一个链表的头节点 head 和一个整数 val ，
         * 请你删除链表中所有满足 Node.val == val 的节点，并返回新的头节点 。
         * 思路：有可能删除头节点，引入dummy node
         * */
        ListNode dummy = new ListNode(0, head);
        ListNode cur = dummy;
        while (cur.next != null) {
            if (cur.next.val == val) {
                cur.next = cur.next.next;
            } else {
                cur = cur.next;
            }
        }
        return dummy.next;
    }

    public ListNode modifiedList(int[] nums, ListNode head) {
        /*3217. 从链表中移除在数组中存在的节点
         * 给你一个整数数组 nums 和一个链表的头节点 head。
         * 从链表中移除所有存在于 nums 中的节点后，返回修改后的链表的头节点。
         * 将 nums 中的元素存于 hashset，遍历链表时判断是否存在
         * */
        Set<Integer> set = new HashSet<>(nums.length);
        for (int num : nums) {
            set.add(num);
        }
        ListNode dummy = new ListNode(0, head);
        ListNode cur = dummy;
        while (cur.next != null) {
            if (set.contains(cur.next.val)) {
                cur.next = cur.next.next;
            } else
                cur = cur.next;
        }
        return dummy.next;
    }

    public ListNode removeNodes(ListNode head) {
        /*2487.从链表中移除节点
        * 给你一个链表的头节点 head 。
        移除每个右侧有一个更大数值的节点。
        返回修改后链表的头节点 head
        * */
        head = reverseList(head);
        // 反转链表，现在是删除比当前节点值小的节点
        ListNode cur = head;
        while (cur.next != null) {
            if (cur.val > cur.next.val) {
                cur.next = cur.next.next;
            } else {
                cur = cur.next;
            }
        }
        return reverseList(head);
    }

    public ListNode mergeInBetween(ListNode list1, int a, int b, ListNode list2) {
        /*1669.合并两个链表
         * 给你两个链表 list1 和 list2 ，它们包含的元素分别为 n 个和 m 个。
         * 请你将 list1 中下标从 a 到 b 的全部节点都删除，并将list2 接在被删除节点的位置。
         * 思路：有可能删除 list1 的头节点，dummy node
         * 需要一个list2 的尾指针
         * 需要一个指针定位插入位置
         * */
        ListNode dummy = new ListNode(0, list1);
        ListNode tail2 = null;
        ListNode cur = list2;
        while (cur.next != null) {
            cur = cur.next;
        }
        tail2 = cur;
        cur = dummy;
        for (int i = 0; i < a; i++) {
            cur = cur.next;
        }
        for (int i = 0; i < (b - a + 1); i++) {
            cur.next = cur.next.next;
        }
        tail2.next = cur.next;
        cur.next = list2;
        return dummy.next;

    }

}
