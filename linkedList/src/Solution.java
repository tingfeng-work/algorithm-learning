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

}
