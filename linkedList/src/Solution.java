import java.util.HashSet;
import java.util.Set;

public class Solution {
    /*    public ListNode middleNode(ListNode head) {
     *//*876. 链表的中间节点*//*
        ListNode slow = head;
        ListNode fast = head;
        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
        }
        return slow;
    }

    public boolean hasCycle(ListNode head) {
        *//*141. 环形链表
     * 判断链表是否有环
     * 思路：快慢指针：如果有环，则比相遇，否则无环
     * *//*
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
        *//* 142.环形链表Ⅱ
           返回环的入口，没有则返回 null
           a:头节点到环入口的距离
           b：相遇时距离环入口的距离
           c：相遇时完成一个环剩下的距离
           2(a+b)=a+b+k(b+c) => a-c = (k-1)(b+c)
           这个式子的意义在于：
           两个节点分别从距离入口 a-c 的距离以及环入口开始走，两节点比在环入口相遇
        * *//*
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
        *//*143.重排链表
        *   给定一个单链表 L 的头节点 head ，单链表 L 表示为：
        *    L0 → L1 → … → Ln - 1 → Ln
        *   请将其重新排列后变为：
            L0 → Ln → L1 → Ln - 1 → L2 → Ln - 2 → …
        *   不能只是单纯的改变节点内部的值，而是需要实际的进行节点交换。
        * *//*
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
        *//*234.回文链表
     * 思路：找到中间节点，将它反转，然后依次比较值
     * *//*
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
        *//* 2130.链表最大孪生和
     * 给定一个大小为偶数的链表
     * 孪生节点表示该节点对称节点。
     * 例如长为4的链表，第0个节点的孪生节点为最后一个节点，也就是第3个节点
     * 求一个节点和他孪生节点和的最大值
     * *//*
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
        *//*237. 删除链表中的节点
     * 链表值唯一，保证所给的node不是最后一个节点，
     * 这里的删除指给定节点的只不存在链表中
     * *//*
        node.val = node.next.val;
        node.next = node.next.next;
    }

    public ListNode removeNthFromEnd(ListNode head, int n) {
        *//*19.删除链表的倒数第 N 个节点
     * 给你一个链表，删除链表的倒数第 n 个结点，并且返回链表的头结点。
     * 思路：因为头节点可以被删除，引入dummy node 简化删除逻辑
     * *//*
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

    *//*    public ListNode deleteDuplicates(ListNode head) {
     *//**//*83.删除排序链表中的重复元素
     * 给定一个已排序的链表的头 head ，
     * 删除所有重复的元素，使每个元素只出现一次 。返回已排序的链表 。
     * 思路：判断当前节点的值与下一个节点的值是否相同，
     * 相同就删除下一个节点，直到不同时才移动当前节点
     * *//**//*
        if(head == null)
            return head;
        ListNode cur = head;
        while (cur.next != null) {
            if (cur.val == cur.next.val) {
                cur.next = cur.next.next;
            } else cur = cur.next;
        }
        return head;
    }*//*
    public ListNode deleteDuplicates(ListNode head) {
        *//*82.删除排序链表中的重复元素
     *  给定一个已排序的链表的头 head ，
     * 删除原始链表中所有重复数字的节点，只留下不同的数字 。返回已排序的链表 。
     * 思路：这题不同的是，只要有重复数字出现，全部删除，可能删除到头节点
     * 引入 dummy node，同时当有重复值出现时，需要循环删除
     * *//*
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
        *//*203.移除链表元素
     * 给你一个链表的头节点 head 和一个整数 val ，
     * 请你删除链表中所有满足 Node.val == val 的节点，并返回新的头节点 。
     * 思路：有可能删除头节点，引入dummy node
     * *//*
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
        *//*3217. 从链表中移除在数组中存在的节点
     * 给你一个整数数组 nums 和一个链表的头节点 head。
     * 从链表中移除所有存在于 nums 中的节点后，返回修改后的链表的头节点。
     * 将 nums 中的元素存于 hashset，遍历链表时判断是否存在
     * *//*
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
        *//*2487.从链表中移除节点
        * 给你一个链表的头节点 head 。
        移除每个右侧有一个更大数值的节点。
        返回修改后链表的头节点 head
        * *//*
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
        *//*1669.合并两个链表
     * 给你两个链表 list1 和 list2 ，它们包含的元素分别为 n 个和 m 个。
     * 请你将 list1 中下标从 a 到 b 的全部节点都删除，并将list2 接在被删除节点的位置。
     * 思路：有可能删除 list1 的头节点，dummy node
     * 需要一个list2 的尾指针
     * 需要一个指针定位插入位置
     * *//*
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

    }*/

    public ListNode reverseList(ListNode head) {
        /*
        * 206.反转链表
        * 给你单链表的头节点 head ，请你反转链表，并返回反转后的链表
        * 输入：head = [1,2,3,4,5]
          输出：[5,4,3,2,1]
        * 思路：本质上是当前节点的next指向前一个节点，所以需要 cur指向当前节点，pre指向前一个节点，
        * 同时，修改后，cur.next 会丢失，所以还需要一个 nxt 指向cur.next
        * */
        ListNode cur = head, pre = null;
        while (cur != null) {
            ListNode nxt = cur.next;
            cur.next = pre;
            pre = cur;
            cur = nxt;
        }
        return pre;
    }

    public ListNode reverseBetween(ListNode head, int left, int right) {
        /*
         * 92.反转链表Ⅱ
         * 给你单链表的头指针 head 和两个整数 left 和 right ，其中 left <= right 。
         * 请你反转从位置 left 到位置 right 的链表节点，返回 反转后的链表 。
         * 思路：类似地，用206的方法反转链表，left前一个节点记为p0，此时指向的是反转链表的尾端，p0.next.next = nxt,同时p0.next = pre即可
         * 但是如果，left=1，此时没有前一个节点，为了简化逻辑，引入dummy节点
         * */
        ListNode dummy = new ListNode();
        dummy.next = head;
        ListNode cur = head;
        ListNode p0 = dummy;
        for (int i = 1; i < left; i++) {
            cur = cur.next;
            p0 = p0.next;
        }

        ListNode pre = null;
        ListNode nxt = null;
        for (int i = 0; i < right - left + 1; i++) {
            nxt = cur.next;
            cur.next = pre;
            pre = cur;
            cur = nxt;
        }
        p0.next.next = nxt;
        p0.next = pre;

        return dummy.next;
    }

    public ListNode reverseKGroup(ListNode head, int k) {
        /*
         * 25. K 个一组翻转链表
         * 给你链表的头节点 head ，每 k 个节点一组进行翻转，请你返回修改后的链表
         * k 是一个正整数，它的值小于或等于链表的长度。如果节点总数不是 k 的整数倍，那么请将最后剩余的节点保持原有顺序。
         * 类似 92.反转链表Ⅱ，只是每次反转前要更新 p0，更新p0为p0.next,而p0.next要被修改，所以需要提前记录
         * */
        int len = 0;
        ListNode cur = head;
        while (cur != null) {
            len++;
            cur = cur.next;
        }
        ListNode dummy = new ListNode(0, head);

        ListNode p0 = dummy, pre = null;
        cur = p0.next;

        while (len >= k) {
            len = len - k;
            for (int i = 0; i < k - 1; i++) {
                ListNode nxt = cur.next;
                cur.next = pre;
                pre = cur;
                cur = nxt;
            }
            ListNode temp = p0.next;
            p0.next.next = cur;
            p0.next = pre;
            p0 = temp;
        }
        return dummy.next;
    }

    public ListNode swapPairs(ListNode head) {
        /*
         * 24. 两两交换链表中的节点
         * 给你一个链表，两两交换其中相邻的节点，并返回交换后链表的头节点。
         * 你必须在不修改节点内部的值的情况下完成本题（即，只能进行节点交换）。
         * 思路：这不就是两个一组反转链表吗？两个一组，不足两个则不操作，反转链表
         * */
        ListNode dummy = new ListNode(0, head);
        ListNode cur = head;
        int len = 0;
        while (cur != null) {
            len++;
            cur = cur.next;
        }

        ListNode p0 = dummy, pre = null;
        cur = p0.next;
        for (; len >= 2; len = len - 2) {
            for (int i = 0; i < 2; i++) {
                ListNode nxt = cur.next;
                cur.next = pre;
                pre = cur;
                cur = nxt;
            }
            ListNode temp = p0.next;
            p0.next.next = cur;
            p0.next = pre;
            p0 = temp;
        }
        return dummy.next;
    }

    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        /*
         * 445. 两数相加 II
         * 给你两个非空链表来代表两个非负整数。数字最高位位于链表开始位置。它们的每个节点只存储一位数字。
         * 将这两数相加并返回一个新的链表
         * 你可以假设除了数字 0 之外，这两个数字都不会以零开头。
         * 思路：反转这两个链表，这样遍历就是从低位开始，在做加法，同时记录一个 flag 用来表示是否有进位产生
         * */
        l1 = reverseList(l1);
        l2 = reverseList(l2);
        int carry = 0;
        ListNode dummy = new ListNode();
        ListNode cur = dummy;
        while (l1 != null || l2 != null || carry != 0) {
            if (l1 != null) carry = carry + l1.val;
            if (l2 != null) carry = carry + l2.val;
            cur.next = new ListNode(carry % 10);
            cur = cur.next;
            carry = carry / 10;
            if (l1 != null) l1 = l1.next;
            if (l2 != null) l2 = l2.next;
        }
        return dummy.next;
    }

    public ListNode doubleIt(ListNode head) {
        /*
         * 2816. 翻倍以链表形式表示的数字
         * 给你一个 非空 链表的头节点 head ，表示一个不含前导零的非负数整数。
         *
         * */
        head = reverseList(head);
        int carry = 0;
        ListNode dummy = new ListNode();
        ListNode cur = dummy;
        while (head != null || carry != 0) {
            if (head != null) carry += head.val + head.val;
            cur.next = new ListNode(carry % 10);
            cur = cur.next;
            carry = carry / 10;
            if (head != null) head = head.next;
        }
        return reverseList(dummy.next);
    }

    public ListNode middleNode(ListNode head) {
        /*
         * 876. 链表的中间结点
         * 给你单链表的头结点 head ，请你找出并返回链表的中间结点。
         * 如果有两个中间结点，则返回第二个中间结点。
         * 思路：快慢指针，一个指针每次移动两个节点，一个指针每次移动一个节点
         * 当快指针为空或快指针下一个节点为空时，慢指针指向的就是中间节点
         * */
        ListNode fast = head, slow = head;
        while (fast != null && fast.next != null) {
            fast = fast.next.next;
            slow = slow.next;
        }
        return slow;
    }

    public boolean hasCycle(ListNode head) {
        /*
         * 141. 环形链表
         * 给你一个链表的头节点 head ，判断链表中是否有环。
         * 思路：同样的使用快慢指针，如果存在环，则最终快慢指针会相遇
         * */
        ListNode fast = head, slow = head;
        while (fast != null && fast.next != null) {
            fast = fast.next.next;
            slow = slow.next;
            if (slow == fast)
                return true;
        }
        return false;
    }

    public ListNode detectCycle(ListNode head) {
        /*
         * 142. 环形链表 II
         * 给定一个链表的头节点  head ，返回链表开始入环的第一个节点。 如果链表无环，则返回 null。
         * */
        ListNode fast = head, slow = head;
        while (fast != null && fast.next != null) {
            fast = fast.next.next;
            slow = slow.next;
            if (fast == slow) {
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
        /*
         * 143. 重排链表
         * 给定一个单链表 L 的头节点 head ，单链表 L 表示为：
         * L0 → L1 → … → Ln - 1 → Ln
         * 请将其重新排列后变为：
         * L0 → Ln → L1 → Ln - 1 → L2 → Ln - 2 → …
         * 不能只是单纯的改变节点内部的值，而是需要实际的进行节点交换。
         * 思路：找到链表的中间节点，然后反转中间节点后的链表
         * 做排序
         * */
        ListNode head2 = middleNode(head);
        head2 = reverseList(head2);
        while (head2.next != null) {
            ListNode nxt = head.next, nxt2 = head2.next;
            head.next = head2;
            head2.next = nxt;
            head = nxt;
            head2 = nxt2;
        }
    }

    public boolean isPalindrome(ListNode head) {
        /*
        234. 回文链表
        * 给你一个单链表的头节点 head ，请你判断该链表是否为回文链表。如果是，返回 true ；否则，返回 false 。
        思路：找到中间节点，然后从中间节点反转，再从头挨个比较，不相等则返回false
        * */
        ListNode fast = head, slow = head;
        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
        }
        ListNode pre = null, cur = slow;
        while (cur != null) {
            ListNode nxt = cur.next;
            cur.next = pre;
            pre = cur;
            cur = nxt;
        }
        while (pre != null) {
            if (pre.val != head.val) return false;
            else {
                pre = pre.next;
                head = head.next;
            }
        }
        return true;
    }

    public int pairSum(ListNode head) {
        /*
         * 2130. 链表最大孪生和
         * 在一个大小为 n 且 n 为 偶数 的链表中，对于 0 <= i <= (n / 2) - 1 的 i ，
         * 第 i 个节点（下标从 0 开始）的孪生节点为第 (n-1-i) 个节点 。
         * 孪生和 定义为一个节点和它孪生节点两者值之和。
         * 给你一个长度为偶数的链表的头节点 head ，请你返回链表的 最大孪生和 。
         * */
        int ans = 0;
        ListNode fast = head, slow = head;
        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
        }
        ListNode pre = null, cur = slow;
        while (cur != null) {
            ListNode nxt = cur.next;
            cur.next = pre;
            pre = cur;
            cur = nxt;
        }
        while (pre != null) {
            int sum = pre.val + head.val;
            if (sum > ans) ans = sum;
            pre = pre.next;
            head = head.next;
        }
        return ans;

    }

    public void deleteNode(ListNode node) {
        /*
         * 237. 删除链表中的节点
         * 给你一个需要删除的节点 node 。你将 无法访问 第一个节点  head。
         * 链表的所有值都是 唯一的，并且保证给定的节点 node 不是链表中的最后一个节点。
         * 删除给定的节点。注意，删除节点并不是指从内存中删除它。这里的意思是：
         * 给定节点的值不应该存在于链表中。
         * 链表中的节点数应该减少 1。
         * node 前面的所有值顺序相同。
         * node 后面的所有值顺序相同。
         * 思路：因为拿不到前一个节点，把后一个节点的值复制过来，然后删除下一个节点
         * */
        node.val = node.next.val;
        node.next = node.next.next;
    }

    public ListNode removeNthFromEnd(ListNode head, int n) {
        /*
         * 19. 删除链表的倒数第 N 个结点
         * 给你一个链表，删除链表的倒数第 n 个结点，并且返回链表的头结点。
         * 思路：这里有可能删除头节点，所以需要一个dummy；
         * 要删除倒数第n个节点，就需要找到倒数第n+1个节点，
         * 采用前后指针，先用一个指针走n步，再来一个慢指针从头节点出发，两个指针一起走，它们之间的距离就始终是 n
         * 这样先走的块指针到达链表尾部的时候，慢指针指向的就是倒数第n+1个节点
         * */
        if (head == null) return null;
        ListNode dummy = new ListNode(0, head), fast = dummy, slow = dummy;
        for (int i = 0; i < n; i++) {
            fast = fast.next;
        }
        while (fast.next != null) {
            fast = fast.next;
            slow = slow.next;
        }
        slow.next = slow.next.next;
        return dummy.next;
    }

    public ListNode deleteDuplicates(ListNode head) {
        /*
         * 83. 删除排序链表中的重复元素
         * 给定一个已排序的链表的头 head ， 删除所有重复的元素，使每个元素只出现一次 。返回 已排序的链表 。
         *
         * */
//        ListNode cur = head;
//        while (cur != null && cur.next != null) {
//            while (cur.next != null && cur.val == cur.next.val) {
//                cur.next = cur.next.next;
//            }
//            cur = cur.next;
//        }
//        return head;
        /*
         * 82. 删除排序链表中的重复元素Ⅱ
         * 给定一个已排序的链表的头 head ， 删除原始链表中所有重复数字的节点，只留下不同的数字 。返回 已排序的链表 。
         * 有可能删除头节点，引入dummy
         * */
        if (head == null) return null;
        ListNode dummy = new ListNode(0, head);
        ListNode cur = dummy;
        while (cur.next.next != null) {
            if (cur.next.val == cur.next.next.val) {
                while (cur.next.next != null && cur.next.val == cur.next.next.val) {
                    cur.next.next = cur.next.next.next;
                }
                cur.next = cur.next.next;
            } else cur = cur.next;
        }
        return dummy.next;
    }

    public ListNode removeElements(ListNode head, int val) {
        /*
         * 203. 移除链表元素
         * 给你一个链表的头节点 head 和一个整数 val ，
         * 请你删除链表中所有满足 Node.val == val 的节点，并返回 新的头节点 。
         * */
        ListNode dummy = new ListNode(0, head);
        ListNode cur = dummy;
        while (cur.next != null) {
            if (cur.next.val == val) cur.next = cur.next.next;
            else cur = cur.next;
        }
        return dummy.next;
    }

    public ListNode modifiedList(int[] nums, ListNode head) {
        /*
        3217. 从链表中移除在数组中存在的节点
        * 给你一个整数数组 nums 和一个链表的头节点 head。从链表中移除所有存在于 nums 中的节点后，返回修改后的链表的头节点。
        * */
        Set<Integer> set = new HashSet<>(nums.length, 1);
        for (int num : nums) {
            set.add(num);
        }
        ListNode dummy = new ListNode(0, head);
        ListNode cur = dummy;
        while (cur.next != null) {
            if (set.contains(cur.next.val)) cur.next = cur.next.next;
            else cur = cur.next;
        }
        return dummy.next;
    }

    public ListNode removeNodes(ListNode head) {
        /*
        * 2487. 从链表中移除节点
        * 给你一个链表的头节点 head 。
        移除每个右侧有一个更大数值的节点。
        返回修改后链表的头节点 head 。
        * 思路：正难则反，将链表反转，题意转化为移除比当前节点小的节点
        * */
        head = reverseList(head);
        ListNode cur = head;
        while (cur.next != null) {
            int val = cur.val;
            if (cur.next.val < val) cur.next = cur.next.next;
            else cur = cur.next;
        }
        return reverseList(head);
    }

    public ListNode mergeInBetween(ListNode list1, int a, int b, ListNode list2) {
        /*
         * 1669. 合并两个链表
         * 给你两个链表 list1 和 list2 ，它们包含的元素分别为 n 个和 m 个。
         * 请你将 list1 中下标从 a 到 b 的全部节点都删除，并将list2 接在被删除节点的位置。
         * a>=1，list1头节点不会被删除
         * */
        ListNode left = list1, right = list1;
        for (int i = 0; i < a - 1; i++) {
            left = left.next;
        }
        for (int i = 0; i < b; i++) {
            right = right.next;
        }
        ListNode tail = list2;
        while (tail.next != null) {
            tail = tail.next;
        }
        left.next = list2;
        tail.next = right.next;
        return list1;
    }


}
