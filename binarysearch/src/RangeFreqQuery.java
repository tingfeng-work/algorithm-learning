import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.function.BiFunction;

public class RangeFreqQuery {

    /*
     * 请你设计一个数据结构，它能求出给定子数组内一个给定值的 频率 。
     * 子数组中一个值的 频率 指的是这个子数组中这个值的出现次数。
     * 我的思路：首先是暴力做法，即从 left 到 right 挨个遍历，遇到值为 value 的 ans++，该做法超时；
     * 进一步想到用二分查找优化，由于是计算频率，所以顺序不重要，故构造子数组从left到right，在子数组上排序后用二分查找
     * 同样超时，不难分析，当子数组很大时，每次query都要创建新的子数组并排序，必然超时
     * 标准答案：同样是利用二分查找，有序数组的构造不同，我是从left到right构造有序数组，而标准答案则是利用数组中下标递增的这个性质，
     * 维护一个 hashmap，key 是数组中的元素值，value 是该元素值在数组中出现的下标数组；
     * 这样每次查找只需要在hashmap中找到对应元素值的下标数组，在下标数组中利用二分查找，这样每次query耗时减少到O(logn)，n是下标数组的大小
     * */

    private final HashMap<Integer, ArrayList<Integer>> map = new HashMap<>();

    public RangeFreqQuery(int[] arr) {
        // RangeFreqQuery(int[] arr) 用下标从 0 开始的整数数组 arr 构造一个类的实例。
        for (int i = 0; i < arr.length; i++) {
            map.computeIfAbsent(arr[i], k -> {
                return new ArrayList<>();
            }).add(i);
        }
    }

    public int query(int left, int right, int value) {
        //int query(int left, int right, int value) 返回子数组 arr[left...right] 中 value 的 频率 。
        ArrayList<Integer> arrayList = map.get(value);
        if (arrayList == null || arrayList.size() == 0) return 0;
        int index1 = binarySearch(arrayList, left);
        int index2 = binarySearch(arrayList, right + 1);
        return index2 - index1;
    }

    private int binarySearch(ArrayList<Integer> arrayList, int target) {
        int left = -1;
        int right = arrayList.size();
        while (left + 1 < right) {
            // 开区间
            int mid = left + (right - left) / 2;
            int x = arrayList.get(mid);
            if (x < target) left = mid;
            else right = mid;
        }
        return right;
    }
}
