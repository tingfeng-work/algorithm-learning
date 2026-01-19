public class Test {
    public static void main(String[] args) {
        Solution so = new Solution();
//        System.out.println(so.shortestCommonSupersequence("bbbaaaba", "bbababbb"));
//        System.out.println("bbbaaababbb".length());
//        String s = "01234";
//        char[] chars = s.toCharArray();
//        System.out.println(new String(chars,0,3));
//        so.longestObstacleCourseAtEachPosition(new int[]{1, 2, 3, 2});
        int[][] envelopes = {
                {5, 4},
                {6, 4},
                {6, 7},
                {2, 3}
        };
        so.maxEnvelopes(envelopes);
    }

}
