package huawei.day03.p2;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;

public class Main {

    public static void main(String[] args) throws IOException {
        BufferedReader br = new BufferedReader(
                new InputStreamReader(System.in)
        );

        // 第1行：展平后的输入张量
        long[] input = parseLongArray(br.readLine());

        // 第2行：输入张量形状 N Cin H W
        int[] inputShape = parseIntArray(br.readLine());

        // 第3行：展平后的卷积核张量
        long[] kernel = parseLongArray(br.readLine());

        // 第4行：卷积核形状 Cout Ck Kh Kw
        int[] kernelShape = parseIntArray(br.readLine());

        // 第5行：分组数量
        int groups = Integer.parseInt(br.readLine().trim());

        // 形状必须包含4个维度
        if (inputShape.length != 4 || kernelShape.length != 4) {
            printError();
            return;
        }

        int batchSize = inputShape[0];
        int inChannels = inputShape[1];
        int inputHeight = inputShape[2];
        int inputWidth = inputShape[3];

        int outChannels = kernelShape[0];
        int kernelChannels = kernelShape[1];
        int kernelHeight = kernelShape[2];
        int kernelWidth = kernelShape[3];

        /*
         * 检查所有维度是否为正数。
         */
        if (batchSize <= 0
                || inChannels <= 0
                || inputHeight <= 0
                || inputWidth <= 0
                || outChannels <= 0
                || kernelChannels <= 0
                || kernelHeight <= 0
                || kernelWidth <= 0
                || groups <= 0) {
            printError();
            return;
        }

        /*
         * 检查展平后的数据数量是否与形状匹配。
         */
        long expectedInputSize =
                1L * batchSize * inChannels * inputHeight * inputWidth;

        long expectedKernelSize =
                1L * outChannels * kernelChannels
                        * kernelHeight * kernelWidth;

        if (input.length != expectedInputSize
                || kernel.length != expectedKernelSize) {
            printError();
            return;
        }

        /*
         * Group卷积的合法性检查：
         *
         * 1. 输入通道可以均匀分组；
         * 2. 输出通道可以均匀分组；
         * 3. 卷积核通道数等于每组输入通道数。
         */
        if (inChannels % groups != 0
                || outChannels % groups != 0
                || kernelChannels != inChannels / groups) {
            printError();
            return;
        }

        /*
         * stride = 1
         * padding = 0
         * dilation = 1
         */
        int outputHeight = inputHeight - kernelHeight + 1;
        int outputWidth = inputWidth - kernelWidth + 1;

        if (outputHeight <= 0 || outputWidth <= 0) {
            printError();
            return;
        }

        int inputChannelsPerGroup = inChannels / groups;
        int outputChannelsPerGroup = outChannels / groups;

        /*
         * 输出张量形状：
         * [batchSize, outChannels, outputHeight, outputWidth]
         */
        long outputSize =
                1L * batchSize * outChannels
                        * outputHeight * outputWidth;

        if (outputSize > Integer.MAX_VALUE) {
            printError();
            return;
        }

        long[] output = new long[(int) outputSize];

        /*
         * Group卷积计算。
         *
         * 循环层次：
         * batch
         *   -> group
         *      -> 当前组的输出通道
         *         -> 输出高度
         *            -> 输出宽度
         *               -> 当前组的输入通道
         *                  -> 卷积核高度
         *                     -> 卷积核宽度
         */
        for (int batch = 0; batch < batchSize; batch++) {
            for (int group = 0; group < groups; group++) {
                int inputChannelStart = group * inputChannelsPerGroup;
                int outputChannelStart = group * outputChannelsPerGroup;
                for (int localOutChannel = 0; localOutChannel < outputChannelsPerGroup; localOutChannel++) {
                    int outChannel = localOutChannel + outputChannelStart;
                    for (int outRow = 0; outRow < outputHeight; outRow++) {
                        for (int outCol = 0; outCol < outputWidth; outCol++) {
                            long sum = 0;
                            for (int kernelChannel = 0; kernelChannel < kernelChannels; kernelChannel++) {
                                int realInputChannel =
                                        inputChannelStart
                                                + kernelChannel;
                                for (int kernelRow = 0; kernelRow < kernelHeight; kernelRow++) {
                                    for (int kernelCol = 0; kernelCol < kernelWidth; kernelCol++) {
                                        int inputRow = outRow + kernelRow;
                                        int inputCol = outCol + kernelCol;

                                        int inputIndex = inputIndex(batch, realInputChannel, inputRow, inputCol, inChannels, inputHeight, inputWidth);
                                        int kernelIndex = kernelIndex(
                                                outChannel,
                                                kernelChannel,
                                                kernelRow,
                                                kernelCol,
                                                kernelChannels,
                                                kernelHeight,
                                                kernelWidth
                                        );
                                        sum += input[inputIndex]
                                                * kernel[kernelIndex];
                                    }
                                }
                            }



                            int outputIndex = outputIndex(
                                    batch,
                                    outChannel,
                                    outRow,
                                    outCol,
                                    outChannels,
                                    outputHeight,
                                    outputWidth
                            );

                            output[outputIndex] = sum;
                        }
                    }
                }
            }
        }

        // 第一行：输出展平后的张量
        StringBuilder dataResult = new StringBuilder();

        for (long value : output) {
            if (dataResult.length() > 0) {
                dataResult.append(' ');
            }
            dataResult.append(value);
        }

        System.out.println(dataResult);

        // 第二行：输出张量形状
        System.out.println(
                batchSize + " "
                        + outChannels + " "
                        + outputHeight + " "
                        + outputWidth
        );
    }

    /*
     * NCHW输入张量的一维下标：
     *
     * input[n][c][h][w]
     * →
     * ((n * C + c) * H + h) * W + w
     */
    private static int inputIndex(
            int batch,
            int channel,
            int row,
            int col,
            int channels,
            int height,
            int width) {

        return ((batch * channels + channel) * height + row)
                * width + col;
    }

    /*
     * 卷积核张量的一维下标：
     *
     * kernel[outChannel][kernelChannel][kh][kw]
     */
    private static int kernelIndex(
            int outChannel,
            int kernelChannel,
            int row,
            int col,
            int kernelChannels,
            int kernelHeight,
            int kernelWidth) {

        return ((outChannel * kernelChannels + kernelChannel)
                * kernelHeight + row) * kernelWidth + col;
    }

    /*
     * 输出张量的一维下标：
     *
     * output[n][outChannel][h][w]
     */
    private static int outputIndex(
            int batch,
            int outChannel,
            int row,
            int col,
            int outChannels,
            int outputHeight,
            int outputWidth) {

        return ((batch * outChannels + outChannel)
                * outputHeight + row) * outputWidth + col;
    }

    private static int[] parseIntArray(String line) {
        if (line == null || line.trim().isEmpty()) {
            return new int[0];
        }

        String[] parts = line.trim().split("\\s+");
        int[] result = new int[parts.length];

        for (int i = 0; i < parts.length; i++) {
            result[i] = Integer.parseInt(parts[i]);
        }

        return result;
    }

    private static long[] parseLongArray(String line) {
        if (line == null || line.trim().isEmpty()) {
            return new long[0];
        }

        String[] parts = line.trim().split("\\s+");
        long[] result = new long[parts.length];

        for (int i = 0; i < parts.length; i++) {
            result[i] = Long.parseLong(parts[i]);
        }

        return result;
    }

    private static void printError() {
        System.out.println("-1");
        System.out.println("-1");
    }
}