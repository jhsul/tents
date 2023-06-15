import { Tensor } from "..";
import { assertArrayEquals, assertTrue, test } from "./testing";

export const testMatmul = async () => {
  await Promise.all([
    test("basic matmul 1/3", () => {
      const a = new Tensor([
        [1, 2, 3],
        [4, 5, 6],
      ]);

      const b = new Tensor([
        [7, 8],
        [9, 10],
        [11, 12],
      ]);

      const c = Tensor._cpuMatmul(a, b);

      const expected = new Tensor([
        [58, 64],
        [139, 154],
      ]);

      assertTrue(Tensor.eq(c, expected));
    }),

    test("basic matmul 2/3", () => {
      const a = new Tensor([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
      ]);

      const b = new Tensor([
        [17, 18, 19, 20],
        [21, 22, 23, 24],
        [25, 26, 27, 28],
        [29, 30, 31, 32],
      ]);

      const c = Tensor._cpuMatmul(a, b);

      const expected = new Tensor([
        [250, 260, 270, 280],
        [618, 644, 670, 696],
        [986, 1028, 1070, 1112],
        [1354, 1412, 1470, 1528],
      ]);

      assertTrue(Tensor.eq(c, expected));
    }),

    test("basic matmul 3/3", () => {
      const a = new Tensor([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
      ]);

      const b = new Tensor([[17], [18], [19], [20]]);

      const c = Tensor._cpuMatmul(a, b);

      const expected = new Tensor([[190], [486], [782], [1078]]);

      assertTrue(Tensor.eq(c, expected));
    }),

    // test("gpu basic matmul", async () => {
    //   // await Tensor.setupDevice();
    //   const a = new Tensor([
    //     [1, 2, 3],
    //     [4, 5, 6],
    //   ]);

    //   const b = new Tensor([
    //     [7, 8],
    //     [9, 10],
    //     [11, 12],
    //   ]);

    //   const c = await Tensor._gpuMatmul(a.gpu(), b.gpu());

    //   const expected = new Tensor([
    //     [58, 64],
    //     [139, 154],
    //   ]);

    //   assertTrue(Tensor.eq(c, expected));
    // }),
    // test("gpu matmul 4x4 and 4x1", async () => {
    //   // await Tensor.setupDevice();
    //   const a = new Tensor([
    //     [1, 2, 3, 4],
    //     [5, 6, 7, 8],
    //     [9, 10, 11, 12],
    //     [13, 14, 15, 16],
    //   ]);

    //   const b = new Tensor([[1], [2], [3], [4]]);

    //   const c = await Tensor._gpuMatmul(a.gpu(), b.gpu());

    //   const expected = new Tensor([[30], [70], [110], [150]]);

    //   assertTrue(Tensor.eq(c, expected));
    // }),

    // test("gpu matmul large", async () => {
    //   const a = Tensor.rand([1000, 1000]);
    //   const b = Tensor.rand([1000, 1000]);

    //   const expected = Tensor._cpuMatmul(a, b);

    //   const c = await Tensor._gpuMatmul(a.gpu(), b.gpu());

    //   assertTrue(Tensor.almostEq(c, expected));
    // }),

    test("cpu batch matmul [2, 2, 2], [2, 2, 2]", async () => {
      const a = new Tensor([
        [
          [1, 2],
          [3, 4],
        ],
        [
          [5, 6],
          [7, 8],
        ],
      ]);

      const b = new Tensor([
        [
          [1, 2],
          [3, 4],
        ],
        [
          [5, 6],
          [7, 8],
        ],
      ]);

      const c = Tensor._cpuBatchMatmul(a, b);

      const expected = new Tensor([
        [
          [7, 10],
          [15, 22],
        ],
        [
          [67, 78],
          [91, 106],
        ],
      ]);

      // console.log(c);
      // console.log(expected);

      assertTrue(Tensor.eq(c, expected));
    }),
    test("cpu batch matmul [1, 2, 2], [2, 2, 2]", async () => {
      const a = new Tensor([
        [
          [1, 2],
          [3, 4],
        ],
      ]);

      const b = new Tensor([
        [
          [1, 2],
          [3, 4],
        ],
        [
          [5, 6],
          [7, 8],
        ],
      ]);

      const c = Tensor._cpuBatchMatmul(a, b);

      const expected = new Tensor([
        [
          [7, 10],
          [15, 22],
        ],
        [
          [19, 22],
          [43, 50],
        ],
      ]);

      assertTrue(Tensor.almostEq(c, expected));
    }),
    test("cpu batch matmul [2, 2, 2], [1, 2, 2]", async () => {
      const a = new Tensor([
        [
          [1, 2],
          [3, 4],
        ],
        [
          [5, 6],
          [7, 8],
        ],
      ]);

      const b = new Tensor([
        [
          [1, 2],
          [3, 4],
        ],
      ]);

      const c = Tensor._cpuBatchMatmul(a, b);

      const expected = new Tensor([
        [
          [7, 10],
          [15, 22],
        ],
        [
          [23, 34],
          [31, 46],
        ],
      ]);

      assertTrue(Tensor.almostEq(c, expected));
    }),
    test("cpu batch matmul [1, 3, 4], [3, 4, 2]", async () => {
      const a = new Tensor([
        [
          [1, 2, 3, 4],
          [5, 6, 7, 8],
          [9, 10, 11, 12],
        ],
      ]);

      const b = new Tensor([
        [
          [1, 2],
          [3, 4],
          [5, 6],
          [7, 8],
        ],
        [
          [9, 10],
          [11, 12],
          [13, 14],
          [15, 16],
        ],
        [
          [17, 18],
          [19, 20],
          [21, 22],
          [23, 24],
        ],
      ]);

      const c = Tensor._cpuBatchMatmul(a, b);

      const expected = new Tensor([
        [
          [50, 60],
          [114, 140],
          [178, 220],
        ],
        [
          [130, 140],
          [322, 348],
          [514, 556],
        ],
        [
          [210, 220],
          [530, 556],
          [850, 892],
        ],
      ]);

      // console.log(c);
      // console.log(expected);

      assertTrue(Tensor.almostEq(c, expected));
    }),

    // test("gpu batch matmul [2, 2, 2], [2, 2, 2]", async () => {
    //   const a = new Tensor([
    //     [
    //       [1, 2],
    //       [3, 4],
    //     ],
    //     [
    //       [5, 6],
    //       [7, 8],
    //     ],
    //   ]);

    //   const b = new Tensor([
    //     [
    //       [1, 2],
    //       [3, 4],
    //     ],
    //     [
    //       [5, 6],
    //       [7, 8],
    //     ],
    //   ]);

    //   const c = await Tensor._gpuBatchMatmul(a.gpu(), b.gpu());

    //   const expected = new Tensor([
    //     [
    //       [7, 10],
    //       [15, 22],
    //     ],
    //     [
    //       [67, 78],
    //       [91, 106],
    //     ],
    //   ]);

    //   // console.log(c);
    //   // console.log(expected);

    //   assertTrue(Tensor.almostEq(c, expected));
    // }),
    // test("gpu batch matmul [1, 2, 2], [2, 2, 2]", async () => {
    //   const a = new Tensor([
    //     [
    //       [1, 2],
    //       [3, 4],
    //     ],
    //   ]);

    //   const b = new Tensor([
    //     [
    //       [1, 2],
    //       [3, 4],
    //     ],
    //     [
    //       [5, 6],
    //       [7, 8],
    //     ],
    //   ]);

    //   const c = await Tensor._gpuBatchMatmul(a.gpu(), b.gpu());

    //   const expected = new Tensor([
    //     [
    //       [7, 10],
    //       [15, 22],
    //     ],
    //     [
    //       [19, 22],
    //       [43, 50],
    //     ],
    //   ]);

    //   // console.log(c);
    //   // console.log(expected);

    //   assertTrue(Tensor.almostEq(c, expected));
    // }),

    // test("gpu batch matmul [2, 2, 2], [1, 2, 2]", async () => {
    //   const a = new Tensor([
    //     [
    //       [1, 2],
    //       [3, 4],
    //     ],
    //     [
    //       [5, 6],
    //       [7, 8],
    //     ],
    //   ]);

    //   const b = new Tensor([
    //     [
    //       [1, 2],
    //       [3, 4],
    //     ],
    //   ]);

    //   const c = await Tensor._gpuBatchMatmul(a.gpu(), b.gpu());

    //   const expected = new Tensor([
    //     [
    //       [7, 10],
    //       [15, 22],
    //     ],
    //     [
    //       [23, 34],
    //       [31, 46],
    //     ],
    //   ]);

    //   assertTrue(Tensor.almostEq(c, expected));
    // }),
    // test("gpu batch matmul [1, 3, 4], [3, 4, 2]", async () => {
    //   const a = new Tensor([
    //     [
    //       [1, 2, 3, 4],
    //       [5, 6, 7, 8],
    //       [9, 10, 11, 12],
    //     ],
    //   ]);

    //   const b = new Tensor([
    //     [
    //       [1, 2],
    //       [3, 4],
    //       [5, 6],
    //       [7, 8],
    //     ],
    //     [
    //       [9, 10],
    //       [11, 12],
    //       [13, 14],
    //       [15, 16],
    //     ],
    //     [
    //       [17, 18],
    //       [19, 20],
    //       [21, 22],
    //       [23, 24],
    //     ],
    //   ]);

    //   const c = await Tensor._gpuBatchMatmul(a.gpu(), b.gpu());

    //   const expected = new Tensor([
    //     [
    //       [50, 60],
    //       [114, 140],
    //       [178, 220],
    //     ],
    //     [
    //       [130, 140],
    //       [322, 348],
    //       [514, 556],
    //     ],
    //     [
    //       [210, 220],
    //       [530, 556],
    //       [850, 892],
    //     ],
    //   ]);

    //   assertTrue(Tensor.almostEq(c, expected));
    // }),

    // test("gpu batch matmul random (small)", async () => {
    //   const a = Tensor.rand([2, 3, 4]);
    //   const b = Tensor.rand([2, 4, 5]);

    //   const expected = Tensor._cpuBatchMatmul(a, b);
    //   const c = await Tensor._gpuBatchMatmul(a.gpu(), b.gpu());

    //   assertTrue(Tensor.almostEq(c, expected));
    // }),

    // test("gpu batch matmul random (large)", async () => {
    //   const a = Tensor.rand([2, 100, 100]);
    //   const b = Tensor.rand([2, 100, 100]);

    //   const expected = Tensor._cpuBatchMatmul(a, b);
    //   const c = await Tensor._gpuBatchMatmul(a.gpu(), b.gpu());

    //   assertTrue(Tensor.almostEq(c, expected));
    // }),

    // test("top-level matmul method", async () => {
    //   const a = Tensor.rand([2, 3, 4]);
    //   const b = Tensor.rand([2, 4, 5]);

    //   const cpuExpected = Tensor._cpuBatchMatmul(a, b);
    //   const cpu = await Tensor.matmul(a, b);

    //   const gpuExpected = await Tensor._gpuBatchMatmul(a.gpu(), b.gpu());
    //   const gpu = await Tensor.matmul(a.gpu(), b.gpu());

    //   assertTrue(Tensor.almostEq(cpu, cpuExpected));
    //   assertTrue(Tensor.almostEq(gpu, gpuExpected));
    // }),
  ]);
};
