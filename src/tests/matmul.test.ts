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

    test("gpu basic matmul 1/3", async () => {
      // await Tensor.setupDevice();
      const a = new Tensor([
        [1, 2, 3],
        [4, 5, 6],
      ]);

      const b = new Tensor([
        [7, 8],
        [9, 10],
        [11, 12],
      ]);

      const c = await Tensor._gpuMatmul(a.gpu(), b.gpu());

      const expected = new Tensor([
        [58, 64],
        [139, 154],
      ]);

      assertTrue(Tensor.eq(c, expected));
    }),
    test("gpu matmul 4x4 and 4x1", async () => {
      // await Tensor.setupDevice();
      const a = new Tensor([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
      ]);

      const b = new Tensor([[1], [2], [3], [4]]);

      const c = await Tensor._gpuMatmul(a.gpu(), b.gpu());

      const expected = new Tensor([[30], [70], [110], [150]]);

      assertTrue(Tensor.eq(c, expected));
    }),
  ]);
};
