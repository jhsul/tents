import { Tensor } from "..";
import { assertArrayEquals, assertTrue, test } from "./testing";

export const testAddition = async () => {
  await Promise.all([
    test("1D vector cpu addition", () => {
      const a = new Tensor([1, 2, 3]);
      const b = new Tensor([5, 6, 1]);

      const c = Tensor._cpuPlus(a, b);

      assertArrayEquals(c.data, [6, 8, 4]);

      // assertArrayEquals(new Tensor([1, 2, 3]).shape, [3]);
    }),

    // test("1D vector gpu addition (small)", async () => {
    //   // await Tensor.setupDevice();
    //   const a = new Tensor([1, 2, 3]);
    //   const b = new Tensor([5, 6, 1]);

    //   const c = await Tensor._gpuPlus(a.gpu(), b.gpu());

    //   // await new Promise((r) => setTimeout(r, 1000));

    //   const expected = new Tensor([6, 8, 4]);
    //   // console.log(c.data);
    //   // console.log(c);
    //   // console.log(expected);

    //   assertTrue(Tensor.eq(c, expected));
    //   // assertArrayEquals(c.data, new Float32Array([6, 8, 4]));

    //   // assertArrayEquals(new Tensor([1, 2, 3]).shape, [3]);
    // }),

    // test("1D vector gpu addition (large)", async () => {
    //   // await Tensor.setupDevice();
    //   const a = new Tensor(new Array(1000000).fill(1));
    //   const b = new Tensor(new Array(1000000).fill(2));

    //   const c = await Tensor._gpuPlus(a.gpu(), b.gpu());

    //   // await new Promise((r) => setTimeout(r, 1000));

    //   const expected = new Tensor(new Array(1000000).fill(3));
    //   // console.log(c.data);
    //   // console.log(c);
    //   // console.log(expected);

    //   assertTrue(Tensor.eq(c, expected));
    //   // assertArrayEquals(c.data, new Float32Array([6, 8, 4]));

    //   // assertTrue(Tensor.eq(a, expected));
    //   // assertArrayEquals(new Tensor([1, 2, 3]).shape, [3]);
    // }),
  ]);
};
