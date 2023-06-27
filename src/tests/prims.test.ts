import { Tensor } from "..";
import { assertArrayEquals, assertTrue, test } from "./testing";

export const testPrimitives = async () => {
  await Promise.all([
    test("negation", () => {
      const a = new Tensor([1, 2, 3]);
      const b = new Tensor([-1, -2, -3]);

      assertTrue(Tensor.eq(a, b.neg()));

      // assertArrayEquals(new Tensor([1, 2, 3]).shape, [3]);
    }),

    test("scaling", () => {
      const a = new Tensor([1, 2, 3]);
      const b = new Tensor([2, 4, 6]);

      assertTrue(Tensor.almostEq(a.scale(2), b));
      assertTrue(Tensor.almostEq(a, b.scale(0.5)));
    }),

    // test("elmult", () => {
    //   const a = new Tensor([1, 2, 3]);
    //   const b = new Tensor([2, 4, 6]);

    //   assertTrue(Tensor.almostEq(a.elmult(b), new Tensor([2, 8, 18])));
    // }),

    test("exp", () => {
      const a = new Tensor([1, 2, 3]);
      const b = new Tensor([Math.exp(1), Math.exp(2), Math.exp(3)]);

      assertTrue(Tensor.almostEq(a.exp(), b));
    }),

    test("2D matrix transpose 2x3", () => {
      const a = new Tensor([
        [1, 2, 3],
        [4, 5, 6],
      ]);

      const b = new Tensor([
        [1, 4],
        [2, 5],
        [3, 6],
      ]);

      // console.log(a);
      // console.log(a);
      assertTrue(Tensor.almostEq(a.T(), b));
      assertTrue(Tensor.almostEq(b.T(), a));
    }),

    test("2D matrix transpose 1x10", () => {
      const a = Tensor.ones([1, 10]);
      const b = Tensor.ones([10, 1]);

      assertTrue(Tensor.eq(a.T(), b));
    }),

    test("relu", () => {
      const a = new Tensor([-1, 0, 2, 3]);
      const expected = new Tensor([0, 0, 2, 3]);

      assertTrue(Tensor.eq(a.relu(), expected));
    }),

    test("softmax", () => {
      const a = new Tensor([
        [1, 2, 1],
        [100, 300, 0],
      ]);

      const expected = new Tensor([
        [0.2119, 5.7612e-1, 2.1194e-1],
        [0, 1.0, 0.0],
      ]);

      assertTrue(Tensor.almostEq(a.softmax(), expected));
    }),
  ]);
};
