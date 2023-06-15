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
    }),

    test("elmult", () => {
      const a = new Tensor([1, 2, 3]);
      const b = new Tensor([2, 4, 6]);

      assertTrue(Tensor.almostEq(a.elmult(b), new Tensor([2, 8, 18])));
    }),

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
      a.T();
      // console.log(a);
      assertTrue(Tensor.almostEq(a, b));
    }),

    test("2D matrix transpose 1x10", () => {
      const a = Tensor.ones([1, 10]);
      const b = Tensor.ones([10, 1]);

      assertTrue(Tensor.eq(a.T(), b));
    }),
  ]);
};
