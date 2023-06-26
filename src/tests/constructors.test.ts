import { Tensor } from "..";
import { assertArrayEquals, assertTrue, test } from "./testing";

export const testConstructors = async () => {
  await Promise.all([
    test("zeros constructor - 1D vector", () => {
      assertArrayEquals(Tensor.zeros(3).data, [0, 0, 0]);
    }),

    test("zeros constructor - 2D matrix", () => {
      assertArrayEquals(Tensor.zeros([2, 3]).data, [0, 0, 0, 0, 0, 0]);
    }),

    test("zeros constructor - 3D tensor", () => {
      assertArrayEquals(
        Tensor.zeros([2, 3, 4]).data,
        new Float32Array(24).fill(0)
      );
    }),

    test("ones constructor - 1D vector", () => {
      assertArrayEquals(Tensor.ones(3).data, [1, 1, 1]);
    }),

    test("ones constructor - 2D matrix", () => {
      assertArrayEquals(Tensor.ones([2, 3]).data, [1, 1, 1, 1, 1, 1]);
    }),

    test("ones constructor - 3D tensor", () => {
      assertArrayEquals(
        Tensor.ones([2, 3, 4]).data,
        new Float32Array(24).fill(1)
      );
    }),

    test("eye constructor", () => {
      const a = Tensor.eye(3);

      const expected = new Tensor([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
      ]);

      assertTrue(Tensor.almostEq(a, expected));
    }),
  ]);
};
