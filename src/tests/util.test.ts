import { Tensor, arrEq } from "..";
import { assertArrayEquals, assertFalse, assertTrue, test } from "./testing";

export const testUtils = async () => {
  await Promise.all([
    test("arrEq function", () => {
      assertTrue(arrEq([1, 2, 3], [1, 2, 3]));
    }),

    test("arrEq function - different array types", () => {
      assertTrue(arrEq(new Float32Array([1, 2, 3]), [1, 2, 3]));
    }),

    test("arrEq function - unequal arrays", () => {
      assertFalse(arrEq([1, 2, 3], [1, 2, 4]));
    }),
  ]);
};
