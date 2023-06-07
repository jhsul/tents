import { Tensor } from "..";
import { assertArrayEquals, assertTrue, test } from "./testing";

export const testMatmul = async () => {
  test("basic matmul", () => {
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
  });
};
