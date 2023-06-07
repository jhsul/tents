import { Tensor } from "..";
import { assertArrayEquals, test } from "./testing";
export const testAddition = async () => {
    test("1D vector cpu addition", () => {
        const a = new Tensor([1, 2, 3]);
        const b = new Tensor([5, 6, 1]);
        const c = Tensor._cpuPlus(a, b);
        assertArrayEquals(c.data, [6, 8, 4]);
        // assertArrayEquals(new Tensor([1, 2, 3]).shape, [3]);
    });
};
