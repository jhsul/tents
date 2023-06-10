import { Tensor } from "..";
import { assertEquals, assertFalse, assertTrue, test, } from "./testing";
export const testBasics = async () => {
    await Promise.all([
        test("equality operation", () => {
            const a = new Tensor([1, 2, 3]);
            const b = new Tensor([1, 2, 3]);
            assertTrue(Tensor.eq(a, b));
            // assertArrayEquals(new Tensor([1, 2, 3]).shape, [3]);
        }),
        test("equality operation - different data", () => {
            const a = new Tensor([1, 2, 3]);
            const b = new Tensor([1, 2, 4]);
            assertFalse(Tensor.eq(a, b));
        }),
        test("equality operation - different shapes", () => {
            const a = Tensor.zeros([2, 3]);
            const b = Tensor.zeros(6);
            assertFalse(Tensor.eq(a, b));
        }),
        test("accessing individual elements - 1D vector", () => {
            const a = new Tensor([1, 2, 3, 4, 5]);
            assertEquals(a.get(3), 4);
        }),
        test("accessing individual elements - 2D matrix", () => {
            const a = new Tensor([
                [1, 2, 3],
                [4, 5, 6],
            ]);
            assertEquals(a.get([1, 2]), 6);
        }),
        test("accessing individual elements - 3D tensor", () => {
            const a = new Tensor([
                [
                    [1, 2, 3],
                    [4, 5, 6],
                ],
                [
                    [7, 8, 9],
                    [10, 11, 12],
                ],
            ]);
            assertEquals(a.get([1, 1, 2]), 12);
            assertEquals(a.get([0, 1, 2]), 6);
        }),
        test("setting elements - 1D vector", () => {
            const a = new Tensor([1, 2, 3, 4, 5]);
            a.set(3, 10);
            assertEquals(a.get(3), 10);
        }),
        test("setting elements - 2D matrix", () => {
            const a = new Tensor([
                [1, 2, 3],
                [4, 5, 6],
            ]);
            a.set([1, 2], 10);
            assertEquals(a.get([1, 2]), 10);
        }),
    ]);
};
