import { Tensor } from "..";
import { assertArrayEquals, test } from "./testing";
export const testShapes = async () => {
    test("1D vector shape", () => {
        assertArrayEquals(new Tensor([1, 2, 3]).shape, [3]);
    });
    test("2D matrix shape", () => {
        assertArrayEquals(new Tensor([
            [1, 1, 1],
            [1, 1, 1],
        ]).shape, [2, 3]);
    });
    test("3D tensor shape", () => {
        assertArrayEquals(new Tensor([
            [
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
            ],
            [
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
            ],
        ]).shape, [2, 3, 4]);
    });
};
