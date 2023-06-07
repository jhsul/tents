import { arrEq } from "..";
import { assertFalse, assertTrue, test } from "./testing";
export const testUtils = async () => {
    test("arrEq function", () => {
        assertTrue(arrEq([1, 2, 3], [1, 2, 3]));
    });
    test("arrEq function - different array types", () => {
        assertTrue(arrEq(new Float32Array([1, 2, 3]), [1, 2, 3]));
    });
    test("arrEq function - unequal arrays", () => {
        assertFalse(arrEq([1, 2, 3], [1, 2, 4]));
    });
};
