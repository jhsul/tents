import { testConstructors } from "./constructors.test";
import { testShapes } from "./shape.test";
import { testUtils } from "./util.test";
import { testAddition } from "./addition.test";
import { testMatmul } from "./matmul.test";
import { testBasics } from "./basics.test";
export const tests = [
    testConstructors,
    testShapes,
    testUtils,
    testAddition,
    testMatmul,
    testBasics,
];
