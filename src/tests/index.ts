import { testConstructors } from "./constructors.test";
import { testShapes } from "./shape.test";
import { testUtils } from "./util.test";
import { testAddition } from "./addition.test";
import { testMatmul } from "./matmul.test";
import { testBasics } from "./basics.test";
import { testPrimitives } from "./prims.test";

// All tests are asynchronous by default
type Test = () => Promise<void>;

export const tests: Test[] = [
  testConstructors,
  testShapes,
  testUtils,
  testAddition,
  testMatmul,
  testBasics,
  testPrimitives,
];
