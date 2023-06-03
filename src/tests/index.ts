import { testConstructors } from "./constructors.test";
import { testShapes } from "./shape.test";
import { testUtils } from "./util.test";

type Test = () => void;

export const tests: Test[] = [testConstructors, testShapes, testUtils];
