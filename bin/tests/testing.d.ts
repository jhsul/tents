/**
 * @author Joshua Cuneo
 *
 * Modified from
 * Unit Test Your JavaScript Code Without a Framework
 * by Amit Gupta
 * https://javascript.plainenglish.io/unit-test-front-end-javascript-code-without-a-framework-8f00c63eb7d4
 */
import { type ArrayType } from "../util";
/**
 * Defines a unit test.
 * @param {string} desc The description of the test being run.
 * @param {function} fn The unit testing function, which should include a call to assert().
 */
export declare const test: (desc: string, fn: () => Promise<void> | void) => Promise<void>;
/**
 * Asserts a given condition is true.
 * @param {boolean} condition The condition to test.
 */
export declare function assertTrue(condition: boolean): void;
/**
 * Asserts a given condition is false.
 * @param {boolean} condition The condition to test.
 */
export declare function assertFalse(condition: boolean): void;
/**
 * Asserts an expected value is equal to an actual value.
 * @param expected The expected value from the computation.
 * @param actual The actual value from the computation.
 */
export declare function assertEquals(expected: any, actual: any): void;
/**
 * Asserts that two arrays are equal.
 * We define array equality as the same elements in the same order.
 * @param {array} expected The expected array from the computation.
 * @param {array} actual The actual array from the computation.
 */
export declare function assertArrayEquals(expected: ArrayType, actual: ArrayType): void;
/**
 * Asserts a given value is null.
 * @param value The value to test.
 */
export declare function assertNull(value: any): void;
