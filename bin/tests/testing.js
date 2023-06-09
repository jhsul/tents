/**
 * @author Joshua Cuneo
 *
 * Modified from
 * Unit Test Your JavaScript Code Without a Framework
 * by Amit Gupta
 * https://javascript.plainenglish.io/unit-test-front-end-javascript-code-without-a-framework-8f00c63eb7d4
 */
import { arrEq } from "../util";
/**
 * Defines a unit test.
 * @param {string} desc The description of the test being run.
 * @param {function} fn The unit testing function, which should include a call to assert().
 */
export const test = async (desc, fn) => {
    try {
        // Run the test aysnchronously. If it's a synchronous function, this will still work
        await fn();
        console.log("%c%s", "color: #00AA00", "\u2713 " + desc);
    }
    catch (error) {
        console.log("%c%s", "color: #AA0000", "\u2718 " + desc);
        console.error(error);
    }
};
/**
 * Asserts a given condition is true.
 * @param {boolean} condition The condition to test.
 */
export function assertTrue(condition) {
    if (!condition) {
        let e = new Error();
        e.message = "Expected TRUE but was FALSE";
        throw e;
    }
}
/**
 * Asserts a given condition is false.
 * @param {boolean} condition The condition to test.
 */
export function assertFalse(condition) {
    if (condition) {
        let e = new Error();
        e.message = "Expected FALSE but was TRUE";
        throw e;
    }
}
/**
 * Asserts an expected value is equal to an actual value.
 * @param expected The expected value from the computation.
 * @param actual The actual value from the computation.
 */
export function assertEquals(expected, actual) {
    let expectedType = Object.prototype.toString.call(expected);
    let actualType = Object.prototype.toString.call(actual);
    //Don't let the user use assertEquals to compare two arrays
    if (expectedType === "[object Array]" && actualType === "[object Array]") {
        let e = new Error();
        e.message =
            "Both parameters are arrays.\n" +
                "Fix parameters or use arrayAssertEquals() instead.";
        throw e;
    }
    //Run the comparison
    if (expected !== actual) {
        let e = new Error();
        e.message =
            "Expected " +
                expected +
                " of type " +
                cleanType(expectedType) +
                "\n but was " +
                actual +
                " of type " +
                cleanType(actualType);
        throw e;
    }
    //Clean up the type so that it's easier to read
    function cleanType(typeString) {
        return typeString.split(" ")[1].replace("]", "");
    }
}
/**
 * Asserts that two arrays are equal.
 * We define array equality as the same elements in the same order.
 * @param {array} expected The expected array from the computation.
 * @param {array} actual The actual array from the computation.
 */
export function assertArrayEquals(expected, actual) {
    if (expected.length !== actual.length) {
        throw new Error("Arrays are of different lengths");
    }
    if (!arrEq(expected, actual)) {
        // console.log(expected);
        // console.log(actual);
        throw new Error("Arrays are not equal");
    }
}
/**
 * Asserts a given value is null.
 * @param value The value to test.
 */
export function assertNull(value) {
    if (value !== null) {
        let e = new Error();
        e.message = "Expected NULL but was " + value;
        throw e;
    }
}
