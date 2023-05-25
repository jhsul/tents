export interface NestedArray extends Array<number | NestedArray> {
}
type ArrayType = Float32Array | Int32Array | number[];
/**
 * Shallow equality check for two array types
 * Note: a number[] and a Float32Array can return true if the elements are the same
 */
export declare const arrEq: (a: ArrayType, b: ArrayType) => boolean;
/**
 * Recursively find the shape of a NestedArray
 */
export declare const findShape: (arr: NestedArray) => number[] | null;
/**
 * Uses the Box Muller Transform to sample from a normal distribution
 * https://stackoverflow.com/a/36481059
 */
export declare const gaussianSample: (mean?: number, stdev?: number) => number;
/**
 * Compute the mean value of an array one element at a time
 * Avoids a potential overflow error
 * Although, this is probably never really an issue with floats...
 * https://stackoverflow.com/a/72565782
 */
export declare const mean: (arr: ArrayType) => number;
export declare const stddev: (arr: ArrayType) => number;
export {};
