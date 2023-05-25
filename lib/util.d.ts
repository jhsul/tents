export interface NestedArray extends Array<number | NestedArray> {
}
type Buffer = Float32Array | Int32Array | number[];
export declare const arrEq: (a: Buffer, b: Buffer) => boolean;
/**
 * Recursively find the shape of a NestedArray
 */
export declare const findShape: (arr: NestedArray) => number[] | null;
export declare const gaussianSample: (mean?: number, stdev?: number) => number;
export {};
