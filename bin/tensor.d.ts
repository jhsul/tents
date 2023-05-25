import { NestedArray } from "./util";
export declare class Tensor {
    shape: Int32Array;
    data: Float32Array;
    constructor(arr?: NestedArray);
    static zeros(shape: number | number[]): Tensor;
    static ones(shape: number | number[]): Tensor;
    static normal(shape: number | number[], mean?: number, stddev?: number): Tensor;
    cpu_plus(other: Tensor): Tensor;
}
