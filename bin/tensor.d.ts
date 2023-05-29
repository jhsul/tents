import { NestedArray } from "./util";
export declare class Tensor {
    shape: Int32Array;
    data: Float32Array;
    constructor(arr?: NestedArray);
    static zeros(shape: number | number[]): Tensor;
    static ones(shape: number | number[]): Tensor;
    static rand(shape: number | number[]): Tensor;
    static randn(shape: number | number[], mean?: number, stddev?: number): Tensor;
    static _cpu_forloop_plus(a: Tensor, b: Tensor): Tensor;
}
