/// <reference types="dist" />
import { NestedArray } from "./util";
export declare class Tensor {
    shape: Int32Array;
    data: Float32Array;
    isGpu: boolean;
    requiresGrad: boolean;
    dataBuffer?: GPUBuffer;
    grad?: Tensor;
    static _device: GPUDevice;
    constructor(arr?: NestedArray, requiresGrad?: boolean);
    static setupDevice(): Promise<void>;
    static zeros(shape: number | number[]): Tensor;
    static ones(shape: number | number[]): Tensor;
    static rand(shape: number | number[]): Tensor;
    static randn(shape: number | number[], mean?: number, stddev?: number): Tensor;
    get(idx: number | number[]): number;
    set(idx: number | number[], val: number): void;
    gpu(): this;
    static eq(a: Tensor, b: Tensor): boolean;
    static neg(a: Tensor): Tensor;
    static plus(a: Tensor, b: Tensor): Promise<Tensor>;
    static _cpuPlus(a: Tensor, b: Tensor): Tensor;
    static _gpuPlus(a: Tensor, b: Tensor): Promise<Tensor>;
    /**
     * Expects two matrices a, b with shapes [n, m], [m, p]
     * https://en.wikipedia.org/wiki/Matrix_multiplication_algorithm#Iterative_algorithm
     */
    static _cpuMatmul(a: Tensor, b: Tensor): Tensor;
    /**
     * Expects two tensors a, b with shapes [s, n, m], [s, m, p]
     * Performs a matrix multiplication on s pairs of matrices a[i,:,:], b[i,:,:]
     */
    static _cpuBatchMatmul(a: Tensor, b: Tensor): Tensor;
}
