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
    gpu(): this;
    static _cpu_forloop_plus(a: Tensor, b: Tensor): Tensor;
    static _gpu_plus(a: Tensor, b: Tensor): Promise<Tensor>;
}
