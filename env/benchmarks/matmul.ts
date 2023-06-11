import { Tensor } from "../../bin";
import { Benchmark } from ".";

export const matmulCpu: Benchmark = async (n) => {
  const a = Tensor.rand([n, n]);
  const b = Tensor.rand([n, n]);

  const start = window.performance.now();

  const c = Tensor._cpuMatmul(a, b);

  return window.performance.now() - start;
};

export const matmulGpu: Benchmark = async (n) => {
  const a = Tensor.rand([n, n]);
  const b = Tensor.rand([n, n]);

  const start = window.performance.now();

  const c = await Tensor._gpuMatmul(a.gpu(), b.gpu());

  return window.performance.now() - start;
};

const _batchMatmulCpu = async (n: number, size: number) => {
  const a = Tensor.rand([n, size, size]);
  const b = Tensor.rand([n, size, size]);

  const start = window.performance.now();

  const c = Tensor._cpuBatchMatmul(a, b);

  return window.performance.now() - start;
};

const _batchMatmulGpu = async (n: number, size: number) => {
  const a = Tensor.rand([n, size, size]);
  const b = Tensor.rand([n, size, size]);

  const start = window.performance.now();

  const c = await Tensor._gpuBatchMatmul(a.gpu(), b.gpu());

  return window.performance.now() - start;
};

export const batchMatmul4Cpu: Benchmark = async (n) => _batchMatmulCpu(n, 4);

export const batchMatmul64Cpu: Benchmark = async (n) => _batchMatmulCpu(n, 64);

export const batchMatmul256Cpu: Benchmark = async (n) =>
  _batchMatmulCpu(n, 256);

export const batchMatmul4Gpu: Benchmark = async (n) => _batchMatmulGpu(n, 4);

export const batchMatmul64Gpu: Benchmark = async (n) => _batchMatmulGpu(n, 64);

export const batchMatmul256Gpu: Benchmark = async (n) =>
  _batchMatmulGpu(n, 256);
