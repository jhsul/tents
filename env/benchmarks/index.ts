import { vecaddCpu, vecaddGpu } from "./addition";
import {
  matmulCpu,
  matmulGpu,
  batchMatmul4Cpu,
  batchMatmul64Cpu,
  batchMatmul256Cpu,
  batchMatmul4Gpu,
  batchMatmul64Gpu,
  batchMatmul256Gpu,
} from "./matmul";

export type Benchmark = (n: number) => Promise<number>;

export const benchmarks: Benchmark[] = [
  vecaddCpu,
  vecaddGpu,
  matmulCpu,
  matmulGpu,
  batchMatmul4Cpu,
  batchMatmul64Cpu,
  batchMatmul256Cpu,
  batchMatmul4Gpu,
  batchMatmul64Gpu,
  batchMatmul256Gpu,
];
