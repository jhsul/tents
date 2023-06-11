import { Tensor } from "../../bin/";
import { Benchmark } from ".";

export const vecaddCpu: Benchmark = async (n) => {
  const a = Tensor.rand(n);
  const b = Tensor.rand(n);

  const start = window.performance.now();

  const c = Tensor._cpuPlus(a, b);

  return window.performance.now() - start;
};

export const vecaddGpu: Benchmark = async (n) => {
  const a = Tensor.rand(n);
  const b = Tensor.rand(n);

  const start = window.performance.now();

  const c = await Tensor._gpuPlus(a.gpu(), b.gpu());

  // console.log(a);
  // console.log(b);
  // console.log(c);

  return window.performance.now() - start;
};
