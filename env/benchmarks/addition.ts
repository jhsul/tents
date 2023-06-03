import { Tensor } from "../../bin/";
import { Benchmark } from "../main";

export const vecaddForloopCpu: Benchmark = async (n) => {
  const a = Tensor.rand(n);
  const b = Tensor.rand(n);

  const start = window.performance.now();

  const c = Tensor._cpu_forloop_plus(a, b);

  return window.performance.now() - start;
};

export const vecaddWebGpu: Benchmark = async (n) => {
  const a = Tensor.rand(n);
  const b = Tensor.rand(n);

  const start = window.performance.now();

  const c = await Tensor._gpu_plus(a.gpu(), b.gpu());

  // console.log(a);
  // console.log(b);
  // console.log(c);

  return window.performance.now() - start;
};
