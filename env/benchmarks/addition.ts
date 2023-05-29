import { Tensor } from "../../bin/";
import { Benchmark } from "../main";

export const vecaddForloopCpu: Benchmark = (n) => {
  const a = Tensor.rand(n);
  const b = Tensor.rand(n);

  const start = window.performance.now();

  const c = Tensor._cpu_forloop_plus(a, b);

  return window.performance.now() - start;
};
