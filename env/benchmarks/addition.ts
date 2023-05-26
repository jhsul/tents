import { Tensor } from "../../bin/";
import { Benchmark } from "../main";

export const cpuForLoopAddition1D: Benchmark = (n) => {
  const a = Tensor.normal(n);
  const b = Tensor.normal(n);

  const c = new Float32Array(n);
  const start = window.performance.now();

  for (let i = 0; i < n; i++) {
    c[i] = a.data[i] + b.data[i];
  }

  return window.performance.now() - start;
};
