import { Tensor } from "../../bin/tensor";
import { Benchmark } from "../main";

export const helloTensor: Benchmark = (n) => {
  const start = Date.now();
  console.log("Hello, Tensors!");

  const a = Tensor.zeros(n);

  console.log(a);
  return Date.now() - start;
};
