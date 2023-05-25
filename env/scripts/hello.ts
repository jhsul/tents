import { Tensor } from "../../bin/tensor";

export const helloTensor = (n: number) => {
  console.log("Hello, Tensors!");

  const a = Tensor.zeros(n);

  console.log(a);
};
