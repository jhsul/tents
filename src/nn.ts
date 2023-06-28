import { Tensor } from ".";

export class Module {
  //   constructor() {}
}

export class Linear extends Module {
  weight: Tensor;
  bias: Tensor;
  constructor(dimA: number, dimB: number) {
    super();

    this.weight = Tensor.rand([dimA, dimB]);
    this.bias = Tensor.rand([dimB, 1]);
  }

  // Return xAt + b
  // Where A = weights and b = bias
  async forward(x: Tensor) {
    return await Tensor.plus(
      await Tensor.matmul(x, this.weight.T()),
      this.bias
    );
  }
}
