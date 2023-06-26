import { Tensor } from ".";

export class Module {
  //   constructor() {}
}

export class Linear extends Module {
  weights: Tensor;
  bias: Tensor;
  constructor(dimA: number, dimB: number) {
    super();

    this.weights = Tensor.rand([dimA, dimB]);
    this.bias = Tensor.rand([dimB, 1]);
  }

  // Return xA + b
  // Where A = weights and b = bias
  async forward(x: Tensor) {
    return await Tensor.plus(await Tensor.matmul(x, this.weights), this.bias);
  }
}
