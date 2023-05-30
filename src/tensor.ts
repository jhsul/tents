import { arrEq, findShape, gaussianSample, NestedArray } from "./util";

export class Tensor {
  shape: Int32Array;
  data: Float32Array;

  isGpu: boolean;
  requiresGrad: boolean;

  grad?: Tensor;

  // Constructors

  constructor(arr?: NestedArray, requiresGrad: boolean = false) {
    /*
    Parse the shape from the array, or throw an exception
    */

    this.isGpu = false;
    this.requiresGrad = requiresGrad;

    // An empty tensor
    if (!arr) {
      this.shape = new Int32Array();
      this.data = new Float32Array();
    }
    // Tensor is given initial values
    else {
      const shape = findShape(arr);
      if (!shape) throw new Error("Invalid shape");

      this.shape = new Int32Array(shape);

      //@ts-expect-error
      this.data = new Float32Array(arr.flat(shape.length));
    }

    // Initialize gradient if necessary
    if (requiresGrad) {
      this.grad = Tensor.zeros(Array.from(this.shape));
    }
  }

  static zeros(shape: number | number[]): Tensor {
    if (typeof shape === "number") {
      shape = [shape];
    } else if (!Array.isArray(shape)) {
      throw new Error("Invalid shape");
    }

    const t = new Tensor();

    t.shape = new Int32Array(shape);
    // TypedArrays are initialized to 0 by default
    t.data = new Float32Array(shape.reduce((a, b) => a * b, 1));
    return t;
  }

  static ones(shape: number | number[]): Tensor {
    if (typeof shape === "number") {
      shape = [shape];
    } else if (!Array.isArray(shape)) {
      throw new Error("Invalid shape");
    }

    const t = new Tensor();

    t.shape = new Int32Array(shape);
    t.data = new Float32Array(shape.reduce((a, b) => a * b, 1)).fill(1.0);
    return t;
  }

  static rand(shape: number | number[]): Tensor {
    if (typeof shape === "number") {
      shape = [shape];
    } else if (!Array.isArray(shape)) {
      throw new Error("Invalid shape");
    }

    const t = new Tensor();

    t.shape = new Int32Array(shape);
    t.data = new Float32Array(shape.reduce((a, b) => a * b, 1)).map(() =>
      Math.random()
    );

    return t;
  }

  static randn(shape: number | number[], mean: number = 0, stddev: number = 1) {
    if (typeof shape === "number") {
      shape = [shape];
    } else if (!Array.isArray(shape)) {
      throw new Error("Invalid shape");
    }

    const t = new Tensor();

    t.shape = new Int32Array(shape);
    t.data = new Float32Array(shape.reduce((a, b) => a * b, 1)).map(() =>
      gaussianSample(mean, stddev)
    );
    return t;
  }

  // CPU Operations

  static _cpu_forloop_plus(a: Tensor, b: Tensor): Tensor {
    if (!arrEq(a.shape, b.shape)) throw new Error("Shape mismatch");

    const t = new Tensor();

    t.shape = new Int32Array(a.shape);
    t.data = new Float32Array(a.data.length);

    for (let i = 0; i < a.data.length; i++) {
      t.data[i] = a.data[i] + b.data[i];
    }

    return t;
  }

  // cpu_plus(other: Tensor): Tensor {
  // if (!arrEq(this.shape, other.shape)) throw new Error("Shape mismatch");

  // const t = new Tensor();

  // t.shape = new Int32Array(this.shape);
  // t.data = new Float32Array(this.data.length);

  // for (let i = 0; i < this.data.length; i++) {
  //   t.data[i] = this.data[i] + other.data[i];
  // }

  // return t;
  // }
}
