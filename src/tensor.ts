import { arrEq, findShape, gaussianSample, NestedArray } from "./util";

export class Tensor {
  shape: Int32Array;
  data: Float32Array;

  // Constructors

  constructor(arr?: NestedArray) {
    /*
    Parse the shape from the array, or throw an exception
    */

    // An empty tensor
    if (!arr) {
      this.shape = new Int32Array();
      this.data = new Float32Array();
      return;
    }

    const shape = findShape(arr);
    if (!shape) throw new Error("Invalid shape");

    this.shape = new Int32Array(shape);

    //@ts-expect-error
    this.data = new Float32Array(arr.flat(shape.length));
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

  static normal(
    shape: number | number[],
    mean: number = 0,
    stddev: number = 1
  ) {
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

  cpu_plus(other: Tensor): Tensor {
    if (!arrEq(this.shape, other.shape)) throw new Error("Shape mismatch");

    const t = new Tensor();

    t.shape = new Int32Array(this.shape);
    t.data = new Float32Array(this.data.length);

    for (let i = 0; i < this.data.length; i++) {
      t.data[i] = this.data[i] + other.data[i];
    }

    return t;
  }
}
