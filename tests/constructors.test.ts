import { Tensor } from "../src/tensor";

describe("the zeros constructor", () => {
  it("should create a 1d tensor", () => {
    expect(Tensor.zeros(3).data).toEqual(new Float32Array([0, 0, 0]));
  });
  it("should create a 2d tensor", () => {
    expect(Tensor.zeros([2, 3]).data).toEqual(
      new Float32Array([0, 0, 0, 0, 0, 0])
    );
  });
  it("should create a 3d tensor", () => {
    expect(Tensor.zeros([2, 3, 4]).data).toEqual(new Float32Array(24).fill(0));
  });
});

describe("the ones constructor", () => {
  it("should create a 1d tensor", () => {
    expect(Tensor.ones(3).data).toEqual(new Float32Array([1, 1, 1]).fill(1));
  });
  it("should create a 2d tensor", () => {
    expect(Tensor.ones([2, 3]).data).toEqual(
      new Float32Array([1, 1, 1, 1, 1, 1]).fill(1)
    );
  });
  it("should create a 3d tensor", () => {
    expect(Tensor.ones([2, 3, 4]).data).toEqual(new Float32Array(24).fill(1));
  });
});
