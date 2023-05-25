import { Tensor } from "../src/tensor";

describe("the shape of a tensor", () => {
  it("should parse a 1d array", () => {
    expect(new Tensor([1, 2, 3]).shape).toEqual(new Int32Array([3]));
  });
  it("should parse a 2x3 matrix", () => {
    expect(
      new Tensor([
        [1, 1, 1],
        [1, 1, 1],
      ]).shape
    ).toEqual(new Int32Array([2, 3]));
  });
  it("should parse a 2x3x4 tensor", () => {
    expect(
      new Tensor([
        [
          [1, 1, 1, 1],
          [1, 1, 1, 1],
          [1, 1, 1, 1],
        ],
        [
          [1, 1, 1, 1],
          [1, 1, 1, 1],
          [1, 1, 1, 1],
        ],
      ]).shape
    ).toEqual(new Int32Array([2, 3, 4]));
  });
  it("should throw an error on an invalid shape", () => {
    expect(
      () =>
        new Tensor([
          [
            [1, 2],
            [3, 4],
          ],
          [[5, 6]],
        ])
    ).toThrow();
  });
});
