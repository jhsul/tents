import { arrEq } from "../src/util";

describe("the arrEq function", () => {
  it("should return true for equal arrays", () => {
    expect(arrEq([1, 2, 3], [1, 2, 3])).toBe(true);
  });

  it("should return false for unequal arrays", () => {
    expect(arrEq([1, 2, 3], [1, 2, 4])).toBe(false);
  });

  it("should work for different array types", () => {
    expect(arrEq(new Float32Array([1, 2, 3]), [1, 2, 3])).toBe(true);
  });
});
