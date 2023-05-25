export interface NestedArray extends Array<number | NestedArray> {}

type ArrayType = Float32Array | Int32Array | number[];

/**
 * Shallow equality check for two array types
 * Note: a number[] and a Float32Array can return true if the elements are the same
 */
export const arrEq = (a: ArrayType, b: ArrayType) =>
  a.length === b.length && a.every((v, i) => v === b[i]);

/**
 * Recursively find the shape of a NestedArray
 */
export const findShape = (arr: NestedArray): number[] | null => {
  // Base case: we have found actual numbers!
  if (arr.every((x) => typeof x === "number")) {
    return [arr.length];
  }
  // We have found more
  else if (arr.every((x) => Array.isArray(x))) {
    //@ts-expect-error
    const shape = findShape(arr[0]);

    if (!shape) return null;

    //@ts-expect-error
    if (arr.every((a) => arrEq(findShape(a), shape))) {
      return [arr.length, ...shape];
    } else return null;
  } else {
    return null;
  }
};

/**
 * Uses the Box Muller Transform to sample from a normal distribution
 * https://stackoverflow.com/a/36481059
 */
export const gaussianSample = (mean = 0, stdev = 1) => {
  let u = 1 - Math.random(); // Converting [0,1) to (0,1]
  let v = Math.random();
  let z = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);

  // Transform to the desired mean and standard deviation:
  return z * stdev + mean;
};

/**
 * Compute the mean value of an array one element at a time
 * Avoids a potential overflow error
 * Although, this is probably never really an issue with floats...
 * https://stackoverflow.com/a/72565782
 */
export const mean = (arr: ArrayType) =>
  //@ts-expect-error - reduce is a valid method on all ArrayTypes, even if typescript doesn't think so
  arr.reduce((m, x, i) => m + (x - m) / (i + 1), 0);
