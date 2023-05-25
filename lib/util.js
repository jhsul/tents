export const arrEq = (a, b) => a.length === b.length && a.every((v, i) => v === b[i]);
/**
 * Recursively find the shape of a NestedArray
 */
export const findShape = (arr) => {
    // Base case: we have found actual numbers!
    if (arr.every((x) => typeof x === "number")) {
        return [arr.length];
    }
    // We have found more
    else if (arr.every((x) => Array.isArray(x))) {
        //@ts-expect-error
        const shape = findShape(arr[0]);
        if (!shape)
            return null;
        //@ts-expect-error
        if (arr.every((a) => arrEq(findShape(a), shape))) {
            return [arr.length, ...shape];
        }
        else
            return null;
    }
    else {
        return null;
    }
};
export const gaussianSample = (mean = 0, stdev = 1) => {
    let u = 1 - Math.random(); // Converting [0,1) to (0,1]
    let v = Math.random();
    let z = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
    // Transform to the desired mean and standard deviation:
    return z * stdev + mean;
};
