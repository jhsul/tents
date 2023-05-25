import { findShape, gaussianSample } from "./util";
export class Tensor {
    shape;
    data;
    constructor(arr) {
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
        if (!shape)
            throw new Error("Invalid shape");
        this.shape = new Int32Array(shape);
        //@ts-expect-error
        this.data = new Float32Array(arr.flat(shape.length));
    }
    static zeros(shape) {
        if (typeof shape === "number") {
            shape = [shape];
        }
        else if (!Array.isArray(shape)) {
            throw new Error("Invalid shape");
        }
        const t = new Tensor();
        t.shape = new Int32Array(shape);
        // TypedArrays are initialized to 0 by default
        t.data = new Float32Array(shape.reduce((a, b) => a * b, 1));
        return t;
    }
    static ones(shape) {
        if (typeof shape === "number") {
            shape = [shape];
        }
        else if (!Array.isArray(shape)) {
            throw new Error("Invalid shape");
        }
        const t = new Tensor();
        t.shape = new Int32Array(shape);
        t.data = new Float32Array(shape.reduce((a, b) => a * b, 1)).fill(1.0);
        return t;
    }
    static normal(shape, mean = 0, stddev = 1) {
        if (typeof shape === "number") {
            shape = [shape];
        }
        else if (!Array.isArray(shape)) {
            throw new Error("Invalid shape");
        }
        const t = new Tensor();
        t.shape = new Int32Array(shape);
        t.data = new Float32Array(shape.reduce((a, b) => a * b, 1)).map(() => gaussianSample(mean, stddev));
        return t;
    }
}
