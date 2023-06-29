import { Tensor } from "./tensor";

// oh typescript
declare global {
  //@ts-ignore
  type VideoFrame = any;
}

// Removing this again lol wtf esbuild
// just for tehe demo...
// await Tensor.setupDevice();

export { Tensor } from "./tensor";
export { Module, Linear } from "./nn";
export { arrEq, mean, stddev } from "./util";
