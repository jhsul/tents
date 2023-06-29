import { Tensor } from "./tensor";

// oh typescript
declare global {
  //@ts-ignore
  type VideoFrame = any;
}

// This should be in here
// But it breaks and I can't use webgpu anyway
// Oh well :(
// await Tensor.setupDevice();

export { Tensor } from "./tensor";
export { Module, Linear } from "./nn";
export { arrEq, mean, stddev } from "./util";
