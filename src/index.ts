import { Tensor } from "./tensor";

// oh typescript
declare global {
  //@ts-ignore
  type VideoFrame = any;
}

await Tensor.setupDevice();

export { Tensor } from "./tensor";
export { arrEq, mean, stddev } from "./util";
