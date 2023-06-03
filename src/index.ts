// oh typescript
declare global {
  //@ts-ignore
  type VideoFrame = any;
}
export { Tensor } from "./tensor";
export { arrEq, mean, stddev } from "./util";
