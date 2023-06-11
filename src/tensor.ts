import { arrEq, findShape, gaussianSample, NestedArray } from "./util";

export class Tensor {
  shape: Int32Array;
  data: Float32Array;

  isGpu: boolean;
  requiresGrad: boolean;

  dataBuffer?: GPUBuffer;

  grad?: Tensor;

  static _device: GPUDevice;

  // Constructors

  constructor(arr?: NestedArray, requiresGrad: boolean = false) {
    /*
    Parse the shape from the array, or throw an exception
    */

    this.isGpu = false;
    this.requiresGrad = requiresGrad;

    // An empty tensor
    if (!arr) {
      this.shape = new Int32Array();
      this.data = new Float32Array();
    }
    // Tensor is given initial values
    else {
      const shape = findShape(arr);
      if (!shape) throw new Error("Invalid shape");

      this.shape = new Int32Array(shape);

      //@ts-expect-error
      this.data = new Float32Array(arr.flat(shape.length));
    }

    // Initialize gradient if necessary
    if (requiresGrad) {
      this.grad = Tensor.zeros(Array.from(this.shape));
    }
  }

  static async setupDevice() {
    if (Tensor._device) {
      console.log("Device already setup!");
      return;
    }

    const adapter = await navigator.gpu?.requestAdapter();
    const device = await adapter?.requestDevice();

    if (!device) throw new Error("WebGPU unavailable!");

    Tensor._device = device;
    console.log("Setup device!");
  }

  static zeros(shape: number | number[]): Tensor {
    if (typeof shape === "number") {
      shape = [shape];
    } else if (!Array.isArray(shape)) {
      throw new Error("Invalid shape");
    }

    const t = new Tensor();

    t.shape = new Int32Array(shape);
    // TypedArrays are initialized to 0 by default
    t.data = new Float32Array(shape.reduce((a, b) => a * b, 1));
    return t;
  }

  static ones(shape: number | number[]): Tensor {
    if (typeof shape === "number") {
      shape = [shape];
    } else if (!Array.isArray(shape)) {
      throw new Error("Invalid shape");
    }

    const t = new Tensor();

    t.shape = new Int32Array(shape);
    t.data = new Float32Array(shape.reduce((a, b) => a * b, 1)).fill(1.0);
    return t;
  }

  static rand(shape: number | number[]): Tensor {
    if (typeof shape === "number") {
      shape = [shape];
    } else if (!Array.isArray(shape)) {
      throw new Error("Invalid shape");
    }

    const t = new Tensor();

    t.shape = new Int32Array(shape);
    t.data = new Float32Array(shape.reduce((a, b) => a * b, 1)).map(() =>
      Math.random()
    );

    return t;
  }

  static randn(shape: number | number[], mean: number = 0, stddev: number = 1) {
    if (typeof shape === "number") {
      shape = [shape];
    } else if (!Array.isArray(shape)) {
      throw new Error("Invalid shape");
    }

    const t = new Tensor();

    t.shape = new Int32Array(shape);
    t.data = new Float32Array(shape.reduce((a, b) => a * b, 1)).map(() =>
      gaussianSample(mean, stddev)
    );
    return t;
  }

  get(idx: number | number[]) {
    if (typeof idx === "number") {
      if (this.shape.length > 1)
        throw new Error("Invalid index for multidimensional tensor");
      return this.data[idx];
    } else if (Array.isArray(idx)) {
      if (idx.length !== this.shape.length) throw new Error("Invalid index");

      let offset = 0;
      for (let i = 0; i < idx.length; i++) {
        let stride = 1;
        for (let j = i + 1; j < this.shape.length; j++) {
          stride *= this.shape[j];
        }
        offset += idx[i] * stride;
      }

      return this.data[offset];
    } else throw new Error("Invalid index");
  }

  set(idx: number | number[], val: number) {
    if (typeof idx === "number") {
      if (this.shape.length > 1)
        throw new Error("Invalid index for multidimensional tensor");

      const old = this.data[idx];
      this.data[idx] = val;
      return old;
    } else if (Array.isArray(idx)) {
      if (idx.length !== this.shape.length) throw new Error("Invalid index");

      let offset = 0;
      for (let i = 0; i < idx.length; i++) {
        let stride = 1;
        for (let j = i + 1; j < this.shape.length; j++) {
          stride *= this.shape[j];
        }
        offset += idx[i] * stride;
      }

      const old = this.data[offset];
      this.data[offset] = val;
      return old;
    } else throw new Error("Invalid index");
  }

  gpu() {
    if (!Tensor._device)
      throw new Error("Can't map tensor to GPU without device");

    if (this.isGpu) throw new Error("Attempting to map tensor to GPU twice");

    const dataBuffer = Tensor._device.createBuffer({
      size: this.data.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    // new Float32Array(dataBuffer.getMappedRange()).set(this.data);
    // dataBuffer.unmap();

    Tensor._device.queue.writeBuffer(dataBuffer, 0, this.data.buffer);

    this.isGpu = true;
    this.dataBuffer = dataBuffer;

    return this;
  }

  // Basic operations

  static eq(a: Tensor, b: Tensor): boolean {
    return arrEq(a.shape, b.shape) && arrEq(a.data, b.data);
  }

  static neg(a: Tensor): Tensor {
    if (a.isGpu)
      throw new Error(
        "Negation is a CPU operation. Run it before calling gpu()"
      );

    const t = new Tensor();
    t.shape = new Int32Array(a.shape);
    t.data = new Float32Array(a.data.length);

    for (let i = 0; i < a.data.length; i++) {
      t.data[i] = -a.data[i];
    }

    return t;
  }

  static async plus(a: Tensor, b: Tensor): Promise<Tensor> {
    if (!arrEq(a.shape, b.shape)) throw new Error("Shape mismatch");

    if (a.isGpu && b.isGpu) {
      return await Tensor._gpuPlus(a, b);
    }

    if (!a.isGpu && !b.isGpu) {
      return Tensor._cpuPlus(a, b);
    } else throw new Error("Tensor device mismatch");
  }

  // static async minus(a: Tensor, b: Tensor): Promise

  // CPU Operations

  static _cpuPlus(a: Tensor, b: Tensor): Tensor {
    const t = new Tensor();

    t.shape = new Int32Array(a.shape);
    t.data = new Float32Array(a.data.length);

    for (let i = 0; i < a.data.length; i++) {
      t.data[i] = a.data[i] + b.data[i];
    }

    return t;
  }

  static async _gpuPlus(a: Tensor, b: Tensor): Promise<Tensor> {
    // if (!arrEq(a.shape, b.shape)) throw new Error("Shape mismatch");

    // if (a.isGpu !== b.isGpu) throw new Error("Tensor device mismatch");

    // if (!a.isGpu) return Tensor._cpu_forloop_plus(a, b);

    const workgroupSize = 128;

    const shaderCode = `

    struct Tensor {
      data: array<f32, ${a.data.length}>,
    };


    @group(0) @binding(0) var<storage, read> a: Tensor;
    @group(0) @binding(1) var<storage, read> b: Tensor;

    @group(0) @binding(2) var<storage, read_write> result: Tensor;

        
    @compute @workgroup_size(${workgroupSize})
    fn main(@builtin(global_invocation_id) id: vec3<u32>) {
      if (id.x >= ${a.data.length}) {
          return;
      }
      let i = id.x;
      result.data[i] = a.data[i] + b.data[i];
    }
    `;

    const shaderModule = Tensor._device.createShaderModule({
      code: shaderCode,
    });

    const result = new Tensor();
    result.shape = new Int32Array(a.shape);
    // result.shape.set(a.shape);

    result.data = new Float32Array(a.data.length);

    const resultBuffer = Tensor._device.createBuffer({
      size: a.data.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    // The resultBuffer will be copied to this after calculations are done
    // Ideally we'd only use one buffer, like we do with writeBuffer
    // But, I can't figure out any way to do this, sice STORAGE | MAP_READ is invalid
    const readbackBuffer = Tensor._device.createBuffer({
      size: a.data.byteLength,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    // Create the bind group layout
    const bindGroupLayout = Tensor._device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "read-only-storage" },
        },
        {
          binding: 1,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "read-only-storage" },
        },
        {
          binding: 2,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "storage" },
        },
      ],
    });

    // Create the bind group
    const bindGroup = Tensor._device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: a.dataBuffer! } },
        { binding: 1, resource: { buffer: b.dataBuffer! } },
        { binding: 2, resource: { buffer: resultBuffer } },
      ],
    });

    // Create the pipeline
    const pipeline = Tensor._device.createComputePipeline({
      layout: Tensor._device.createPipelineLayout({
        bindGroupLayouts: [bindGroupLayout],
      }),
      compute: {
        module: shaderModule,
        entryPoint: "main",
      },
    });

    // Create the command encoder
    const commandEncoder = Tensor._device.createCommandEncoder();

    // Begin the compute pass
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);

    // Dispatch the compute job

    passEncoder.dispatchWorkgroups(Math.ceil(a.data.length / workgroupSize));

    // End the compute pass
    passEncoder.end();

    // Copy the result buffer to the readback buffer
    commandEncoder.copyBufferToBuffer(
      resultBuffer,
      0,
      readbackBuffer,
      0,
      a.data.byteLength
    );

    // Submit the command
    Tensor._device.queue.submit([commandEncoder.finish()]);

    await readbackBuffer.mapAsync(GPUMapMode.READ);
    const resultData = new Float32Array(readbackBuffer.getMappedRange());
    result.data.set(resultData);

    readbackBuffer.unmap();

    // Cleanup the buffers
    a.dataBuffer?.destroy();
    b.dataBuffer?.destroy();

    resultBuffer.destroy();
    readbackBuffer.destroy();

    // copy the output data to the result tensor's data
    // result.data.set(outputData);

    // Now just clean up our input tensors
    a.isGpu = false;
    b.isGpu = false;

    a.dataBuffer = undefined;
    b.dataBuffer = undefined;

    return result;
  }

  /**
   * Expects two matrices a, b with shapes [n, m], [m, p]
   * https://en.wikipedia.org/wiki/Matrix_multiplication_algorithm#Iterative_algorithm
   */
  static _cpuMatmul(a: Tensor, b: Tensor): Tensor {
    const c = new Tensor();

    const n = a.shape[0];
    const m = a.shape[1];
    const p = b.shape[1];

    c.shape = new Int32Array([n, p]);
    c.data = new Float32Array(n * p);

    for (let i = 0; i < n; i++) {
      for (let j = 0; j < p; j++) {
        let sum = 0;
        for (let k = 0; k < m; k++) {
          sum += a.data[i * m + k] * b.data[k * p + j];
        }
        c.data[i * p + j] = sum;
      }
    }
    return c;
  }

  /**
   * Expects two matrices a, b with shapes [n, m], [m, p]
   */
  static async _gpuMatmul(a: Tensor, b: Tensor): Promise<Tensor> {
    const n = a.shape[0];
    const m = a.shape[1];
    const p = b.shape[1];

    const result = new Tensor();
    result.shape = new Int32Array([n, p]);
    result.data = new Float32Array(n * p);

    const workgroupSize = [16, 16];

    const shaderCode = `
   

      struct TensorA {
        data: array<f32, ${n * m}>,
      }

      struct TensorB {
        data: array<f32, ${m * p}>,
      }

      struct TensorOut {
        data: array<f32, ${n * p}>,
      }

      @group(0) @binding(0) var<storage, read> A: TensorA;
      @group(0) @binding(1) var<storage, read> B: TensorB;

      @group(0) @binding(2) var<storage, read_write> C: TensorOut;

      @compute @workgroup_size(${workgroupSize[0]}, ${workgroupSize[1]})
      fn main(@builtin(global_invocation_id) id: vec3<u32>) {

        if (id.x >= ${n} || id.y >= ${p}) {
          return;
        }

        let i = id.x;
        let j = id.y;

        var sum: f32 = 0.0;
        for (var k = 0u; k < ${m}; k = k + 1u) {
          sum = sum + A.data[i * ${m} + k] * B.data[k * ${p} + j];
        }

        C.data[i * ${p} + j] = sum;
      }
    `;

    const shaderModule = Tensor._device.createShaderModule({
      code: shaderCode,
    });

    const resultBuffer = Tensor._device.createBuffer({
      size: result.data.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    const readbackBuffer = Tensor._device.createBuffer({
      size: result.data.byteLength,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    // Create the bind group layout
    const bindGroupLayout = Tensor._device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "read-only-storage" },
        },
        {
          binding: 1,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "read-only-storage" },
        },
        {
          binding: 2,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "storage" },
        },
      ],
    });

    const bindGroup = Tensor._device.createBindGroup({
      layout: bindGroupLayout,

      entries: [
        { binding: 0, resource: { buffer: a.dataBuffer! } },
        { binding: 1, resource: { buffer: b.dataBuffer! } },
        { binding: 2, resource: { buffer: resultBuffer } },
      ],
    });

    const pipeline = Tensor._device.createComputePipeline({
      layout: Tensor._device.createPipelineLayout({
        bindGroupLayouts: [bindGroupLayout],
      }),
      compute: {
        module: shaderModule,
        entryPoint: "main",
      },
    });

    const commandEncoder = Tensor._device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);

    // Dispatch the compute job
    passEncoder.dispatchWorkgroups(
      Math.ceil(n / workgroupSize[0]),
      Math.ceil(p / workgroupSize[1])
    );

    // End the compute pass
    passEncoder.end();

    // Copy the result buffer to the readback buffer
    commandEncoder.copyBufferToBuffer(
      resultBuffer,
      0,
      readbackBuffer,
      0,
      result.data.byteLength
    );

    // Submit the command
    Tensor._device.queue.submit([commandEncoder.finish()]);

    await readbackBuffer.mapAsync(GPUMapMode.READ);
    const resultData = new Float32Array(readbackBuffer.getMappedRange());
    result.data.set(resultData);

    readbackBuffer.unmap();

    // Cleanup the buffers
    a.dataBuffer?.destroy();
    b.dataBuffer?.destroy();

    resultBuffer.destroy();
    readbackBuffer.destroy();

    // copy the output data to the result tensor's data
    // result.data.set(outputData);

    // Now just clean up our input tensors
    a.isGpu = false;
    b.isGpu = false;

    a.dataBuffer = undefined;
    b.dataBuffer = undefined;

    return result;
  }

  /**
   * Expects two tensors a, b with shapes [s, n, m], [s, m, p]
   * Performs a matrix multiplication on s pairs of matrices a[i,:,:], b[i,:,:]
   */
  static _cpuBatchMatmul(a: Tensor, b: Tensor): Tensor {
    const s = a.shape[0];

    const c = new Tensor();

    const n = a.shape[1];
    const m = a.shape[2];
    const p = b.shape[2];

    c.shape = new Int32Array([s, n, p]);
    c.data = new Float32Array(s * n * p);

    for (let r = 0; r < s; r++) {
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < p; j++) {
          let sum = 0;
          for (let k = 0; k < m; k++) {
            sum += a.get([i, k]) * b.get([k, j]);
          }
          c.set([r, i, j], sum);
        }
      }
    }

    return c;
  }
}
