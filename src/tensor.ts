import { arrEq, findShape, gaussianSample, NestedArray } from "./util";

export class Tensor {
  static _device: GPUDevice;
  shape: Int32Array;
  data: Float32Array;

  // GPU
  isGpu: boolean;
  dataBuffer?: GPUBuffer;

  // Gradient
  requiresGrad: boolean;
  isLeaf: boolean = false;
  grad?: Tensor;

  // For tensors built by differentiable operations
  // It takes the propogated gradient, as well as the actual inputs
  gradFn?: (grad: Tensor, inputs: Tensor[]) => Promise<Tensor[]>;
  inputs?: Tensor[];

  // For non-leaf tensors in case we want to retain the gradient
  // Otherwise, only leaf gradients are stored for memory efficiency
  // This is the default behavior in pytorch, which uses retain_grad() here
  shouldRetainGrad: boolean = false;

  retainGrad() {
    this.shouldRetainGrad = true;
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

      this.data = new Float32Array(shape.reduce((a, b) => a * b, 1));

      //@ts-expect-error
      this.data.set(arr.flat(shape.length));
      // this.data = new Float32Array(arr.flat(shape.length));
    }

    // Initialize gradient if necessary
    if (requiresGrad) {
      this.isLeaf = true;
    }
  }

  static zeros(
    shape: number | number[] | Int32Array,
    requiresGrad: boolean = false
  ): Tensor {
    if (typeof shape === "number") {
      shape = [shape];
    }

    const t = new Tensor();

    t.shape = new Int32Array(shape);
    // TypedArrays are initialized to 0 by default

    //@ts-expect-errors
    t.data = new Float32Array(shape.reduce((a, b) => a * b, 1));
    if (requiresGrad) {
      t.requiresGrad = true;
      t.isLeaf = true;
    }
    return t;
  }

  static ones(
    shape: number | number[] | Int32Array,
    requiresGrad: boolean = false
  ): Tensor {
    if (typeof shape === "number") {
      shape = [shape];
    }

    const t = new Tensor();

    t.shape = new Int32Array(shape);

    //@ts-expect-error
    t.data = new Float32Array(shape.reduce((a, b) => a * b, 1)).fill(1.0);
    if (requiresGrad) {
      t.requiresGrad = true;
      t.isLeaf = true;
    }
    return t;
  }

  static rand(
    shape: number | number[] | Int32Array,
    requiresGrad = false
  ): Tensor {
    if (typeof shape === "number") {
      shape = [shape];
    }
    const t = new Tensor();

    t.shape = new Int32Array(shape);
    //@ts-expect-error
    t.data = new Float32Array(shape.reduce((a, b) => a * b, 1)).map(() =>
      Math.random()
    );

    if (requiresGrad) {
      t.requiresGrad = true;
      t.isLeaf = true;
    }

    return t;
  }

  static randn(
    shape: number | number[] | Int32Array,
    mean: number = 0,
    stddev: number = 1
  ) {
    if (typeof shape === "number") {
      shape = [shape];
    }

    const t = new Tensor();

    t.shape = new Int32Array(shape);

    //@ts-expect-error
    t.data = new Float32Array(shape.reduce((a, b) => a * b, 1)).map(() =>
      gaussianSample(mean, stddev)
    );
    return t;
  }

  // Identity matrix
  static eye(size: number): Tensor {
    const t = new Tensor();

    t.shape = new Int32Array([size, size]);

    t.data = new Float32Array(size * size);

    for (let i = 0; i < size; i++) {
      t.data[i * size + i] = 1;
    }

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

  // Random explode functions

  _cpuCheck() {
    if (this.isGpu)
      throw new Error("Attempting to perform CPU operation on GPU tensor");
  }

  static _checkShapes(a: Tensor, b: Tensor) {
    if (!arrEq(a.shape, b.shape)) throw new Error(`Shape mismatch! ${a} ${b}`);
  }

  static _checkDevices(a: Tensor, b: Tensor) {
    if (a.isGpu !== b.isGpu) throw new Error("Device mismatch");
  }

  // CPU-Only Operations
  // These are operations which are performed on the CPU
  // The ymust not be called if the tensor is mapped to the GPU

  neg(): Tensor {
    return this.scale(-1);
  }

  scale(s: number): Tensor {
    this._cpuCheck();

    const newShape = new Int32Array(this.shape);
    const newData = new Float32Array(this.data.length);

    for (let i = 0; i < this.data.length; i++) {
      newData[i] = this.data[i] * s;
    }

    const t = new Tensor();

    t.shape = newShape;
    t.data = newData;

    if (this.requiresGrad) {
      // t.isLeaf = false;
      t.requiresGrad = true;
      t.inputs = [this];

      t.gradFn = async (grad, inputs) => {
        return [grad.scale(s)];
      };
    }

    return t;
  }

  static elmult(a: Tensor, b: Tensor): Tensor {
    a._cpuCheck();
    b._cpuCheck();

    Tensor._checkShapes(a, b);

    const newShape = new Int32Array(a.shape);
    const newData = new Float32Array(a.data.length);

    for (let i = 0; i < a.data.length; i++) {
      newData[i] = a.data[i] * b.data[i];
    }

    const t = new Tensor();
    t.shape = newShape;
    t.data = newData;

    return t;
  }

  pow(s: number): Tensor {
    this._cpuCheck();

    const newShape = new Int32Array(this.shape);
    const newData = new Float32Array(this.data.length);

    for (let i = 0; i < this.data.length; i++) {
      newData[i] = Math.pow(this.data[i], s);
    }

    const t = new Tensor();

    t.shape = newShape;
    t.data = newData;

    if (this.requiresGrad) {
      // t.isLeaf = false;
      t.requiresGrad = true;
      t.inputs = [this];

      // The inputs here should only contain one tensor
      t.gradFn = async (grad, inputs) => {
        return [Tensor.elmult(grad, inputs[0].pow(s - 1).scale(s))];
      };
    }
    return t;
  }

  exp(): Tensor {
    this._cpuCheck();

    const newShape = new Int32Array(this.shape);
    const newData = new Float32Array(this.data.length);

    for (let i = 0; i < this.data.length; i++) {
      newData[i] = Math.exp(this.data[i]);
    }

    const t = new Tensor();

    t.shape = newShape;
    t.data = newData;

    if (this.requiresGrad) {
      t.requiresGrad = true;
      t.inputs = [this];

      // The inputs here should only contain one tensor
      t.gradFn = async (grad, inputs) => {
        return [Tensor.elmult(grad, inputs[0].exp())];
      };
    }
    return t;
  }

  relu(): Tensor {
    this._cpuCheck();

    const newShape = new Int32Array(this.shape);
    const newData = new Float32Array(this.data.length);

    for (let i = 0; i < this.data.length; i++) {
      newData[i] = this.data[i] > 0 ? this.data[i] : 0;
    }

    const t = new Tensor();

    t.shape = newShape;
    t.data = newData;

    if (this.requiresGrad) {
      t.requiresGrad = true;
      t.inputs = [this];
      t.gradFn = async (grad, inputs) => {
        const newGrad = new Tensor();
        newGrad.shape = new Int32Array(grad.shape);
        newGrad.data = grad.data.map((v, i) => (inputs[0].data[i] > 0 ? v : 0));
        return [newGrad];
      };
    }

    return t;
  }

  T(): Tensor {
    this._cpuCheck();

    const newShape = new Int32Array(this.shape).reverse();
    const newData = new Float32Array(this.data.length);

    // 2D matrix transpose
    if (this.shape.length === 2) {
      for (let r = 0; r < this.shape[0]; r++) {
        for (let c = 0; c < this.shape[1]; c++) {
          newData[c * this.shape[0] + r] = this.data[r * this.shape[1] + c];
        }
      }

      const t = new Tensor();

      t.shape = newShape;
      t.data = newData;

      return t;

      // console.log("DURING");
      // console.log(this.data);
      // console.log(this.shape);
      // console.log(this);
    } else if (this.shape.length === 3) {
      // batch transpose
      return this;
    } else
      throw new Error(
        "Only 2D matrices and 3D batch tensors may be transposed!"
      );
    // console.log(newData);
  }

  /**
   * Expects a 2D matrix of shape [m, n]
   * Calculates softmax across dimension 1
   */
  softmax(): Tensor {
    this._cpuCheck();

    if (this.shape.length != 2) {
      throw new Error("Softmax only supported for 2D matrices");
    }

    const [m, n] = this.shape;

    const newShape = new Int32Array(this.shape);
    const newData = new Float32Array(this.data.length);

    // 2D matrix softmax
    for (let r = 0; r < m; r++) {
      // Calculate the sum for this row
      let sum = 0;
      for (let c = 0; c < n; c++) {
        sum += Math.exp(this.data[r * n + c]);
      }
      for (let c = 0; c < n; c++) {
        newData[r * n + c] = Math.exp(this.data[r * n + c]) / sum;
      }
    }

    const t = new Tensor();
    t.shape = newShape;
    t.data = newData;

    return t;
  }

  /**
   * Calculates the cross entropy loss for multi-class classification
   * Expects both tensors to be 2D matrices of shape [m, n]
   * m is the number of samples
   * n is the number of classes
   * y should be a one-hot encoded matrix
   * logits are PRE-SOFTMAX values
   *
   * The gradient uses the mean
   */
  static crossEntropy(logits: Tensor, y: Tensor): Tensor {
    Tensor._checkShapes(y, logits);
    const eps = 1e-10;

    if (y.shape.length !== 2)
      throw new Error("Cross entropy only supported for 2D matrices");

    const [m, n] = y.shape;

    // The softmaxxed outputs
    // We do this here because we can take advantage of math trickery
    // Jacobian of softmax is hard
    // Jacobian of CE loss of softmax is ez
    const s = logits.softmax();

    const newShape = new Int32Array([m, 1]);
    const newData = new Float32Array(m);

    const t = new Tensor();
    t.shape = newShape;
    t.data = newData;

    for (let r = 0; r < m; r++) {
      let sum = 0;
      for (let c = 0; c < n; c++) {
        sum += y.get([r, c]) * Math.log(s.get([r, c]) || eps);
      }
      t.set([r, 0], -sum);
    }

    // y should NEVER require gradient
    // It's the ground truth!
    if (logits.requiresGrad) {
      t.requiresGrad = true;
      t.inputs = [logits];
      t.gradFn = async (grad, inputs) => {
        const nextGrad = await Tensor.plus(s, y.scale(-1));

        // This should be scaled by 1/m because we want
        // the mean over all samples
        // If we ignore this, this would be equivalent to sum reduction
        // But pytorch uses mean reduction by default
        return [nextGrad.scale(1 / m)];
      };
    }

    return t;
  }

  // Random utility
  sum(): number {
    return this.data.reduce((a, b) => a + b, 0);
  }
  mean(): number {
    return this.sum() / this.data.length;
  }

  onehot(numClasses: number): Tensor {
    if (this.shape.length !== 1) {
      throw new Error("Only 1D tensors can be one-hot encoded");
    }

    const t = Tensor.zeros([this.shape[0], numClasses]);

    for (let i = 0; i < this.data.length; i++) {
      const cls = Math.round(this.data[i]);
      t.set([i, cls], 1);
    }

    return t;
  }

  // Boolean Operations

  static eq(a: Tensor, b: Tensor): boolean {
    return arrEq(a.shape, b.shape) && arrEq(a.data, b.data);
  }

  static almostEq(a: Tensor, b: Tensor, eps: number = 1e-3): boolean {
    return (
      arrEq(a.shape, b.shape) &&
      a.data.every((v, i) => Math.abs(v - b.data[i]) < eps)
    );
  }

  // Top-Level Operations

  static async plus(a: Tensor, b: Tensor, noGrad = false): Promise<Tensor> {
    this._checkShapes(a, b);
    this._checkDevices(a, b);

    let t: Tensor;

    if (a.isGpu) {
      t = await Tensor._gpuPlus(a, b);
    } else {
      t = Tensor._cpuPlus(a, b);
    }

    if ((a.requiresGrad || b.requiresGrad) && !noGrad) {
      t.requiresGrad = true;
      // t.isLeaf = false;
      t.inputs = [a, b];

      t.gradFn = async (grad, inputs) => {
        return [grad, grad];
      };
    }
    return t;
  }

  static async matmul(a: Tensor, b: Tensor, noGrad = false): Promise<Tensor> {
    this._checkDevices(a, b);

    let t: Tensor;

    // Simple 2x2 matrix multiplication
    if (a.shape.length === 2 && b.shape.length === 2) {
      if (a.shape[1] !== b.shape[0])
        throw new Error(`Invalid shapes for 2x2 matmul! ${a} ${b}`);

      if (a.isGpu) {
        t = await Tensor._gpuMatmul(a, b);
      } else {
        t = Tensor._cpuMatmul(a, b);
      }

      // ONLY DO GRADIENTS FOR 2D MATRICES
      if ((a.requiresGrad || b.requiresGrad) && !noGrad) {
        t.requiresGrad = true;
        t.inputs = [a, b];
        t.gradFn = async (grad, inputs) => {
          return [
            await Tensor.matmul(grad, inputs[1].T(), true),
            await Tensor.matmul(inputs[0].T(), grad, true),
          ];
        };
      }
    } else if (a.shape.length === 3 && b.shape.length === 3) {
      if (
        // If the inner matrices may be mismatched
        a.shape[2] !== b.shape[1] ||
        // If outer indices differ, then one of them must be 1 (for broadcasting)
        (a.shape[0] !== b.shape[0] && a.shape[0] !== 1 && b.shape[0] !== 1)
      )
        throw new Error(`Invalid shapes for batch matmul! ${a} ${b}`);

      if (a.isGpu) {
        t = await Tensor._gpuBatchMatmul(a, b);
      } else t = Tensor._cpuBatchMatmul(a, b);
    } else throw new Error(`Invalid shapes for matmul! ${a} ${b}`);

    return t;
  }

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
   * Expects two tensors a, b with shapes:
   * - [s, n, m], [s, m, p]
   * - [1, n, m], [s, m, p]
   * - [s, n, m], [1, m, p]
   *
   * Performs a matrix multiplication on s pairs of matrices a[i,:,:], b[i,:,:]
   * Broadcasts as necessary
   *
   * This method assumes the shapes are correct
   * The checking should be done in the main matmul method
   */
  static _cpuBatchMatmul(a: Tensor, b: Tensor): Tensor {
    let broadA = false;
    let broadB = false;

    let s = a.shape[0];

    if (a.shape[0] === 1) {
      broadA = true;
      s = b.shape[0];
    }
    if (b.shape[0] === 1) {
      broadB = true;
      s = a.shape[0];
    }

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
            sum +=
              a.get([broadA ? 0 : r, i, k]) * b.get([broadB ? 0 : r, k, j]);
          }
          c.set([r, i, j], sum);
        }
      }
    }

    return c;
  }

  /**
   * Expects two tensors a, b with shapes:
   * - [s, n, m], [s, m, p]
   * - [1, n, m], [s, m, p]
   * - [s, n, m], [1, m, p]
   *
   * Performs a matrix multiplication on s pairs of matrices a[i,:,:], b[i,:,:]
   * Broadcasts as necessary
   */
  static async _gpuBatchMatmul(a: Tensor, b: Tensor): Promise<Tensor> {
    const broadA = a.shape[0] === 1;
    const broadB = b.shape[0] === 1;
    const s = broadA ? b.shape[0] : a.shape[0];

    const n = a.shape[1];
    const m = a.shape[2];
    const p = b.shape[2];

    const result = new Tensor();
    result.shape = new Int32Array([s, n, p]);
    result.data = new Float32Array(s * n * p);

    const workgroupSize = [8, 8, 4];

    const shaderCode = `
    struct TensorA {
      data: array<f32, ${broadA ? n * m : s * n * m}>,
    }

    struct TensorB {
      data: array<f32, ${broadB ? m * p : s * m * p}>,
    }

    struct TensorOut {
      data: array<f32, ${s * n * p}>,
    }

    @group(0) @binding(0) var<storage, read> A: TensorA;
    @group(0) @binding(1) var<storage, read> B: TensorB;
    @group(0) @binding(2) var<storage, read_write> C: TensorOut;

    @compute @workgroup_size(${workgroupSize[0]}, ${workgroupSize[1]}, ${
      workgroupSize[2]
    })
      fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let s = id.z;
    let i = id.x;
    let j = id.y;

    if (s >= ${s} || i >= ${n} || j >= ${p}) {
      return;
    }

    var sum: f32 = 0.0;
    for (var k = 0u; k < ${m}; k = k + 1u) {
      sum = sum + A.data[
        ${
          broadA
            ? "(i * " + m + " + k)"
            : "(s * " + n + " * " + m + " + i * " + m + " + k)"
        }] 
        * B.data[
        ${
          broadB
            ? "(k * " + p + " + j)"
            : "(s * " + m + " * " + p + " + k * " + p + " + j)"
        }];
    }

    C.data[s * ${n} * ${p} + i * ${p} + j] = sum;
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
      Math.ceil(p / workgroupSize[1]),
      Math.ceil(s / workgroupSize[2])
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

  // Automatic Differentiation

  async backward(grad?: Tensor) {
    if (!this.requiresGrad)
      throw new Error("Cannot call backward() on non-differentiable tensor!");

    // if (!grad) grad = Tensor.ones(this.shape);

    if (!grad) {
      grad = Tensor.ones(this.shape);
      // if (this.shape.length === 1) {
      //   grad = Tensor.ones(this.shape);
      // }
      // else if(this.shape.length === 2) {
      //   grad =
      // }
    }

    // If the current tensor was the result of an operation
    // then we need to backpropogate through that
    if (this.gradFn) {
      const nextGrads = await this.gradFn(grad, this.inputs!);

      // console.log("Nextgrads:");
      // console.log(nextGrads);
      // Does this count as parallelism?
      await Promise.all(
        this.inputs!.map(async (input, i) => {
          input.backward(nextGrads[i]);
        })
      );
    }

    if (this.isLeaf || this.shouldRetainGrad) {
      this.grad = grad;
    }
  }

  // No need for this since we are being hella memory inefficient
  // and just recomputing everything all the time anyway lol
  // zeroGrad() {
  //   if (!this.requiresGrad)
  //     throw new Error("Cannot call zeroGrad() on non-differentiable tensor!");

  //   for (let i = 0; i < this.grad!.data.length; i++) {
  //     this.grad!.data[i] = 0;
  //   }

  //   return this;
  // }
}
