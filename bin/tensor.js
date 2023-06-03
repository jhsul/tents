import { arrEq, findShape, gaussianSample } from "./util";
export class Tensor {
    shape;
    data;
    isGpu;
    requiresGrad;
    dataBuffer;
    grad;
    static _device;
    // Constructors
    constructor(arr, requiresGrad = false) {
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
            if (!shape)
                throw new Error("Invalid shape");
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
        if (Tensor._device)
            return;
        const adapter = await navigator.gpu?.requestAdapter();
        const device = await adapter?.requestDevice();
        if (!device)
            throw new Error("WebGPU unavailable!");
        Tensor._device = device;
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
    static rand(shape) {
        if (typeof shape === "number") {
            shape = [shape];
        }
        else if (!Array.isArray(shape)) {
            throw new Error("Invalid shape");
        }
        const t = new Tensor();
        t.shape = new Int32Array(shape);
        t.data = new Float32Array(shape.reduce((a, b) => a * b, 1)).map(() => Math.random());
        return t;
    }
    static randn(shape, mean = 0, stddev = 1) {
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
    gpu() {
        if (!Tensor._device)
            throw new Error("Can't map tensor to GPU without device");
        if (this.isGpu)
            throw new Error("Attempting to map tensor to GPU twice");
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
    // CPU Operations
    static _cpu_forloop_plus(a, b) {
        if (!arrEq(a.shape, b.shape))
            throw new Error("Shape mismatch");
        const t = new Tensor();
        t.shape = new Int32Array(a.shape);
        t.data = new Float32Array(a.data.length);
        for (let i = 0; i < a.data.length; i++) {
            t.data[i] = a.data[i] + b.data[i];
        }
        return t;
    }
    static async _gpu_plus(a, b) {
        if (!arrEq(a.shape, b.shape))
            throw new Error("Shape mismatch");
        if (a.isGpu !== b.isGpu)
            throw new Error("Tensor device mismatch");
        if (!a.isGpu)
            return Tensor._cpu_forloop_plus(a, b);
        const shaderCode = `

    struct Tensor {
      data: array<f32, ${a.data.length}>,
    };
    

    @group(0) @binding(0) var<storage, read> a: Tensor;
    @group(0) @binding(1) var<storage, read> b: Tensor;

    @group(0) @binding(2) var<storage, read_write> result: Tensor;

        
    @compute @workgroup_size(64)
    fn main(@builtin(global_invocation_id) id: vec3<u32>) {
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
                { binding: 0, resource: { buffer: a.dataBuffer } },
                { binding: 1, resource: { buffer: b.dataBuffer } },
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
        passEncoder.dispatchWorkgroups(Math.ceil(a.data.length / 64));
        // End the compute pass
        passEncoder.end();
        // Copy the result buffer to the readback buffer
        commandEncoder.copyBufferToBuffer(resultBuffer, 0, readbackBuffer, 0, a.data.byteLength);
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
}
