<h1 align="center">
TenTS
</h1>
<p align="center">
TenTS is a WebGPU-accelerated Tensor library for the browser written in TypeScript with a PyTorch-like API. <b>Currently, matrix operations run between 10x and 100x slower than they would natively with PyTorch/CUDA. You should never use this!</b>
</p>

## Usage

```sh
pnpm i tents
```

Example:

```ts
import { Tensor } from "tents";

const a = Tensor.rand([1000, 1000]);
const b = Tensor.rand([1000, 1000]);

const cpu = await Tensor.matmul(a, b);
const gpu = await Tensor.matmul(a.gpu(), b.gpu());

if (Tensor.almostEq(cpu, gpu)) console.log("🎉");
```

## Documentation

### Introduction

TenTS introduces a `Tensor` class designed to mimic PyTorch's `tensor`. They can be constructed from nested arrays, or via constructors. Data is stored internally using a `Float32Array`. _Tensors are immutable_; although you can manually edit them if you choose to, all TenTS operations construct new tensor objects with new data buffers.

### Constructors

The default constructor accepts nested arrays. All constructors accept an optional `requiresGrad` argument which defaults to `false`.

```ts
// Standard, non-differentiable matrix
const x = new Tensor([
  [1, 2, 3],
  [4, 5, 6],
]);
```

This is useful for small tensors, but more useful constructors are also provided. Note:

```ts
// Fill constructors
const a = Tensor.zeros([2, 2]);
const b = Tensor.ones(4);

// Sampled from uniform random distribution
const c = Tensor.rand([2, 3]);

// Sampled from standard normal distribution
const d = Tensor.randn(10);

// Sampled from normal distribution with μ=5, σ=0.01
const e = Tensor.randn(10, 5, 0.01);
```

### Unary Operations

All unary operations are _synchronous_ since they are only implemented CPU-side.

```ts
const a = new Tensor([-1, 0, 1, 2]);

const b = a.neg(); // [1, 0, -1, -2]
const c = a.scale(2); // [-2, 0, 2, 4]
const d = a.pow(2); // [1, 0, 1, 4];
const e = a.exp(); // [1/e, 1, e, e^2]

const matrix = new Tensor([
  [1, 2],
  [3, 4],
]);

const matrixT = matrix.T(); // [[1, 3], [2, 4]]
```

### Binary Operations

TenTS includes exact and approximate equality operations. Both are _synchronous_.

```ts
const actualEq = Tensor.eq(a, b);
const usefulEq = Tensor.almostEq(a, b);

if (actualEq !== usefulEq) console.log("float moment");

// You can specify ε if you want (default = 1e-3)
const badEq = Tensor.almostEq(a, b, 1);
```

The bread and butter of TenTS are `plus()` and `matmul()`. These are both _asynchronous_ methods.

```ts
const a = new Tensor([1, 2, 3]);
const b = new tensor([4, 5, 6]);

const c = await Tensor.plus(a, b);
// [5, 7, 9]
```

The `matmul` function supports both standard 2D matrix multiplication, and 3D batch multiplication. In the latter case, broadcasting is supported.

```ts
// Basic 2D matrix multiplication

const a = new Tensor([
  [1, 2],
  [3, 4],
]);

const b = new Tensor([
  [5, 6],
  [7, 8],
]);

const c = await Tensor.matmul(a, b);
// [
//   [19, 22],
//   [43, 50],
// ]
```

### GPU Acceleration

Instead of PyTorch's general `.to(device)` syntax, TenTS uses a simpler `.gpu()` method. Despite GPU writes being asynchronous, this method is _synchronous_ and does not return a promise. The GPU device is only actually awaited during operations.

Using WebGPU Acceleration is as simple as calling `.gpu()` before an operation. Just make sure all or no operands are gpu mapped!

```ts
const a = Tensor.rand([100, 100]);
const b = Tensor.rand([100, 100]);

const c = await Tensor.matmul(a.gpu(), b.gpu());
```

WebGPU acceleration is available for only the `plus()` and `matmul()` functions.

### Automatic Differentiation

TenTS also includes an automatic differentiation system. Requiring gradients is as simple

## Local Dev Environment

To start a dev environment, first install the project locally:

```sh
# Clone and enter the repository
git clone https://github.com/jhsul/tents && cd tents

# Install dependencies
pnpm i

# Build the project (should create bin/)
pnpm build

# Enter the development environment
cd env

# Install separate dependencies (for the env)
pnpm i

# Start the vite dev server
pnpm run dev
```

⚠️ **Note**: The Vite environment bundles directly from the `bin/` folder, so make sure to run `pnpm build` in the root directory before starting the dev server.
