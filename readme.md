<h1 align="center">
TenTS
</h1>
<p align="center">
TenTS is a WebGPU-accelerated Tensor library for the browser written in TypeScript with a PyTorch-like API. <b>Currently, matrix operations run between 10x and 100x slower than they would natively with PyTorch/CUDA. You should not use this (yet)!</b>
</p>

### Usage

```sh
pnpm i tents
```

Example:

```typescript
import { Tensor } from "tents";

const a = Tensor.rand([1000, 1000]);
const b = Tensor.rand([1000, 1000]);

const cpu = await Tensor.matmul(a, b);
const gpu = await Tensor.matmul(a.gpu(), b.gpu());

if (Tensor.almostEq(cpu, gpu)) console.log("🎉");
```

### Local Setup

To start a dev environment, first install the project locally:

```sh
# Clone and enter the repository
git clone https://github.com/jhsul/tents && cd tents

# Install dependencies
pnpm i

# Build the project (should create bin/)
pnpm build
```

For testing in the browser, a Vite webapp is available in the `env` folder:

```sh
# Enter the Vite application
cd env

# Install dependencies (react LMAO)
pnpm i

# Start the dev server
pnpm run dev
```

⚠️ **Note**: The Vite environment bundles directly from the `bin/` folder, so make sure to run `pnpm build` in the root directory before starting the dev server.
