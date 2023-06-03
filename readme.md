<h1 align="center">
TenTS
</h1>
<p align="center">
TenTS is a WebGPU-accelerated Tensor library for the browser written in TypeScript with a PyTorch-like API. It is slow as fuck, and you should probably never use it!
</p>

### Usage

🛑 This is not set up correctly, yet. Right now this will just copy the repo into your project directory.

```sh
pnpm i tents
```

Example:

```typescript
import { Tensor } from "tents";

// Run this once to access the device
await Tensor.setupDevice();

const a = Tensor.rand([1000, 1000]);
const b = Tensor.rand([1000, 1000]);

const cpu = Tensor.matmul(a, b);

const gpu = await Tensor.matmul(a.gpu(), b.gpu());

if (Tensor.eq(cpu, gpu)) console.log("🎉");
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
