import { Linear, Tensor } from "../bin";

// @ts-ignore
// import MNIST from "../mnist/mnist_train.csv?raw";

const BATCH_SIZE = 64;

const setupData = async () => {
  const start = window.performance.now();
  console.log("Loading MNIST training data...");

  // let x = new Tensor();
  // let y = new Tensor();
  //@ts-ignore
  const { default: data } = (await import("../mnist/mnist_train.csv?raw")) as {
    default: string;
  };

  // Load the lines and shuffle them
  const lines = data
    .split("\n")
    .slice(1, -1)
    .sort(() => Math.random() - 0.5);

  const numBatches = Math.ceil(lines.length / BATCH_SIZE);

  // Arrays of size numBatches
  // Each batch is a matrix in x and a vector in y
  const x: Tensor[] = [];
  const y: Tensor[] = [];

  for (let b = 0; b < numBatches; b++) {
    const batchLines = lines.slice(b * BATCH_SIZE, (b + 1) * BATCH_SIZE);

    const batchX = new Tensor();
    const batchY = new Tensor();

    batchX.shape = new Int32Array([batchLines.length, 784]);
    batchX.data = new Float32Array(batchLines.length * 784);

    batchY.shape = new Int32Array([batchLines.length]);
    batchY.data = new Float32Array(batchLines.length);

    for (const [index, line] of batchLines.entries()) {
      const words = line.split(",");

      batchY.data[index] = parseInt(words[0]);

      for (let i = 0; i < 784; i++) {
        batchX.set([index, i], parseFloat(words[i + 1]) / 255);
      }
    }
    x.push(batchX);
    y.push(batchY.onehot(10));
  }

  console.log(
    `Loaded MNIST training data in ${(window.performance.now() - start).toFixed(
      2
    )}ms`
  );

  // console.log(x, y);

  return [x, y];
  // console.log(MNIST);
};

export const trainNN = async () => {
  console.log("Training neural network!");
  const [x, y] = await setupData();

  let fc1W = Tensor.rand([784, 128], true);
  console.log(fc1W);
  // let fc1B = Tensor.randn([1, 128]);

  let fc2W = Tensor.rand([128, 10], true);
  // let fc2B = Tensor.randn([1, 128]);

  // const params = [fc1W, fc2W];

  for (let epoch = 0; epoch < 1; epoch++) {
    for (let batch = 0; batch < x.length; batch++) {
      // console.log(fc1W);
      // console.log(fc1B);
      // Do the forward pass

      // NO BIAS TERM :((
      // Broadcasting addition would break my autodiff system lol
      const h1 = await Tensor.matmul(x[batch], fc1W);

      const h1Relu = h1.relu();

      const h2 = await Tensor.matmul(h1Relu, fc2W);

      // console.log("Y, H2, YPred");
      // console.log(y[batch]);
      // console.log(h1);
      // console.log(h1Relu);
      // console.log(h2);
      // console.log(h2.softmax());

      // Calculate the CE loss
      const loss = Tensor.crossEntropy(h2, y[batch]);

      console.log(`Epoch ${epoch} Batch ${batch} Loss: ${loss.mean()}`);
      // console.log(loss.mean());

      // Do the backward pass
      await loss.backward();

      // console.log(fc1W);
      // console.log(fc2W);

      // Momentarily disable gradients
      // fc1W.requiresGrad = false;
      // fc2W.requiresGrad = false;

      // Update the parameters
      // fc1W = await Tensor.plus(fc1W, fc1W.grad.scale(-0.01), true);
      // fc2W = await Tensor.plus(fc2W, fc2W.grad.scale(-0.01), true);

      fc1W.requiresGrad = true;
      fc1W.isLeaf = true;

      fc2W.requiresGrad = true;
      fc2W.isLeaf = true;
    }
  }
};
