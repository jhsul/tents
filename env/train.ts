import { Linear, Tensor } from "../bin";
class MLP {
  fc1: Linear;
  fc2: Linear;
  constructor() {
    this.fc1 = new Linear(784, 128);
    this.fc2 = new Linear(128, 10);
  }

  async forward(x: Tensor) {
    const h1 = await this.fc1.forward(x);
    const h2 = h1.relu();
    const h3 = await this.fc2.forward(h2);

    return h3;
  }
}

export const trainNN = async () => {
  console.log("Training neural network!");
  const model = new MLP();

  for (let epoch = 0; epoch < 10; epoch++) {}
};
