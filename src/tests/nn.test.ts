import { Tensor, Linear } from "..";

import { assertArrayEquals, assertTrue, test } from "./testing";

export const testNns = async () => {
  await Promise.all([
    test("onehot conversion", () => {
      const a = new Tensor([1, 2, 3]);

      const b = a.onehot(4);

      const expected = new Tensor([
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
      ]);

      //   console.log(b);
      //   console.log(expected);

      assertTrue(Tensor.almostEq(b, expected));
    }),
    test("CE loss with softmax backprop (simple)", async () => {
      // Ground truth (one-hot)
      const y = new Tensor([1, 2, 3]);
      const yOnehot = y.onehot(4);

      //   console.log(yOnehot);

      // Pre-softmaxxed!
      const logits = new Tensor(
        [
          [1, 5, 1, 1],
          [2, 1, 5, 1],
          [0, 0, 0, 10],
        ],
        true
      );

      const loss = Tensor.crossEntropy(logits, yOnehot);

      console.log(loss);
      //   loss.backward();

      //   const expected = await Tensor.plus(logits.softmax(), yOnehot.scale(-1));

      //   console.log(logits);
      //   console.log(logits.grad);
      //   console.log(expected);
      //   assertTrue(Tensor.almostEq(logits.grad!, expected));

      //   console.log(yPred.softmax());
      //   console.log(loss);
      //   console.log(loss.sum());
      //   console.log(loss.mean());

      // assertArrayEquals(new Tensor([1, 2, 3]).shape, [3]);
    }),
  ]);
};
