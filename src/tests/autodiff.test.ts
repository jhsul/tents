import { Tensor } from "..";
import { assertArrayEquals, assertTrue, test } from "./testing";

export const testAutodiff = async () => {
  await Promise.all([
    test("scaling derivative", async () => {
      const a = new Tensor([1, 2, 3], true);
      const b = a.scale(2);

      await b.backward();

      const expected = new Tensor([2, 2, 2]);

      assertTrue(Tensor.almostEq(a.grad!, expected));
    }),

    test("scaling multiple times", async () => {
      const a = new Tensor([1, 2, 3], true);
      const a1 = a.scale(2);
      const a2 = a1.scale(3);
      a2.retainGrad();
      const Q = a2.neg();

      await Q.backward();

      assertTrue(Tensor.almostEq(a.grad!, new Tensor([-6, -6, -6])));
      assertTrue(!a1.grad);
      assertTrue(Tensor.almostEq(a2.grad!, new Tensor([-1, -1, -1])));
    }),

    test("pow derivative", async () => {
      const a = new Tensor([1, 2, 3], true);
      const b = a.pow(3);

      await b.backward();

      const expected = new Tensor([3, 12, 27]);

      // console.log(a);
      // console.log(b);

      assertTrue(Tensor.almostEq(a.grad!, expected));
    }),

    test("scaling and pow", async () => {
      const a = new Tensor([1, 2, 3], true);
      const Q = a.pow(3).scale(2);

      await Q.backward();

      const expected = new Tensor([6, 24, 54]);

      assertTrue(Tensor.almostEq(a.grad!, expected));
    }),
    test("pytorch docs autograd simple example", async () => {
      const a = new Tensor([2, 3], true);
      const b = new Tensor([6, 4], true);

      // console.log(a);
      // console.log(b);

      // console.log("---");

      const a1 = a.pow(3);
      const a2 = a1.scale(3);

      // console.log(a1);
      // console.log(a2);

      const b1 = b.pow(2);
      const b2 = b1.neg();

      // Q = 3*a**3 - b**2
      const Q = await Tensor.plus(a.pow(3).scale(3), b.pow(2).neg());
      // const Q = await Tensor.plus(a2, b2);

      await Q.backward();

      const dQda = a.pow(2).scale(9);
      const dQdb = b.scale(-2);

      assertTrue(Tensor.almostEq(dQda, a.grad!));
      assertTrue(Tensor.almostEq(dQdb, b.grad!));
    }),
  ]);
};
