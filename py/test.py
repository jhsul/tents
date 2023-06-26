import torch as th


a = th.tensor([[1., 2.], [3, 4]], requires_grad=True)
b = th.tensor([[5., 6.], [7, 8]], requires_grad=True)


Q = th.matmul(a, b)

# Q.backward(th.eye(2))
Q.sum().backward()

print("Gradients:")
print(a.grad)
print(b.grad)

print("Q:")

print(Q)
