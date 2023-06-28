import torch as th
import torch.nn as nn


softmax_nn = nn.Softmax(dim=1)
loss_fn = nn.CrossEntropyLoss(reduction="mean")


# For a 4 class classification example
ground_truth = th.tensor([1, 2, 3])
ground_truth_onehot = th.tensor(
    [[0, 1, 0, 0],
     [0, 0, 1, 0],
     [0, 0, 0, 1]])

# Pre-softmaxxed outputs of some NN
logits = th.tensor([
    [1., 5, 1, 1],
    [2, 1, 5, 1],
    [0, 0, 0, 10]
], requires_grad=True)

with th.no_grad():
    y_pred = softmax_nn(logits)


loss = loss_fn(logits, ground_truth)
loss.backward()

print(loss)
print(logits.grad*3)
# print(y_pred.grad)
# print(softmaxxed_y_pred)
print(y_pred - ground_truth_onehot)
