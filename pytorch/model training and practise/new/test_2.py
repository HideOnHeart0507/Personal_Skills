import torch

outputs = torch.tensor([[0.1,0.2], [0.3,0.4]])

outputs.argmax(1)

input_target = torch.tensor([0,1])
print(preds==input_target)