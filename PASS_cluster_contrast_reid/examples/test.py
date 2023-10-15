import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.model1 = nn.Conv2d(1,3,3,1, bias=True)


    def forward(self, x):
        x = self.model1(x)
        return x

model = MyModel()
total_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {total_params}")