import torch
from torch import nn

class BinaryClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, out_size):
        super(BinaryClassifier, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.layer4 = nn.Linear(hidden_size, out_size)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.dropout(x)
        x = torch.relu(self.layer2(x))
        x = self.dropout(x)
        x = torch.relu(self.layer3(x))
        x = self.dropout(x)
        x = self.layer4(x)  # Directly output the logits
        x = torch.sigmoid(x)  # Apply sigmoid to get probabilities

        # Apply step function to convert probabilities to binary (0 or 1)
        x = (x > 0.5).float()
        return x
