import torch
import torch.nn as nn


class DeepMultiClassifier(nn.Module):
    def __init__(self, input_size, hidden_sizes, out_size, dropout_prob=0.5):
        super(DeepMultiClassifier, self).__init__()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(p=dropout_prob)
        self.activation_function = nn.ReLU()

        # Input layer
        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))

        # Hidden layers
        for i in range(1, len(hidden_sizes)):
            self.layers.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))

        # Output layer
        self.output_layer = nn.Linear(hidden_sizes[-1], out_size)

    def forward(self, x):
        for layer in self.layers:
            x = self.activation_function(layer(x))
            x = self.dropout(x)
        x = self.output_layer(x)  # Output logits directly
        return x

    def set_activation_function(self, function):
        self.activation_function = function