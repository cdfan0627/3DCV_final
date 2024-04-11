import torch
import torch.nn as nn
import torch.nn.functional as F

class SpecNetwork(nn.Module):
    def __init__(self, input_size=14, hidden_size=64, output_size=3):
        super(SpecNetwork, self).__init__()
        self.hidden1 = nn.Linear(input_size, hidden_size)
        self.hidden2 = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        print("Hellow")

    def forward(self, vector, view, rough):
        spec_color = torch.cat([vector, view, rough], dim=-1)
        spec_color1 = self.relu(self.hidden1(spec_color))
        spec_color2 = self.relu(self.hidden2(spec_color1))
        spec_color3 = self.sigmoid(self.output(spec_color2))
        return spec_color3

