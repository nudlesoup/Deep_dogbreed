import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DogBreedPretrainedResnet18(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = models.resnet18(pretrained=True)
        # Replace last layer
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Sequential(
            nn.Linear(num_ftrs, 120),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, xb):
        return self.network(xb)


class DogBreedPretrainedVGG19(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = models.vgg19(pretrained=True)
        # Replace last layer
        self.network.classifier = nn.Sequential(
            nn.Linear(in_features=25088, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=120, bias=True),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, xb):
        return self.network(xb)