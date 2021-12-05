import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Complex(nn.Module):
    def __init__(self):
        super(Complex, self).__init__()
        #model_res = models.resnet18(pretrained=True)
        model_res = models.vgg19(pretrained=True)

        self.resnet = model_res
        for name, param in model_res.named_parameters():
            if ("bn" not in name):
                param.requires_grad = False

        # for param in _resnet.parameters():
        #     param.requires_grad = False
        self.rfc1 = nn.Linear(512, 512)
        #model_dense=models.densenet121(pretrained=True)
        model_dense=models.alexnet(pretrained=True)

        self.densenet = model_dense

        for name, param in model_dense.named_parameters():
            if ("bn" not in name):
                param.requires_grad = False


        self.dfc1 = nn.Linear(1024, 512)

        self.final_fc1 = nn.Linear(1024, 512)
        self.final_fc2 = nn.Linear(512, 120)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        y = x.detach().clone()

        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        x = self.resnet.avgpool(x)
        x = x.view(x.size(0), -1)
        x = nn.functional.relu(self.rfc1(x))

        y = self.densenet.features(y)
        y = F.relu(y)
        y = F.adaptive_avg_pool2d(y, (1, 1))
        y = y.view(y.size(0), -1)
        y = nn.functional.relu(self.dfc1(y))

        x = torch.cat((x, y), 1)
        x = nn.functional.relu(self.final_fc1(x))
        x = self.dropout(x)
        x = self.final_fc2(x)

        return x


class ComplexDog(nn.Module):
    def __init__(self):
        super(ComplexDog, self).__init__()

        model_vgg = models.vgg19(pretrained=True)

        for name, param in model_vgg.named_parameters():
            if ("bn" not in name):
                param.requires_grad = False

        model_vgg.classifier[6] = nn.Linear(4096, 512)
        self.vgg = model_vgg
        self.rfc1 = nn.Linear(512, 512)
        model_dense=models.densenet121(pretrained=True)
        self.densenet = model_dense

        for name, param in model_dense.named_parameters():
            if ("bn" not in name):
                param.requires_grad = False


        self.dfc1 = nn.Linear(1024, 512)

        self.final_fc1 = nn.Linear(1024, 512)
        self.final_fc2 = nn.Linear(512, 120)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        y = x.detach().clone()

        x = self.vgg(x)
        x = x.view(x.size(0), -1)
        x = nn.functional.relu(self.rfc1(x))

        y = self.densenet.features(y)
        y = F.relu(y)
        y = F.adaptive_avg_pool2d(y, (1, 1))
        y = y.view(y.size(0), -1)
        y = nn.functional.relu(self.dfc1(y))

        x = torch.cat((x, y), 1)
        x = nn.functional.relu(self.final_fc1(x))
        x = self.dropout(x)
        x = self.final_fc2(x)

        return x

class ComplexDogAlex(nn.Module):
    def __init__(self):
        super(ComplexDogAlex, self).__init__()

        model_vgg = models.vgg19(pretrained=True)

        for name, param in model_vgg.named_parameters():
            if ("bn" not in name):
                param.requires_grad = False

        model_vgg.classifier[6] = nn.Linear(4096, 512)
        self.vgg = model_vgg
        self.rfc1 = nn.Linear(512, 512)
        #model_dense=models.densenet121(pretrained=True)
        model_alex=models.alexnet(pretrained=True)
        model_alex.classifier[6] = nn.Linear(4096, 512)

        self.alexnet = model_alex

        for name, param in model_alex.named_parameters():
            if ("bn" not in name):
                param.requires_grad = False


        self.dfc1 = nn.Linear(1024, 512)

        self.final_fc1 = nn.Linear(1024, 512)
        self.final_fc2 = nn.Linear(512, 120)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        y = x.detach().clone()

        x = self.vgg(x)
        x = x.view(x.size(0), -1)
        x = nn.functional.relu(self.rfc1(x))

        #y = self.densenet.features(y)
        y = self.alexnet.features(x)
        y = F.relu(y)
        y = F.adaptive_avg_pool2d(y, (1, 1))
        y = y.view(y.size(0), -1)
        y = nn.functional.relu(self.dfc1(y))

        x = torch.cat((x, y), 1)
        x = nn.functional.relu(self.final_fc1(x))
        x = self.dropout(x)
        x = self.final_fc2(x)

        return x