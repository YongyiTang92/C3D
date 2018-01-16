import torch.nn as nn
import math
import torch


def conv3x3x3(in_planes, out_planes, stride=1, padding=1, kernel_size=3):
    "3x3x3 convolution with padding"
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class C3D_BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(C3D_BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class C21D_BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(C21D_BasicBlock, self).__init__()
        # self.conv1 = conv3x3x3(inplanes, planes, (1, stride, stride), kernel_size=(1, 3, 3))
        # self.conv1_2 = conv3x3x3(planes, planes, (stride, 1, 1), kernel_size=(3, 1, 1))
        # self.bn1 = nn.BatchNorm3d(planes)
        # self.relu = nn.ReLU(inplace=True)
        # self.conv2 = conv3x3x3(planes, planes, kernel_size=(1, 3, 3))
        # self.conv2_2 = conv3x3x3(planes, planes, kernel_size=(3, 1, 1))
        # self.bn2 = nn.BatchNorm3d(planes)
        # self.downsample = downsample
        # self.stride = stride

        self.net = nn.Sequential(
            conv3x3x3(inplanes, planes, (1, stride, stride), kernel_size=(1, 3, 3)),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True),
            conv3x3x3(planes, planes, (stride, 1, 1), kernel_size=(3, 1, 1)),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True),
            conv3x3x3(planes, planes, kernel_size=(1, 3, 3)),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True),
            conv3x3x3(planes, planes, kernel_size=(3, 1, 1)),
            nn.BatchNorm3d(planes),
        )

    def forward(self, x):
        residual = x

        # out = self.conv1(x)
        # out = self.relu(out)
        # out = self.conv1_2(out)
        # out = self.bn1(out)
        # out = self.relu(out)

        # out = self.conv2(out)
        # out = self.relu(out)
        # out = self.conv2_2(out)
        # out = self.bn2(out)
        out = self.net(x)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet_3D(nn.Module):
    def __init__(self, block, layers, num_classes=101, in_channel=3):
        self.inplanes = 64
        super(ResNet_3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channel, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.drop = nn.Dropout(0.9)
        self.avgpool = nn.AvgPool3d((1, 7, 7))
        # self.fc = nn.Linear(512 * block.expansion, 1000)
        self.fc_new = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.drop(x)
        x = self.fc_new(x)

        return x


def R3D_34():
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_3D(C3D_BasicBlock, [3, 4, 6, 3])
    return model


def R21D_34():
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_3D(C21D_BasicBlock, [3, 4, 6, 3])
    return model
