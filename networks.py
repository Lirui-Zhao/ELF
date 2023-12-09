import torch.nn as nn
import torch.nn.functional as F
import torch
import sys

# Acknowledgement to
# https://github.com/kuangliu/pytorch-cifar,
# https://github.com/BIGBALLON/CIFAR-ZOO,

# adapted from
# https://github.com/VICO-UoE/DatasetCondensation
# https://github.com/GeorgeCazenavette/mtt-distillation

''' Swish activation '''

class Swish(nn.Module): # Swish(x) = x∗σ(x)
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.sigmoid(input)


''' MLP '''

class MLP(nn.Module):
    def __init__(self, channel, num_classes):
        super(MLP, self).__init__()
        self.fc_1 = nn.Linear(28 * 28 * 1 if channel == 1 else 32 * 32 * 3, 128)
        self.fc_2 = nn.Linear(128, 128)
        self.fc_3 = nn.Linear(128, num_classes)

    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = F.relu(self.fc_1(out))
        out = F.relu(self.fc_2(out))
        out = self.fc_3(out)
        return out


''' ConvNet '''

class ConvNet(nn.Module):
    def __init__(
        self,
        channel,
        num_classes,
        net_width,
        net_depth,
        net_act,
        net_norm,
        net_pooling,
        im_size=(32, 32),
    ):
        super(ConvNet, self).__init__()

        self.features, shape_feat = self._make_layers(
            channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size
        )
        num_feat = shape_feat[0] * shape_feat[1] * shape_feat[2]
        self.classifier = nn.Linear(num_feat, num_classes)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def embed(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        return out
    
    def _get_activation(self, net_act):
        if net_act == 'sigmoid':
            return nn.Sigmoid()
        elif net_act == 'relu':
            return nn.ReLU(inplace=True)
        elif net_act == 'leakyrelu':
            return nn.LeakyReLU(negative_slope=0.01)
        else:
            exit('unknown activation function: %s' % net_act)

    def _get_pooling(self, net_pooling):
        if net_pooling == 'maxpooling':
            return nn.MaxPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'avgpooling':
            return nn.AvgPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'none':
            return None
        else:
            exit('unknown net_pooling: %s' % net_pooling)

    def _get_normlayer(self, net_norm, shape_feat):
        # shape_feat = (c*h*w)
        if net_norm == 'batchnorm':
            return nn.BatchNorm2d(shape_feat[0], affine=True)
        elif net_norm == 'layernorm':
            return nn.LayerNorm(shape_feat, elementwise_affine=True)
        elif net_norm == 'instancenorm':
            return nn.GroupNorm(shape_feat[0], shape_feat[0], affine=True)
        elif net_norm == 'groupnorm':
            return nn.GroupNorm(4, shape_feat[0], affine=True)
        elif net_norm == 'none':
            return None
        else:
            exit('unknown net_norm: %s' % net_norm)

    def _make_layers(
        self, channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size
    ):
        layers = []
        in_channels = channel
        # 专门为MNIST设置的，这就是从bozhao继承下来的屎山导致的，自己写的改好了
        if im_size[0] == 28:
            im_size = (32, 32)
        shape_feat = [in_channels, im_size[0], im_size[1]]
        # ConvNet的depth，默认为3
        for d in range(net_depth):
            # 卷积
            layers += [
                nn.Conv2d(
                    in_channels,
                    net_width,
                    kernel_size=3,
                    padding=3 if channel == 1 and d == 0 else 1,
                )
            ]
            shape_feat[0] = net_width
            # norm
            if net_norm != 'none':
                layers += [self._get_normlayer(net_norm, shape_feat)]
            # 激活
            layers += [self._get_activation(net_act)]
            in_channels = net_width
            # 池化
            if net_pooling != 'none':
                layers += [self._get_pooling(net_pooling)]
                shape_feat[1] //= 2
                shape_feat[2] //= 2

        return nn.Sequential(*layers), shape_feat


''' ConvNet '''

class ConvNetGAP(nn.Module):
    def __init__(
        self,
        channel,
        num_classes,
        net_width,
        net_depth,
        net_act,
        net_norm,
        net_pooling,
        im_size=(32, 32),
    ):
        super(ConvNetGAP, self).__init__()

        self.features, shape_feat = self._make_layers(
            channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size
        )
        num_feat = shape_feat[0] * shape_feat[1] * shape_feat[2]
        # self.classifier = nn.Linear(num_feat, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(shape_feat[0], num_classes)

    def forward(self, x):
        out = self.features(x)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _get_activation(self, net_act):
        if net_act == 'sigmoid':
            return nn.Sigmoid()
        elif net_act == 'relu':
            return nn.ReLU(inplace=True)
        elif net_act == 'leakyrelu':
            return nn.LeakyReLU(negative_slope=0.01)
        elif net_act == 'swish':
            return Swish()
        else:
            exit('unknown activation function: %s' % net_act)

    def _get_pooling(self, net_pooling):
        if net_pooling == 'maxpooling':
            return nn.MaxPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'avgpooling':
            return nn.AvgPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'none':
            return None
        else:
            exit('unknown net_pooling: %s' % net_pooling)

    def _get_normlayer(self, net_norm, shape_feat):
        # shape_feat = (c*h*w)
        if net_norm == 'batchnorm':
            return nn.BatchNorm2d(shape_feat[0], affine=True)
        elif net_norm == 'layernorm':
            return nn.LayerNorm(shape_feat, elementwise_affine=True)
        elif net_norm == 'instancenorm':
            return nn.GroupNorm(shape_feat[0], shape_feat[0], affine=True)
        elif net_norm == 'groupnorm':
            return nn.GroupNorm(4, shape_feat[0], affine=True)
        elif net_norm == 'none':
            return None
        else:
            exit('unknown net_norm: %s' % net_norm)

    def _make_layers(
        self, channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size
    ):
        layers = []
        in_channels = channel
        if im_size[0] == 28:
            im_size = (32, 32)
        shape_feat = [in_channels, im_size[0], im_size[1]]
        for d in range(net_depth):
            layers += [
                nn.Conv2d(
                    in_channels,
                    net_width,
                    kernel_size=3,
                    padding=3 if channel == 1 and d == 0 else 1,
                )
            ]
            shape_feat[0] = net_width
            if net_norm != 'none':
                layers += [self._get_normlayer(net_norm, shape_feat)]
            layers += [self._get_activation(net_act)]
            in_channels = net_width
            if net_pooling != 'none':
                layers += [self._get_pooling(net_pooling)]
                shape_feat[1] //= 2
                shape_feat[2] //= 2

        return nn.Sequential(*layers), shape_feat


''' LeNet '''

class LeNet(nn.Module):
    def __init__(self, channel, num_classes):
        super(LeNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channel, 6, kernel_size=5, padding=2 if channel==1 else 0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc_1 = nn.Linear(16 * 5 * 5, 120)
        self.fc_2 = nn.Linear(120, 84)
        self.fc_3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))
        x = self.fc_3(x)
        return x




''' AlexNet '''

class AlexNet(nn.Module):
    def __init__(self, channel, num_classes):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(
                channel, 128, kernel_size=5, stride=1, padding=4 if channel == 1 else 2
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc = nn.Linear(192 * 4 * 4, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def embed(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x

''' AlexNetBN '''

class AlexNetBN(nn.Module):
    def __init__(self, channel, num_classes):
        super(AlexNetBN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channel, 128, kernel_size=5, stride=1, padding=4 if channel==1 else 2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 192, kernel_size=5, padding=2),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc = nn.Linear(192 * 4 * 4, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def embed(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x

''' VGG '''

cfg_vgg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [
        64,
        64,
        'M',
        128,
        128,
        'M',
        256,
        256,
        256,
        'M',
        512,
        512,
        512,
        'M',
        512,
        512,
        512,
        'M',
    ],
    'VGG19': [
        64,
        64,
        'M',
        128,
        128,
        'M',
        256,
        256,
        256,
        256,
        'M',
        512,
        512,
        512,
        512,
        'M',
        512,
        512,
        512,
        512,
        'M',
    ],
}


class VGG(nn.Module):
    def __init__(self, vgg_name, channel, num_classes, norm='instancenorm'):
        super(VGG, self).__init__()
        self.channel = channel
        self.features = self._make_layers(cfg_vgg[vgg_name], norm)
        self.classifier = nn.Linear(512 if vgg_name != 'VGGS' else 128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def embed(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x

    def _make_layers(self, cfg, norm):
        layers = []
        in_channels = self.channel
        for ic, x in enumerate(cfg):
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [
                    nn.Conv2d(
                        in_channels,
                        x,
                        kernel_size=3,
                        padding=3 if self.channel == 1 and ic == 0 else 1,
                    ),
                    nn.GroupNorm(x, x, affine=True)
                    if norm == 'instancenorm'
                    else nn.BatchNorm2d(x),
                    nn.ReLU(inplace=True),
                ]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def VGG11(channel, num_classes):
    return VGG('VGG11', channel, num_classes)


def VGG11BN(channel, num_classes):
    return VGG('VGG11', channel, num_classes, norm='batchnorm')


def VGG13(channel, num_classes):
    return VGG('VGG13', channel, num_classes)


def VGG16(channel, num_classes):
    return VGG('VGG16', channel, num_classes)


def VGG19(channel, num_classes):
    return VGG('VGG19', channel, num_classes)


class VGG_feature(nn.Module):
    def __init__(self, vgg_name, feature_depth, channel, num_classes, norm='instancenorm'):
        super(VGG_feature, self).__init__()
        self.channel = channel
        self.num_in_list=[i for i,x in enumerate(cfg_vgg[vgg_name])  if x!='M'][feature_depth]
        self.layer1 = self._make_layers(cfg_vgg[vgg_name][:self.num_in_list], norm, AvgPool2d=False)
        self.layer2 = self._make_layers(cfg_vgg[vgg_name][self.num_in_list:], norm, AvgPool2d=True)
        self.classifier = nn.Linear(512 if vgg_name != 'VGGS' else 128, num_classes)

    def forward(self, img=None,feature=None):
        out1=None
        if img != None:
            out1 = self.layer1(img)
            feature = out1
        out2 = self.layer2(feature)
        out2 = out2.view(out2.size(0), -1)
        out2 = self.classifier(out2)
        return out1,out2

    def _make_layers(self, cfg, norm, AvgPool2d):
        layers = []
        in_channels = self.channel
        for ic, x in enumerate(cfg):
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [
                    nn.Conv2d(
                        in_channels,
                        x,
                        kernel_size=3,
                        padding=3 if self.channel == 1 and ic == 0 else 1,
                    ),
                    nn.GroupNorm(x, x, affine=True)
                    if norm == 'instancenorm'
                    else nn.BatchNorm2d(x),
                    nn.ReLU(inplace=True),
                ]
                in_channels = x
                self.channel = in_channels 
        if AvgPool2d:
            layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def VGG11_feature(feature_depth,channel, num_classes):
    return VGG_feature('VGG11',feature_depth, channel, num_classes)

def VGG11BN_feature(feature_depth,channel, num_classes):
    return VGG_feature('VGG11',feature_depth, channel, num_classes, norm='batchnorm')


class VGG_ImageNet_feature(nn.Module):
    def __init__(self, vgg_name,feature_depth, channel, num_classes, norm='instancenorm'):
        super(VGG_ImageNet_feature, self).__init__()
        self.channel = channel
        # self.features = self._make_layers(cfg_vgg[vgg_name], norm)
        self.num_in_list=[i for i,x in enumerate(cfg_vgg[vgg_name])  if x!='M'][feature_depth]
        self.layer1 = self._make_layers(cfg_vgg[vgg_name][:self.num_in_list], norm)
        self.layer2 = self._make_layers(cfg_vgg[vgg_name][self.num_in_list:], norm)
        self.classifier = nn.Sequential(
            nn.Linear(2048, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )

    def forward(self, img = None, feature = None):
        out1 = None
        if img != None:
            out1 = self.layer1(img)
            feature = out1
        out2 = self.layer2(feature)
        out2 = out2.view(out2.size(0), -1)
        out2 = self.classifier(out2)
        return out1,out2

    def _make_layers(self, cfg, norm):
        layers = []
        in_channels = self.channel
        for ic, x in enumerate(cfg):
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [
                    nn.Conv2d(
                        in_channels,
                        x,
                        kernel_size=3,
                        padding=1,
                    ),
                    nn.GroupNorm(x, x, affine=True)
                    if norm == 'instancenorm'
                    else nn.BatchNorm2d(x),
                    nn.ReLU(inplace=True),
                ]
                in_channels = x
                self.channel = in_channels 
        # layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

def VGG11ImageNet_feature(feature_depth,channel, num_classes):
    return VGG_ImageNet_feature('VGG11', feature_depth,channel, num_classes)

def VGG11BNImageNet_feature(feature_depth,channel, num_classes):
    return VGG_ImageNet_feature('VGG11', feature_depth,channel, num_classes,norm='batchnorm')


class VGG_ImageNet(nn.Module):
    def __init__(self, vgg_name, channel, num_classes, norm='instancenorm'):
        super(VGG_ImageNet, self).__init__()
        self.channel = channel
        # self.features = self._make_layers(cfg_vgg[vgg_name], norm)
        self.layer = self._make_layers(cfg_vgg[vgg_name], norm)
        self.classifier = nn.Sequential(
            nn.Linear(2048, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        out = self.layer(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg, norm):
        layers = []
        in_channels = self.channel
        for ic, x in enumerate(cfg):
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [
                    nn.Conv2d(
                        in_channels,
                        x,
                        kernel_size=3,
                        padding=1,
                    ),
                    nn.GroupNorm(x, x, affine=True)
                    if norm == 'instancenorm'
                    else nn.BatchNorm2d(x),
                    nn.ReLU(inplace=True),
                ]
                in_channels = x
                self.channel = in_channels 
        # layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def VGG11ImageNet(channel, num_classes):
    return VGG_ImageNet('VGG11',channel, num_classes)

def VGG11BNImageNet(channel, num_classes):
    return VGG_ImageNet('VGG11',channel, num_classes,norm='batchnorm')


''' ResNet_AP '''
# The conv(stride=2) is replaced by conv(stride=1) + avgpool(kernel_size=2, stride=2)
class BasicBlock_AP(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, norm='instancenorm'):
        super(BasicBlock_AP, self).__init__()
        self.norm = norm
        self.stride = stride
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )  # modification
        self.bn1 = (
            nn.GroupNorm(planes, planes, affine=True)
            if self.norm == 'instancenorm'
            else nn.BatchNorm2d(planes)
        )
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = (
            nn.GroupNorm(planes, planes, affine=True)
            if self.norm == 'instancenorm'
            else nn.BatchNorm2d(planes)
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=1,
                    bias=False,
                ),
                nn.AvgPool2d(kernel_size=2, stride=2),  # modification
                nn.GroupNorm(
                    self.expansion * planes, self.expansion * planes, affine=True
                )
                if self.norm == 'instancenorm'
                else nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        if self.stride != 1:  # modification
            out = F.avg_pool2d(out, kernel_size=2, stride=2)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck_AP(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, norm='instancenorm'):
        super(Bottleneck_AP, self).__init__()
        self.norm = norm
        self.stride = stride
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = (
            nn.GroupNorm(planes, planes, affine=True)
            if self.norm == 'instancenorm'
            else nn.BatchNorm2d(planes)
        )
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )  # modification
        self.bn2 = (
            nn.GroupNorm(planes, planes, affine=True)
            if self.norm == 'instancenorm'
            else nn.BatchNorm2d(planes)
        )
        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )
        self.bn3 = (
            nn.GroupNorm(self.expansion * planes, self.expansion * planes, affine=True)
            if self.norm == 'instancenorm'
            else nn.BatchNorm2d(self.expansion * planes)
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=1,
                    bias=False,
                ),
                nn.AvgPool2d(kernel_size=2, stride=2),  # modification
                nn.GroupNorm(
                    self.expansion * planes, self.expansion * planes, affine=True
                )
                if self.norm == 'instancenorm'
                else nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        if self.stride != 1:  # modification
            out = F.avg_pool2d(out, kernel_size=2, stride=2)
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet_AP(nn.Module):
    def __init__(
        self, block, num_blocks, channel=3, num_classes=10, norm='instancenorm'
    ):
        super(ResNet_AP, self).__init__()
        self.in_planes = 64
        self.norm = norm

        self.conv1 = nn.Conv2d(
            channel, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = (
            nn.GroupNorm(64, 64, affine=True)
            if self.norm == 'instancenorm'
            else nn.BatchNorm2d(64)
        )
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.classifier = nn.Linear(
            512 * block.expansion * 3 * 3
            if channel == 1
            else 512 * block.expansion * 4 * 4,
            num_classes,
        )  # modification

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.norm))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, kernel_size=1, stride=1)  # modification
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out
    
    def embed(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, kernel_size=1, stride=1) # modification
        out = out.view(out.size(0), -1)
        return out


def ResNet18BN_AP(channel, num_classes):
    return ResNet_AP(
        BasicBlock_AP,
        [2, 2, 2, 2],
        channel=channel,
        num_classes=num_classes,
        norm='batchnorm',
    )


def ResNet18_AP(channel, num_classes):
    return ResNet_AP(
        BasicBlock_AP, [2, 2, 2, 2], channel=channel, num_classes=num_classes
    )


''' ResNet '''
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, norm='instancenorm'):
        super(BasicBlock, self).__init__()
        self.norm = norm
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = (
            nn.GroupNorm(planes, planes, affine=True)
            if self.norm == 'instancenorm'
            else nn.BatchNorm2d(planes)
        )
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = (
            nn.GroupNorm(planes, planes, affine=True)
            if self.norm == 'instancenorm'
            else nn.BatchNorm2d(planes)
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.GroupNorm(
                    self.expansion * planes, self.expansion * planes, affine=True
                )
                if self.norm == 'instancenorm'
                else nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, norm='instancenorm'):
        super(Bottleneck, self).__init__()
        self.norm = norm
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = (
            nn.GroupNorm(planes, planes, affine=True)
            if self.norm == 'instancenorm'
            else nn.BatchNorm2d(planes)
        )
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = (
            nn.GroupNorm(planes, planes, affine=True)
            if self.norm == 'instancenorm'
            else nn.BatchNorm2d(planes)
        )
        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )
        self.bn3 = (
            nn.GroupNorm(self.expansion * planes, self.expansion * planes, affine=True)
            if self.norm == 'instancenorm'
            else nn.BatchNorm2d(self.expansion * planes)
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.GroupNorm(
                    self.expansion * planes, self.expansion * planes, affine=True
                )
                if self.norm == 'instancenorm'
                else nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(
        self, block, num_blocks, channel=3, num_classes=10, norm='instancenorm'
    ):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.norm = norm

        self.conv1 = nn.Conv2d(
            channel, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = (
            nn.GroupNorm(64, 64, affine=True)
            if self.norm == 'instancenorm'
            else nn.BatchNorm2d(64)
        )
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.classifier = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.norm))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out
    
    def embed(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return out


def ResNet18BN(channel, num_classes):
    return ResNet(
        BasicBlock,
        [2, 2, 2, 2],
        channel=channel,
        num_classes=num_classes,
        norm='batchnorm',
    )


def ResNet18(channel, num_classes):
    return ResNet(BasicBlock, [2, 2, 2, 2], channel=channel, num_classes=num_classes)


def ResNet34(channel, num_classes):
    return ResNet(BasicBlock, [3, 4, 6, 3], channel=channel, num_classes=num_classes)


def ResNet50(channel, num_classes):
    return ResNet(Bottleneck, [3, 4, 6, 3], channel=channel, num_classes=num_classes)


def ResNet101(channel, num_classes):
    return ResNet(Bottleneck, [3, 4, 23, 3], channel=channel, num_classes=num_classes)


def ResNet152(channel, num_classes):
    return ResNet(Bottleneck, [3, 8, 36, 3], channel=channel, num_classes=num_classes)

class ResNetImageNet(nn.Module):
    def __init__(
        self, block, num_blocks, channel=3, num_classes=10, norm='instancenorm'
    ):
        super(ResNetImageNet, self).__init__()
        self.in_planes = 64
        self.norm = norm

        self.conv1 = nn.Conv2d(
            channel, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = (
            nn.GroupNorm(64, 64, affine=True)
            if self.norm == 'instancenorm'
            else nn.BatchNorm2d(64)
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.norm))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # out = F.avg_pool2d(out, 4)
        # out = out.view(out.size(0), -1)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out



def ResNet18ImageNet(channel, num_classes):
    return ResNetImageNet(
        BasicBlock, [2, 2, 2, 2], channel=channel, num_classes=num_classes
    )

def ResNet18BNImageNet(channel, num_classes):
    return ResNetImageNet(
        BasicBlock, [2, 2, 2, 2], channel=channel, num_classes=num_classes, norm='batchnorm'
    )

def ResNet6ImageNet(channel, num_classes):
    return ResNetImageNet(
        BasicBlock, [1, 1, 1, 1], channel=channel, num_classes=num_classes
    )


''' ConvNetD4'''

class ConvNet_feature(nn.Module):
    def __init__(
        self,
        channel,
        num_classes,
        net_width,
        net_depth,
        feature_depth,
        net_act,
        net_norm,
        net_pooling,
        im_size=(32, 32),
    ):
        super(ConvNet_feature, self).__init__()
        shape_feat = [channel, im_size[0], im_size[1]]
        self.layer1, shape_feat = self._make_layers(
            shape_feat, net_width, feature_depth, net_norm, net_act, net_pooling
        )
        self.layer2, shape_feat = self._make_layers(
            shape_feat, net_width, net_depth - feature_depth, net_norm, net_act, net_pooling
        )
        num_feat = shape_feat[0] * shape_feat[1] * shape_feat[2]
        self.classifier = nn.Linear(num_feat, num_classes)

    def forward(self, img=None , feature=None):
        out1 = None
        if img != None:
            out1=self.layer1(img)
            feature= out1
        out2 = self.layer2(feature)
        out2 = out2.view(out2.size(0), -1)
        out2 = self.classifier(out2)
        return out1,out2

    def _get_activation(self, net_act):
        if net_act == 'sigmoid':
            return nn.Sigmoid()
        elif net_act == 'relu':
            return nn.ReLU(inplace=True)
        elif net_act == 'leakyrelu':
            return nn.LeakyReLU(negative_slope=0.01)
        else:
            exit('unknown activation function: %s' % net_act)

    def _get_pooling(self, net_pooling):
        if net_pooling == 'maxpooling':
            return nn.MaxPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'avgpooling':
            return nn.AvgPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'none':
            return None
        else:
            exit('unknown net_pooling: %s' % net_pooling)

    def _get_normlayer(self, net_norm, shape_feat):
        # shape_feat = (c*h*w)
        if net_norm == 'batchnorm':
            return nn.BatchNorm2d(shape_feat[0], affine=True)
        elif net_norm == 'layernorm':
            return nn.LayerNorm(shape_feat, elementwise_affine=True)
        elif net_norm == 'instancenorm':
            return nn.GroupNorm(shape_feat[0], shape_feat[0], affine=True)
        elif net_norm == 'groupnorm':
            return nn.GroupNorm(4, shape_feat[0], affine=True)
        elif net_norm == 'none':
            return None
        else:
            exit('unknown net_norm: %s' % net_norm)

    def _make_layers(
        self, shape_feat, net_width, layer_depth, net_norm, net_act, net_pooling
    ):
        layers = []
        in_channels = shape_feat[0]
        
        for _ in range(layer_depth):
            # 卷积
            layers += [
                nn.Conv2d(
                    in_channels,
                    net_width,
                    kernel_size=3,
                    padding= 1,
                )
            ]
            shape_feat[0] = net_width
            # norm
            if net_norm != 'none':
                layers += [self._get_normlayer(net_norm, shape_feat)]
            # 激活
            layers += [self._get_activation(net_act)]
            
            in_channels = net_width
            # 池化
            if net_pooling != 'none':
                layers += [self._get_pooling(net_pooling)]
                shape_feat[1] //= 2
                shape_feat[2] //= 2

        return nn.Sequential(*layers), shape_feat


''' LayeredResNet'''
class BasicBlock_Layered(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, norm='instancenorm'):
        super(BasicBlock_Layered, self).__init__()
        self.norm = norm
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = (
            nn.GroupNorm(planes, planes, affine=True)
            if self.norm == 'instancenorm'
            else nn.BatchNorm2d(planes)
        )
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = (
            nn.GroupNorm(planes, planes, affine=True)
            if self.norm == 'instancenorm'
            else nn.BatchNorm2d(planes)
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.GroupNorm(
                    self.expansion * planes, self.expansion * planes, affine=True
                )
                if self.norm == 'instancenorm'
                else nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck_Layered(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, norm='instancenorm'):
        super(Bottleneck_Layered, self).__init__()
        self.norm = norm
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = (
            nn.GroupNorm(planes, planes, affine=True)
            if self.norm == 'instancenorm'
            else nn.BatchNorm2d(planes)
        )
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = (
            nn.GroupNorm(planes, planes, affine=True)
            if self.norm == 'instancenorm'
            else nn.BatchNorm2d(planes)
        )
        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )
        self.bn3 = (
            nn.GroupNorm(self.expansion * planes, self.expansion * planes, affine=True)
            if self.norm == 'instancenorm'
            else nn.BatchNorm2d(self.expansion * planes)
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.GroupNorm(
                    self.expansion * planes, self.expansion * planes, affine=True
                )
                if self.norm == 'instancenorm'
                else nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet_Layered(nn.Module):
    def __init__(
        self, block, num_blocks, channel=3, num_classes=10, norm='instancenorm'
    ):
        super(ResNet_Layered, self).__init__()
        self.in_planes = 64
        self.norm = norm

        self.conv1 = nn.Conv2d(
            channel, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = (
            nn.GroupNorm(64, 64, affine=True)
            if self.norm == 'instancenorm'
            else nn.BatchNorm2d(64)
        )
        self.layer2 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer3 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer4 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer5_befor = self._make_half_layer(block, 512, num_blocks[3], stride=2)
        self.layer5_after = self._make_half_layer(block, 512, num_blocks[3], stride=1)
        self.classifier = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.norm))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _make_half_layer(self, block, planes, num_blocks, stride):
        layers = []
        layers.append(block(self.in_planes, planes, stride, self.norm))
        self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, img=None,feature=None):
        out1 = None
        if img != None:
            out1 = F.relu(self.bn1(self.conv1(img)))
            out1 = self.layer2(out1) 
            out1 = self.layer3(out1)
            out1 = self.layer4(out1)
            out1 = self.layer5_befor(out1)
            feature = out1
        out2 = self.layer5_after(feature)
        out2 = F.avg_pool2d(out2, 4)
        out2 = out2.view(out2.size(0), -1)
        out2 = self.classifier(out2)
        return out1,out2

def ResNet18_Layered(channel, num_classes):
    return ResNet_Layered(BasicBlock_Layered, [2, 2, 2, 2], channel=channel, num_classes=num_classes)

def ResNet18BN_Layered(channel, num_classes):
    return ResNet_Layered(BasicBlock_Layered, [2, 2, 2, 2], channel=channel, num_classes=num_classes,norm='batchnorm')


class ResNetImageNet_L4(nn.Module):
    def __init__(
        self, block, num_blocks, channel=3, num_classes=10, norm='instancenorm'
    ):
        super(ResNetImageNet_L4, self).__init__()
        self.in_planes = 64
        self.norm = norm

        self.conv1 = nn.Conv2d(
            channel, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = (
            nn.GroupNorm(64, 64, affine=True)
            if self.norm == 'instancenorm'
            else nn.BatchNorm2d(64)
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer2 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer3 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer4 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer5 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.norm))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, img=None,feature=None):
        out1 = None
        if img != None:
            out1 = F.relu(self.bn1(self.conv1(img)))
            out1 = self.maxpool(out1)
            out1 = self.layer2(out1)
            out1 = self.layer3(out1)
            out1 = self.layer4(out1)
            feature = out1
        
        out2 = self.layer5(feature)
        
        out2 = self.avgpool(out2)
        out2 = torch.flatten(out2, 1)
        out2 = self.classifier(out2)
        return out1, out2

def ResNet18ImageNet_L4(channel, num_classes):
    return ResNetImageNet_L4(
        BasicBlock, [2, 2, 2, 2], channel=channel, num_classes=num_classes
    )

def ResNet18BNImageNet_L4(channel, num_classes):
    return ResNetImageNet_L4(
        BasicBlock, [2, 2, 2, 2], channel=channel, num_classes=num_classes,norm='batchnorm'
    )

