# adapted from
# https://github.com/GeorgeCazenavette/mtt-distillation

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import kornia as K
import torch.optim as optim
import tqdm
import random
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from scipy.ndimage.interpolation import rotate as scipyrotate
from reparam_module import ReparamModule
from networks import (
    MLP,
    ConvNet,
    LeNet,
    AlexNet,
    VGG11BN,
    VGG11,
    ResNet18,
    ResNet18BN,
    ResNet18BN_AP,
    ResNet18_AP,
    ConvNet_feature,
    ResNet18_Layered,
    ResNet18BN_Layered,
    VGG11_feature,
    VGG11BN_feature,
    ResNet18ImageNet,
    ResNet18BNImageNet,
    VGG11BNImageNet,
    VGG11ImageNet,
    ResNet18ImageNet_L4,
    ResNet18BNImageNet_L4,
    VGG11ImageNet_feature,
    VGG11BNImageNet_feature,
)


class Config:
    imagenette = [0, 217, 482, 491, 497, 566, 569, 571, 574, 701]

    # ["australian_terrier", "border_terrier", "samoyed", "beagle", "shih-tzu", "english_foxhound", "rhodesian_ridgeback", "dingo", "golden_retriever", "english_sheepdog"]
    imagewoof = [193, 182, 258, 162, 155, 167, 159, 273, 207, 229]

    # ["tabby_cat", "bengal_cat", "persian_cat", "siamese_cat", "egyptian_cat", "lion", "tiger", "jaguar", "snow_leopard", "lynx"]
    imagemeow = [281, 282, 283, 284, 285, 291, 292, 290, 289, 287]

    # ["peacock", "flamingo", "macaw", "pelican", "king_penguin", "bald_eagle", "toucan", "ostrich", "black_swan", "cockatoo"]
    imagesquawk = [84, 130, 88, 144, 145, 22, 96, 9, 100, 89]

    # ["pineapple", "banana", "strawberry", "orange", "lemon", "pomegranate", "fig", "bell_pepper", "cucumber", "green_apple"]
    imagefruit = [953, 954, 949, 950, 951, 957, 952, 945, 943, 948]

    # ["bee", "ladys slipper", "banana", "lemon", "corn", "school_bus", "honeycomb", "lion", "garden_spider", "goldfinch"]
    imageyellow = [309, 986, 954, 951, 987, 779, 599, 291, 72, 11]

    dict = {
        "imagenette": imagenette,
        "imagewoof": imagewoof,
        "imagefruit": imagefruit,
        "imageyellow": imageyellow,
        "imagemeow": imagemeow,
        "imagesquawk": imagesquawk,
    }


config = Config()


def get_dataset(dataset, data_path, batch_size=1, subset="imagenette", args=None):

    class_map = None
    loader_train_dict = None
    class_map_inv = None

    if dataset == 'CIFAR10':
        channel = 3
        im_size = (32, 32)
        num_classes = 10
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        if args.zca:
            transform = transforms.Compose([transforms.ToTensor()])
        else:
            transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
            )
        dst_train = datasets.CIFAR10(
            data_path, train=True, download=True, transform=transform
        )  # no augmentation
        dst_test = datasets.CIFAR10(
            data_path, train=False, download=True, transform=transform
        )
        class_names = dst_train.classes
        class_map = {x: x for x in range(num_classes)}
    elif dataset == 'CIFAR100':
        channel = 3
        im_size = (32, 32)
        num_classes = 100
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]

        if args.zca:
            transform = transforms.Compose([transforms.ToTensor()])
        else:
            transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
            )
        dst_train = datasets.CIFAR100(
            data_path, train=True, download=True, transform=transform
        )  # no augmentation
        dst_test = datasets.CIFAR100(
            data_path, train=False, download=True, transform=transform
        )
        class_names = dst_train.classes
        class_map = {x: x for x in range(num_classes)}
    elif dataset == 'Tiny':
        channel = 3
        im_size = (64, 64)
        num_classes = 200
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        if args.zca:
            transform = transforms.Compose([transforms.ToTensor()])
        else:
            transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
            )
        dst_train = datasets.ImageFolder(
            os.path.join(data_path, "train"), transform=transform
        )  # no augmentation
        dst_test = datasets.ImageFolder(
            os.path.join(data_path, "val"), transform=transform
        )
        class_names = dst_train.classes
        class_map = {x: x for x in range(num_classes)}
    elif dataset =='ImageNet':
        channel = 3
        im_size = (64, 64)
        num_classes = 1000
        ###########################################################################################
        ###########################################################################################
        #######################please update here for ImageNet path################################
        ###########################################################################################
        # data_path = 'YOUR_DATA_PATH_HERE'

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize(im_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(im_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        # dst_train = datasets.ImageFolder(os.path.join(data_path, "ILSVRC2012_img_train"), transform=data_transforms['train']) # no augmentation
        dst_train = datasets.ImageFolder(os.path.join(data_path, "train"), transform=data_transforms['train']) 
        dst_test = datasets.ImageFolder(os.path.join(data_path, "val"), transform=data_transforms['val'])
        class_names = dst_train.classes
        class_map = {x:x for x in range(num_classes)}
    elif dataset == 'ImageNet128':
        channel = 3
        im_size = (128, 128)
        num_classes = 10

        config.img_net_classes = config.dict[subset]

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        if args.zca:
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize(im_size),
                    transforms.CenterCrop(im_size),
                ]
            )
        else:
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std),
                    transforms.Resize(im_size),
                    transforms.CenterCrop(im_size),
                ]
            )

        dst_train = datasets.ImageNet(
            data_path, split="train", transform=transform
        )  # no augmentation
        dst_train_dict = {
            c: torch.utils.data.Subset(
                dst_train,
                np.squeeze(
                    np.argwhere(np.equal(dst_train.targets, config.img_net_classes[c]))
                ),
            )
            for c in range(len(config.img_net_classes))
        }
        dst_train = torch.utils.data.Subset(
            dst_train,
            np.squeeze(np.argwhere(np.isin(dst_train.targets, config.img_net_classes))),
        )
        loader_train_dict = {
            c: torch.utils.data.DataLoader(
                dst_train_dict[c], batch_size=batch_size, shuffle=True, num_workers=16
            )
            for c in range(len(config.img_net_classes))
        }
        dst_test = datasets.ImageNet(data_path, split="val", transform=transform)
        dst_test = torch.utils.data.Subset(
            dst_test,
            np.squeeze(np.argwhere(np.isin(dst_test.targets, config.img_net_classes))),
        )
        for c in range(len(config.img_net_classes)):
            dst_test.dataset.targets[
                dst_test.dataset.targets == config.img_net_classes[c]
            ] = c
            dst_train.dataset.targets[
                dst_train.dataset.targets == config.img_net_classes[c]
            ] = c
        print(dst_test.dataset)
        class_map = {x: i for i, x in enumerate(config.img_net_classes)}  
    else:
        exit('unknown dataset: %s' % dataset)

    if args.zca:
        images = []
        labels = []
        print("Train ZCA")
        for i in range(len(dst_train)):
            im, lab = dst_train[i]
            images.append(im)
            labels.append(lab)
        images = torch.stack(images, dim=0).to(args.device)
        labels = torch.tensor(labels, dtype=torch.long, device="cpu")
        zca = K.enhance.ZCAWhitening(eps=0.1, compute_inv=True)
        zca.fit(images)
        zca_images = zca(images).to("cpu")
        dst_train = TensorDataset(zca_images, labels)
        print("Finished train ZCA")

        images = []
        labels = []
        print("Test ZCA")
        for i in range(len(dst_test)):
            im, lab = dst_test[i]
            images.append(im)
            labels.append(lab)
        images = torch.stack(images, dim=0).to(args.device)
        labels = torch.tensor(labels, dtype=torch.long, device="cpu")

        zca_images = zca(images).to("cpu")
        dst_test = TensorDataset(zca_images, labels)
        print("Finished test ZCA")
        args.zca_trans = zca
    testloader = torch.utils.data.DataLoader(
        dst_test, batch_size=args.batch_train, shuffle=False, num_workers=12
    )

    return (
        channel,
        im_size,
        num_classes,
        class_names,
        mean,
        std,
        dst_train,
        dst_test,
        testloader,
        loader_train_dict,
        class_map,
        class_map_inv,
    )


class TensorDataset(Dataset):
    def __init__(self, images, labels):  # images: n x c x h x w tensor
        self.images = images.detach().float()
        self.labels = labels.detach()

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return self.images.shape[0]

class TensorFeatureDataset(Dataset):
    def __init__(self, images, features, labels, transform=None):  # images: n x c x h x w tensor
        self.images = images.detach().float()
        self.features = features.detach().float()
        self.labels = labels.detach()
        self.transform = transform

    def __getitem__(self, index):
        image=self.images[index]
        if self.transform != None:
            image = self.transform(image)
        feature=self.features[index]
        label=self.labels[index]
        return image,feature , label

    def __len__(self):
        return self.images.shape[0]



def get_default_convnet_setting():
    net_width, net_depth, net_act, net_norm, net_pooling = (
        128,
        3,
        'relu',
        'instancenorm',
        'avgpooling',
    )
    return net_width, net_depth, net_act, net_norm, net_pooling


def get_network(model, channel, num_classes, im_size=(32, 32), dist=True, args=None):
    net_width, net_depth, net_act, net_norm, net_pooling = get_default_convnet_setting()

    if model == 'MLP':
        net = MLP(channel=channel, num_classes=num_classes)
    elif model == 'ConvNet':
        net = ConvNet(
            channel=channel,
            num_classes=num_classes,
            net_width=net_width,
            net_depth=net_depth,
            net_act=net_act,
            net_norm=net_norm,
            net_pooling=net_pooling,
            im_size=im_size,
        )
    elif model == 'LeNet':
        net = LeNet(channel=channel, num_classes=num_classes)
    elif model == 'AlexNet':
        net = AlexNet(channel=channel, num_classes=num_classes)
    elif model == 'VGG11':
        net = VGG11(channel=channel, num_classes=num_classes)
    elif model == 'VGG11BN':
        net = VGG11BN(channel=channel, num_classes=num_classes)
    elif model == 'ResNet18':
        net = ResNet18(channel=channel, num_classes=num_classes)
    elif model == 'ResNet18BN':
        net = ResNet18BN(channel=channel, num_classes=num_classes)
    elif model == 'ResNet18BN_AP':
        net = ResNet18BN_AP(channel=channel, num_classes=num_classes)
    elif model == 'ResNet18_AP':
        net = ResNet18_AP(channel=channel, num_classes=num_classes)
    elif model == 'ConvNetD1':
        net = ConvNet(
            channel=channel,
            num_classes=num_classes,
            net_width=net_width,
            net_depth=1,
            net_act=net_act,
            net_norm=net_norm,
            net_pooling=net_pooling,
            im_size=im_size,
        )
    elif model == 'ConvNetD2':
        net = ConvNet(
            channel=channel,
            num_classes=num_classes,
            net_width=net_width,
            net_depth=2,
            net_act=net_act,
            net_norm=net_norm,
            net_pooling=net_pooling,
            im_size=im_size,
        )
    elif model == 'ConvNetD3':
        net = ConvNet(
            channel=channel,
            num_classes=num_classes,
            net_width=net_width,
            net_depth=3,
            net_act=net_act,
            net_norm=net_norm,
            net_pooling=net_pooling,
            im_size=im_size,
        )
    elif model == 'ConvNetD4':
        net = ConvNet(
            channel=channel,
            num_classes=num_classes,
            net_width=net_width,
            net_depth=4,
            net_act=net_act,
            net_norm=net_norm,
            net_pooling=net_pooling,
            im_size=im_size,
        )
    elif model == 'ConvNetD4BN':
        net = ConvNet(
            channel=channel,
            num_classes=num_classes,
            net_width=net_width,
            net_depth=4,
            net_act=net_act,
            net_norm='batchnorm',
            net_pooling=net_pooling,
            im_size=im_size,
        )
    elif model == 'ConvNetD5':
        net = ConvNet(
            channel=channel,
            num_classes=num_classes,
            net_width=net_width,
            net_depth=5,
            net_act=net_act,
            net_norm=net_norm,
            net_pooling=net_pooling,
            im_size=im_size,
        )
    elif model == 'ConvNetD6':
        net = ConvNet(
            channel=channel,
            num_classes=num_classes,
            net_width=net_width,
            net_depth=6,
            net_act=net_act,
            net_norm=net_norm,
            net_pooling=net_pooling,
            im_size=im_size,
        )
    elif model == 'ConvNetD7':
        net = ConvNet(
            channel=channel,
            num_classes=num_classes,
            net_width=net_width,
            net_depth=7,
            net_act=net_act,
            net_norm=net_norm,
            net_pooling=net_pooling,
            im_size=im_size,
        )
    elif model == 'ConvNetD8':
        net = ConvNet(
            channel=channel,
            num_classes=num_classes,
            net_width=net_width,
            net_depth=8,
            net_act=net_act,
            net_norm=net_norm,
            net_pooling=net_pooling,
            im_size=im_size,
        )
    elif model == 'ConvNetW32':
        net = ConvNet(
            channel=channel,
            num_classes=num_classes,
            net_width=32,
            net_depth=net_depth,
            net_act=net_act,
            net_norm=net_norm,
            net_pooling=net_pooling,
        )
    elif model == 'ConvNetW64':
        net = ConvNet(
            channel=channel,
            num_classes=num_classes,
            net_width=64,
            net_depth=net_depth,
            net_act=net_act,
            net_norm=net_norm,
            net_pooling=net_pooling,
        )
    elif model == 'ConvNetW128':
        net = ConvNet(
            channel=channel,
            num_classes=num_classes,
            net_width=128,
            net_depth=net_depth,
            net_act=net_act,
            net_norm=net_norm,
            net_pooling=net_pooling,
        )
    elif model == 'ConvNetW256':
        net = ConvNet(
            channel=channel,
            num_classes=num_classes,
            net_width=256,
            net_depth=net_depth,
            net_act=net_act,
            net_norm=net_norm,
            net_pooling=net_pooling,
        )
    elif model == 'ConvNetW512':
        net = ConvNet(
            channel=channel,
            num_classes=num_classes,
            net_width=512,
            net_depth=net_depth,
            net_act=net_act,
            net_norm=net_norm,
            net_pooling=net_pooling,
            im_size=im_size
        )
    elif model == 'ConvNetW1024':
        net = ConvNet(
            channel=channel,
            num_classes=num_classes,
            net_width=1024,
            net_depth=net_depth,
            net_act=net_act,
            net_norm=net_norm,
            net_pooling=net_pooling,
        )
    elif model == "ConvNetKIP":
        net = ConvNet(
            channel=channel,
            num_classes=num_classes,
            net_width=1024,
            net_depth=net_depth,
            net_act=net_act,
            net_norm="none",
            net_pooling=net_pooling,
        )
    elif model == 'ConvNetAS':
        net = ConvNet(
            channel=channel,
            num_classes=num_classes,
            net_width=net_width,
            net_depth=net_depth,
            net_act='sigmoid',
            net_norm=net_norm,
            net_pooling=net_pooling,
        )
    elif model == 'ConvNetAR':
        net = ConvNet(
            channel=channel,
            num_classes=num_classes,
            net_width=net_width,
            net_depth=net_depth,
            net_act='relu',
            net_norm=net_norm,
            net_pooling=net_pooling,
        )
    elif model == 'ConvNetAL':
        net = ConvNet(
            channel=channel,
            num_classes=num_classes,
            net_width=net_width,
            net_depth=net_depth,
            net_act='leakyrelu',
            net_norm=net_norm,
            net_pooling=net_pooling,
        )
    elif model == 'ConvNetNN':
        net = ConvNet(
            channel=channel,
            num_classes=num_classes,
            net_width=net_width,
            net_depth=net_depth,
            net_act=net_act,
            net_norm='none',
            net_pooling=net_pooling,
        )
    elif model == 'ConvNetBN':
        net = ConvNet(
            channel=channel,
            num_classes=num_classes,
            net_width=net_width,
            net_depth=net_depth,
            net_act=net_act,
            net_norm='batchnorm',
            net_pooling=net_pooling,
        )
    elif model == 'ConvNetLN':
        net = ConvNet(
            channel=channel,
            num_classes=num_classes,
            net_width=net_width,
            net_depth=net_depth,
            net_act=net_act,
            net_norm='layernorm',
            net_pooling=net_pooling,
        )
    elif model == 'ConvNetIN':
        net = ConvNet(
            channel=channel,
            num_classes=num_classes,
            net_width=net_width,
            net_depth=net_depth,
            net_act=net_act,
            net_norm='instancenorm',
            net_pooling=net_pooling,
        )
    elif model == 'ConvNetGN':
        net = ConvNet(
            channel=channel,
            num_classes=num_classes,
            net_width=net_width,
            net_depth=net_depth,
            net_act=net_act,
            net_norm='groupnorm',
            net_pooling=net_pooling,
        )
    elif model == 'ConvNetNP':
        net = ConvNet(
            channel=channel,
            num_classes=num_classes,
            net_width=net_width,
            net_depth=net_depth,
            net_act=net_act,
            net_norm=net_norm,
            net_pooling='none',
        )
    elif model == 'ConvNetMP':
        net = ConvNet(
            channel=channel,
            num_classes=num_classes,
            net_width=net_width,
            net_depth=net_depth,
            net_act=net_act,
            net_norm=net_norm,
            net_pooling='maxpooling',
        )
    elif model == 'ConvNetAP':
        net = ConvNet(
            channel=channel,
            num_classes=num_classes,
            net_width=net_width,
            net_depth=net_depth,
            net_act=net_act,
            net_norm=net_norm,
            net_pooling='avgpooling',
        )
    elif model == 'ConvNet_L3':
        net = ConvNet_feature(
            channel=channel,
            num_classes=num_classes,
            net_width=net_width,
            net_depth=net_depth,
            feature_depth=3,
            net_act=net_act,
            net_norm=net_norm,
            net_pooling=net_pooling,
            im_size=im_size,
        )
    elif model == 'ConvNetNN_L3':
        net = ConvNet_feature(
            channel=channel,
            num_classes=num_classes,
            net_width=net_width,
            net_depth=net_depth,
            feature_depth=3,
            net_act=net_act,
            net_norm='none',
            net_pooling=net_pooling,
            im_size=im_size,
        )
    elif model == 'ConvNetBN_L3':
        net = ConvNet_feature(
            channel=channel,
            num_classes=num_classes,
            net_width=net_width,
            net_depth=net_depth,
            feature_depth=3,
            net_act=net_act,
            net_norm='batchnorm',
            net_pooling=net_pooling,
            im_size=im_size,
        )
    elif model == 'ConvNetLN_L3':
        net = ConvNet_feature(
            channel=channel,
            num_classes=num_classes,
            net_width=net_width,
            net_depth=net_depth,
            feature_depth=3,
            net_act=net_act,
            net_norm='layernorm',
            net_pooling=net_pooling,
            im_size=im_size,
        )
    elif model == 'ConvNetIN_L3':
        net = ConvNet_feature(
            channel=channel,
            num_classes=num_classes,
            net_width=net_width,
            net_depth=net_depth,
            feature_depth=3,
            net_act=net_act,
            net_norm='instancenorm',
            net_pooling=net_pooling,
            im_size=im_size,
        )
    elif model == 'ConvNetGN_L3':
        net = ConvNet_feature(
            channel=channel,
            num_classes=num_classes,
            net_width=net_width,
            net_depth=net_depth,
            feature_depth=3,
            net_act=net_act,
            net_norm='groupnorm',
            net_pooling=net_pooling,
            im_size=im_size,
        )
    elif model == 'ConvNetD4_L4':
        net = ConvNet_feature(
            channel=channel,
            num_classes=num_classes,
            net_width=net_width,
            net_depth=4,
            feature_depth=4,
            net_act=net_act,
            net_norm=net_norm,
            net_pooling=net_pooling,
            im_size=im_size,
        )
    elif model == 'ConvNetD4BN_L4':
        net = ConvNet_feature(
            channel=channel,
            num_classes=num_classes,
            net_width=net_width,
            net_depth=4,
            feature_depth=4,
            net_act=net_act,
            net_norm='batchnorm',
            net_pooling=net_pooling,
            im_size=im_size,
        )
    elif model == 'ConvNetW512_L3':
        net = ConvNet_feature(
            channel=channel,
            num_classes=num_classes,
            net_width=512,
            net_depth=net_depth,
            feature_depth=3,
            net_act=net_act,
            net_norm=net_norm,
            net_pooling=net_pooling,
            im_size=im_size,
        )
    elif model == 'ConvNetW256_L3':
        net = ConvNet_feature(
            channel=channel,
            num_classes=num_classes,
            net_width=256,
            net_depth=net_depth,
            feature_depth=3,
            net_act=net_act,
            net_norm=net_norm,
            net_pooling=net_pooling,
            im_size=im_size,
        )
    elif model == 'ConvNetD4W512':
        net = ConvNet(
            channel=channel,
            num_classes=num_classes,
            net_width=512,
            net_depth=4,
            net_act=net_act,
            net_norm=net_norm,
            net_pooling=net_pooling,
            im_size=im_size,
        )
    elif model == 'ConvNetD4W256_L4':
        net = ConvNet_feature(
            channel=channel,
            num_classes=num_classes,
            net_width=256,
            net_depth=net_depth,
            feature_depth=4,
            net_act=net_act,
            net_norm=net_norm,
            net_pooling=net_pooling,
            im_size=im_size,
        )
    elif model == 'ConvNetD4W512':
        net = ConvNet(
            channel=channel,
            num_classes=num_classes,
            net_width=512,
            net_depth=4,
            net_act=net_act,
            net_norm=net_norm,
            net_pooling=net_pooling,
            im_size=im_size,
        )
    elif model == 'ConvNetD4W512_L4':
        net = ConvNet_feature(
            channel=channel,
            num_classes=num_classes,
            net_width=512,
            net_depth=net_depth,
            feature_depth=4,
            net_act=net_act,
            net_norm=net_norm,
            net_pooling=net_pooling,
            im_size=im_size,
        )
    elif model == 'ConvNetD4W256':
        net = ConvNet(
            channel=channel,
            num_classes=num_classes,
            net_width=256,
            net_depth=4,
            net_act=net_act,
            net_norm=net_norm,
            net_pooling=net_pooling,
            im_size=im_size,
        )
    elif model == 'VGG11_L5':
        net= VGG11_feature(feature_depth=5, channel=channel, num_classes=num_classes)
    elif model == 'VGG11BN_L5':
        net= VGG11BN_feature(feature_depth=5, channel=channel, num_classes=num_classes)
    elif model == 'ResNet18_Layered':
        net = ResNet18_Layered(channel=channel, num_classes=num_classes)
    elif model == 'ResNet18BN_Layered':
        net = ResNet18BN_Layered(channel=channel, num_classes=num_classes)
    elif model == 'ResNet18ImageNet':
        net = ResNet18ImageNet(channel=channel, num_classes=num_classes)
    elif model == 'ResNet18BNImageNet':
        net = ResNet18BNImageNet(channel=channel, num_classes=num_classes)
    elif model == 'ResNet18ImageNet_L4':
        net = ResNet18ImageNet_L4(channel=channel, num_classes=num_classes)
    elif model == 'ResNet18BNImageNet_L4':
        net = ResNet18BNImageNet_L4(channel=channel, num_classes=num_classes)
    elif model == 'VGG11ImageNet_L6':
        net = VGG11ImageNet_feature(feature_depth=6, channel=channel, num_classes=num_classes)
    elif model == 'VGG11BNImageNet_L6':
        net = VGG11BNImageNet_feature(feature_depth=6, channel=channel, num_classes=num_classes)
    elif model == 'VGG11ImageNet':
        net = VGG11ImageNet(channel=channel, num_classes=num_classes)
    elif model == 'VGG11BNImageNet':
        net = VGG11BNImageNet(channel=channel, num_classes=num_classes)
    else:
        net = None
        exit('DC error: unknown model:'+str(model))
    if dist:
        gpu_num = torch.cuda.device_count()
        if gpu_num > 0:
            device = 'cuda'
            if gpu_num > 1:
                net = nn.DataParallel(net)
        else:
            device = 'cpu'
        net = net.to(device)

    return net


def get_time():
    return str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))


def epoch(mode, dataloader, net, optimizer, criterion, args, aug, texture=False):
    loss_avg, acc_avg, num_exp = 0, 0, 0
    net = net.to(args.device)

    # if args.dataset == "ImageNet":
    #     class_map = {x: i for i, x in enumerate(config.img_net_classes)}

    if mode == 'train':
        net.train()
    else:
        net.eval()

    for i_batch, datum in enumerate(dataloader):
        # print("batch: {}".format(i_batch))
        img = datum[0].float().to(args.device)
        lab = datum[1].long().to(args.device)

        if aug:
            if args.dsa:
                img = DiffAugment(img, args.dsa_strategy, param=args.dsa_param)
            else:
                img = augment(img, args.dc_aug_param, device=args.device)



        n_b = lab.shape[0]

        output = net(img)
        loss = criterion(output, lab)

        acc = np.sum(
            np.equal(
                np.argmax(output.cpu().data.numpy(), axis=-1), lab.cpu().data.numpy()
            )
        )

        loss_avg += loss.item() * n_b
        acc_avg += acc
        num_exp += n_b

        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    loss_avg /= num_exp
    acc_avg /= num_exp

    return loss_avg, acc_avg

def epoch_for_label(mode, dataloader, net, optimizer, criterion, args, aug, texture=False):
    loss_avg, acc_avg, num_exp = 0, 0, 0
    net = net.to(args.device)

    if args.dataset == "ImageNet":
        class_map = {x: i for i, x in enumerate(config.img_net_classes)}

    if mode == 'train':
        net.train()
    else:
        net.eval()

    for i_batch, datum in enumerate(dataloader):
        img = datum[0].float().to(args.device)
        img = DiffAugment(img, args.dsa_strategy, param=args.dsa_param).to(args.device)
        output = net(img)
        if mode == "train":
            lab = datum[1].float().to(args.device)
            loss = criterion(output, lab)
        else:
            lab = datum[1].long().to(args.device)
            onehot = torch.nn.functional.one_hot(lab, 100).float().to(args.device)
            loss = criterion(output, onehot)
            acc = np.sum(
            np.equal(
                np.argmax(output.cpu().data.numpy(), axis=-1), lab.cpu().data.numpy()
            )
            )
            acc_avg += acc    
        n_b = lab.shape[0]
        loss_avg += loss.item() * n_b
        num_exp += n_b

        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    loss_avg /= num_exp
    acc_avg /= num_exp

    return loss_avg, acc_avg


def evaluate_synset(
    it_eval,
    net,
    images_train,
    labels_train,
    testloader,
    args,
    return_loss=False,
    texture=False,
):
    net = net.to(args.device)
    images_train = images_train.to(args.device)
    labels_train = labels_train.to(args.device)
    lr = float(args.lr_net)
    Epoch = int(args.epoch_eval_train)
    lr_schedule = [Epoch // 2 + 1]
    optimizer = torch.optim.SGD(
        net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005
    )

    criterion = nn.CrossEntropyLoss().to(args.device)

    dst_train = TensorDataset(images_train, labels_train)
    trainloader = torch.utils.data.DataLoader(
        dst_train, batch_size=args.batch_train, shuffle=True, num_workers=args.num_workers
    )

    start = time.time()
    acc_train_list = []
    loss_train_list = []

    for ep in range(Epoch + 1):
        loss_train, acc_train = epoch(
            'train',
            trainloader,
            net,
            optimizer,
            criterion,
            args,
            aug=True,
            texture=texture,
        )
        acc_train_list.append(acc_train)
        loss_train_list.append(loss_train)
        if ep == Epoch:
            with torch.no_grad():
                loss_test, acc_test = epoch(
                    'test', testloader, net, optimizer, criterion, args, aug=False
                )
        if ep in lr_schedule:
            lr *= 0.1
            optimizer = torch.optim.SGD(
                net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005
            )

    time_train = time.time() - start

    print(
        '%s Evaluate_%02d: epoch = %04d train time = %d s train loss = %.6f train acc = %.4f, test acc = %.4f'
        % (get_time(), it_eval, Epoch, int(time_train), loss_train, acc_train, acc_test)
    )

    if return_loss:
        return net, acc_train_list, acc_test, loss_train_list, loss_test
    else:
        return net, acc_train_list, acc_test


def epoch_for_feature(
    mode, dataloader, net, optimizer, criterion, args, aug, feature_criterion
):
    loss_avg, acc_avg, num_exp = 0, 0, 0
    net = net.to(args.device)

    if mode == 'train':
        net.train()
    else:
        net.eval()

    for i_batch, datum in enumerate(dataloader):
        if mode == 'train':
            img = datum[0].float().to(args.device)
            ftr = datum[1].float().to(args.device)
            if args.dataset=='ImageNet' or args.distilled_data_dir.startswith('result_FrePo'):
                lab = datum[2].to(args.device)
            else:
                lab = datum[2].long().to(args.device)
            if aug:
                if args.dsa:
                    img = DiffAugment(img, args.dsa_strategy, param=args.dsa_param)
                else:
                    img = augment(img, args.dc_aug_param, device=args.device)

            n_b = lab.shape[0]

            img_ftr, img_output = net(img)

            loss=None
            if args.loss_mode=='task':
                loss = criterion(img_output, lab)
            else:
                _,ftr_output = net(feature=ftr)
                if args.loss_mode=='front_rear_task':
                    loss = args.lamda_front * feature_criterion(img_ftr, ftr) + args.lamda_rear * criterion(ftr_output, lab) + criterion(img_output, lab) # 前后全
                elif args.loss_mode=='front_task':
                    loss = args.lamda_front * criterion(img_ftr, ftr) + criterion(img_output, lab) # 前全
                elif args.loss_mode=='rear_task':
                    loss = args.lamda_rear * criterion(ftr_output, lab) + criterion(img_output, lab) # 后全
                elif args.loss_mode=='front_rear':
                    loss = args.lamda_front * feature_criterion(img_ftr, ftr) + criterion(ftr_output, lab) # 前后

            if args.dataset=='ImageNet' and mode == 'train':
                acc = np.sum(np.equal(np.argmax(img_output.cpu().data.numpy(), axis=-1), np.argmax(lab.cpu().data.numpy(), axis=-1)))
            else:
                acc = np.sum(np.equal(np.argmax(img_output.cpu().data.numpy(), axis=-1), np.argmax(lab.cpu().data.numpy(), axis=-1)))

            loss_avg += loss.item() * n_b
            acc_avg += acc
            num_exp += n_b
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            img = datum[0].float().to(args.device)
            lab = datum[1].long().to(args.device)
            if aug:
                if args.dsa:
                    img = DiffAugment(img, args.dsa_strategy, param=args.dsa_param)
                else:
                    img = augment(img, args.dc_aug_param, device=args.device)

            n_b = lab.shape[0]

            _, img_output = net(img)
            loss = criterion(img_output, lab)
            acc = np.sum(
                np.equal(
                    np.argmax(img_output.cpu().data.numpy(), axis=-1),
                    lab.cpu().data.numpy(),
                )
            )

            loss_avg += loss.item() * n_b
            acc_avg += acc
            num_exp += n_b
    loss_avg /= num_exp
    acc_avg /= num_exp

    return loss_avg, acc_avg

def evaluate_feature_synset(
    it_eval,
    net,
    images_train,
    feature_train,
    labels_train,
    testloader,
    args,
    return_loss=False,
):
    net = net.to(args.device)
    images_train = images_train.to(args.device)
    feature_train = feature_train.to(args.device)
    labels_train = labels_train.to(args.device)
    lr = float(args.lr_net)
    Epoch = int(args.epoch_eval_train)
    lr_schedule = [Epoch // 2 + 1]
    optimizer = torch.optim.SGD(
        net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005
    )

    criterion = nn.CrossEntropyLoss().to(args.device)
    if args.feature_loss_mode=='CE':
        feature_criterion = nn.CrossEntropyLoss().to(args.device)
    elif args.feature_loss_mode=='L2':
        feature_criterion = nn.MSELoss().to(args.device)
    elif args.feature_loss_mode=='L1':
        feature_criterion = nn.L1Loss().to(args.device)
    elif args.feature_loss_mode=='COS':
        def cos_dis(feature, feature_label):
            dis = torch.tensor(0.0).to(args.device)
            feature_vec = []
            feature_label_vec = []
            for index in range(len(feature)):
                feature_vec.append(feature[index].reshape((-1)))
                feature_label_vec.append(feature_label[index].reshape((-1)))
            feature_vec = torch.cat(feature_vec, dim=0)
            feature_label_vec = torch.cat(feature_label_vec, dim=0)
            dis = 1 - torch.sum(feature_vec * feature_label_vec, dim=-1) / (torch.norm(feature_vec, dim=-1) * torch.norm(feature_label_vec, dim=-1) + 0.000001)
            return dis
        feature_criterion = cos_dis
        # feature_criterion = nn.CosineSimilarity().to(args.device)
    else:
        raise NotImplementedError
    
    if args.transforms_normalize_syn:
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        print('use transforms.Normalize for syn')
        transform = transforms.Compose(
                [transforms.Normalize(mean=mean, std=std)]
            )
        dst_train = TensorFeatureDataset(images_train, feature_train, labels_train,transform)
    else:
        dst_train = TensorFeatureDataset(images_train, feature_train, labels_train)
    trainloader = torch.utils.data.DataLoader(
        dst_train, batch_size=args.batch_train, shuffle=True, num_workers=args.num_workers
    )
    

    start = time.time()
    acc_train_list = []
    loss_train_list = []
    
    for ep in range(Epoch + 1):
        if args.dataset=='ImageNet' and ep%200==0:
            print(f'epoch={ep}')
        loss_train, acc_train = epoch_for_feature(
            'train',
            trainloader,
            net,
            optimizer,
            criterion,
            args,
            aug=True,
            feature_criterion=feature_criterion,
        )
        acc_train_list.append(acc_train)
        loss_train_list.append(loss_train)
        if ep == Epoch:
            with torch.no_grad():
                loss_test, acc_test = epoch_for_feature(
                    'test', testloader, net, optimizer, criterion, args, aug=False, feature_criterion=feature_criterion
                )
        if ep in lr_schedule:
            lr *= 0.1
            optimizer = torch.optim.SGD(
                net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005
            )
        # scheduler.step()

    time_train = time.time() - start
    print(
        '\n%s Evaluate_%02d: epoch = %04d train time = %d s train loss = %.6f train acc = %.4f, test acc = %.4f'
        % (get_time(), it_eval, Epoch, int(time_train), loss_train, acc_train, acc_test)
    )

    if return_loss:
        return net, acc_train_list, acc_test, loss_train_list, loss_test
    else:
        return net, acc_train_list, acc_test


def augment(images, dc_aug_param, device):
    # This can be sped up in the future.
    if dc_aug_param != None and dc_aug_param['strategy'] != 'none':
        scale = dc_aug_param['scale']
        crop = dc_aug_param['crop']
        rotate = dc_aug_param['rotate']
        noise = dc_aug_param['noise']
        strategy = dc_aug_param['strategy']

        shape = images.shape
        mean = []
        for c in range(shape[1]):
            mean.append(float(torch.mean(images[:, c])))

        def cropfun(i):
            im_ = torch.zeros(
                shape[1],
                shape[2] + crop * 2,
                shape[3] + crop * 2,
                dtype=torch.float,
                device=device,
            )
            for c in range(shape[1]):
                im_[c] = mean[c]
            im_[:, crop : crop + shape[2], crop : crop + shape[3]] = images[i]
            r, c = (
                np.random.permutation(crop * 2)[0],
                np.random.permutation(crop * 2)[0],
            )
            images[i] = im_[:, r : r + shape[2], c : c + shape[3]]

        def scalefun(i):
            h = int((np.random.uniform(1 - scale, 1 + scale)) * shape[2])
            w = int((np.random.uniform(1 - scale, 1 + scale)) * shape[2])
            tmp = F.interpolate(
                images[i : i + 1],
                [h, w],
            )[0]
            mhw = max(h, w, shape[2], shape[3])
            im_ = torch.zeros(shape[1], mhw, mhw, dtype=torch.float, device=device)
            r = int((mhw - h) / 2)
            c = int((mhw - w) / 2)
            im_[:, r : r + h, c : c + w] = tmp
            r = int((mhw - shape[2]) / 2)
            c = int((mhw - shape[3]) / 2)
            images[i] = im_[:, r : r + shape[2], c : c + shape[3]]

        def rotatefun(i):
            im_ = scipyrotate(
                images[i].cpu().data.numpy(),
                angle=np.random.randint(-rotate, rotate),
                axes=(-2, -1),
                cval=np.mean(mean),
            )
            r = int((im_.shape[-2] - shape[-2]) / 2)
            c = int((im_.shape[-1] - shape[-1]) / 2)
            images[i] = torch.tensor(
                im_[:, r : r + shape[-2], c : c + shape[-1]],
                dtype=torch.float,
                device=device,
            )

        def noisefun(i):
            images[i] = images[i] + noise * torch.randn(
                shape[1:], dtype=torch.float, device=device
            )

        augs = strategy.split('_')

        for i in range(shape[0]):
            choice = np.random.permutation(augs)[
                0
            ]  # randomly implement one augmentation
            if choice == 'crop':
                cropfun(i)
            elif choice == 'scale':
                scalefun(i)
            elif choice == 'rotate':
                rotatefun(i)
            elif choice == 'noise':
                noisefun(i)

    return images


def get_daparam(dataset, model, model_eval, ipc):
    # We find that augmentation doesn't always benefit the performance.
    # So we do augmentation for some of the settings.

    dc_aug_param = dict()
    dc_aug_param['crop'] = 4
    dc_aug_param['scale'] = 0.2
    dc_aug_param['rotate'] = 45
    dc_aug_param['noise'] = 0.001
    dc_aug_param['strategy'] = 'none'

    if dataset == 'MNIST':
        dc_aug_param['strategy'] = 'crop_scale_rotate'

    if model_eval in [
        'ConvNetBN'
    ]:  # Data augmentation makes model training with Batch Norm layer easier.
        dc_aug_param['strategy'] = 'crop_noise'

    return dc_aug_param


def get_eval_pool(eval_mode, model_eval):
    if eval_mode == 'itself': 
        model_eval_pool = [model_eval]
    elif eval_mode == 'cross': 
        model_eval_pool = [
            'ConvNet_L3',
            'ConvNetBN_L3',
            'ResNet18_Layered',
            'ResNet18BN_Layered',
            'VGG11_L5',
            'VGG11BN_L5'
        ]
    elif eval_mode == 'cross_512':
        model_eval_pool = [
            'ResNet18_Layered',
            'ResNet18BN_Layered',
            'VGG11_L5',
            'VGG11BN_L5'
        ]
    elif eval_mode == 'cross_128':
        model_eval_pool = [
            'ConvNet_L3',
            'ConvNetBN_L3',
        ]
    elif eval_mode == 'cross_imagenet_convnet': 
        model_eval_pool = [
            'ConvNetD4_L4',
            'ConvNetD4BN_L4',
        ]
    elif eval_mode == 'cross_imagenet_resnet': 
        model_eval_pool = [
            'ResNet18ImageNet_L4',
            'ResNet18BNImageNet_L4',
        ]
    elif eval_mode == 'cross_imagenet_vgg': 
        model_eval_pool = [
            'VGG11ImageNet_L6',
            'VGG11BNImageNet_L6',
        ]
    elif eval_mode == 'cross_imagenet': 
        model_eval_pool = [
            'ConvNetD4_L4',
            'ConvNetD4BN_L4',
            'ResNet18ImageNet_L4',
            'ResNet18BNImageNet_L4',
            'VGG11ImageNet_L6',
            'VGG11BNImageNet_L6',
        ]
    
    # eval_model for MTT
    elif eval_mode == 'M': # multiple architectures
        # model_eval_pool = ['MLP', 'ConvNet', 'AlexNet', 'VGG11', 'ResNet18', 'LeNet']
        model_eval_pool = ['ConvNet', 'AlexNet', 'VGG11', 'ResNet18_AP', 'ResNet18']
        # model_eval_pool = ['MLP', 'ConvNet', 'AlexNet', 'VGG11', 'ResNet18']
    elif eval_mode == 'W': # ablation study on network width
        model_eval_pool = ['ConvNetW32', 'ConvNetW64', 'ConvNetW128', 'ConvNetW256']
    elif eval_mode == 'D': # ablation study on network depth
        model_eval_pool = ['ConvNetD1', 'ConvNetD2', 'ConvNetD3', 'ConvNetD4']
    elif eval_mode == 'A': # ablation study on network activation function
        model_eval_pool = ['ConvNetAS', 'ConvNetAR', 'ConvNetAL']
    elif eval_mode == 'P': # ablation study on network pooling layer
        model_eval_pool = ['ConvNetNP', 'ConvNetMP', 'ConvNetAP']
    elif eval_mode == 'N': # ablation study on network normalization layer
        model_eval_pool = ['ConvNetNN', 'ConvNetBN', 'ConvNetLN', 'ConvNetIN', 'ConvNetGN']
    elif eval_mode == 'S': # itself
        model_eval_pool = [model_eval[:model_eval.index('BN')]] if 'BN' in model_eval else [model_eval]
    elif eval_mode == 'C':
        model_eval_pool = [model_eval, 'ConvNet']
    else:
        model_eval_pool = [model_eval]
    return model_eval_pool


class ParamDiffAug:
    def __init__(self):
        self.aug_mode = 'S'  #'multiple or single'
        self.prob_flip = 0.5
        self.ratio_scale = 1.2
        self.ratio_rotate = 15.0
        self.ratio_crop_pad = 0.125
        self.ratio_cutout = 0.5  # the size would be 0.5x0.5
        self.ratio_noise = 0.05
        self.brightness = 1.0
        self.saturation = 2.0
        self.contrast = 0.5


def set_seed_DiffAug(param):
    if param.latestseed == -1:
        return
    else:
        torch.random.manual_seed(param.latestseed)
        param.latestseed += 1


def DiffAugment(x, strategy='', seed=-1, param=None):
    if seed == -1:
        param.batchmode = False
    else:
        param.batchmode = True

    param.latestseed = seed

    if strategy == 'None' or strategy == 'none':
        return x

    if strategy:
        if param.aug_mode == 'M':  # original
            for p in strategy.split('_'):
                for f in AUGMENT_FNS[p]:
                    x = f(x, param)
        elif param.aug_mode == 'S':
            pbties = strategy.split('_')
            set_seed_DiffAug(param)
            p = pbties[torch.randint(0, len(pbties), size=(1,)).item()]
            for f in AUGMENT_FNS[p]:
                x = f(x, param)
        else:
            exit('Error ZH: unknown augmentation mode.')
        x = x.contiguous()
    return x


# We implement the following differentiable augmentation strategies based on the code provided in https://github.com/mit-han-lab/data-efficient-gans.
def rand_scale(x, param):
    # x>1, max scale
    # sx, sy: (0, +oo), 1: orignial size, 0.5: enlarge 2 times
    ratio = param.ratio_scale
    set_seed_DiffAug(param)
    sx = torch.rand(x.shape[0]) * (ratio - 1.0 / ratio) + 1.0 / ratio
    set_seed_DiffAug(param)
    sy = torch.rand(x.shape[0]) * (ratio - 1.0 / ratio) + 1.0 / ratio
    theta = [
        [
            [sx[i], 0, 0],
            [0, sy[i], 0],
        ]
        for i in range(x.shape[0])
    ]
    theta = torch.tensor(theta, dtype=torch.float)
    if param.batchmode:  # batch-wise:
        theta[:] = theta[0]
    grid = F.affine_grid(theta, x.shape, align_corners=True).to(x.device)
    x = F.grid_sample(x, grid, align_corners=True)
    return x


def rand_rotate(x, param):  # [-180, 180], 90: anticlockwise 90 degree
    ratio = param.ratio_rotate
    set_seed_DiffAug(param)
    theta = (torch.rand(x.shape[0]) - 0.5) * 2 * ratio / 180 * float(np.pi)
    theta = [
        [
            [torch.cos(theta[i]), torch.sin(-theta[i]), 0],
            [torch.sin(theta[i]), torch.cos(theta[i]), 0],
        ]
        for i in range(x.shape[0])
    ]
    theta = torch.tensor(theta, dtype=torch.float)
    if param.batchmode:  # batch-wise:
        theta[:] = theta[0]
    grid = F.affine_grid(theta, x.shape, align_corners=True).to(x.device)
    x = F.grid_sample(x, grid, align_corners=True)
    return x


def rand_flip(x, param):
    prob = param.prob_flip
    set_seed_DiffAug(param)
    randf = torch.rand(x.size(0), 1, 1, 1, device=x.device)
    if param.batchmode:  # batch-wise:
        randf[:] = randf[0]
    return torch.where(randf < prob, x.flip(3), x)


def rand_brightness(x, param):
    ratio = param.brightness
    set_seed_DiffAug(param)
    randb = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    if param.batchmode:  # batch-wise:
        randb[:] = randb[0]
    x = x + (randb - 0.5) * ratio
    return x


def rand_saturation(x, param):
    ratio = param.saturation
    x_mean = x.mean(dim=1, keepdim=True)
    set_seed_DiffAug(param)
    rands = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    if param.batchmode:  # batch-wise:
        rands[:] = rands[0]
    x = (x - x_mean) * (rands * ratio) + x_mean
    return x


def rand_contrast(x, param):
    ratio = param.contrast
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    set_seed_DiffAug(param)
    randc = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    if param.batchmode:  # batch-wise:
        randc[:] = randc[0]
    x = (x - x_mean) * (randc + ratio) + x_mean
    return x


def rand_crop(x, param):
    # The image is padded on its surrounding and then cropped.
    ratio = param.ratio_crop_pad
    shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    set_seed_DiffAug(param)
    translation_x = torch.randint(
        -shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device
    )
    set_seed_DiffAug(param)
    translation_y = torch.randint(
        -shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device
    )
    if param.batchmode:  # batch-wise:
        translation_x[:] = translation_x[0]
        translation_y[:] = translation_y[0]
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(2), dtype=torch.long, device=x.device),
        torch.arange(x.size(3), dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    x = (
        x_pad.permute(0, 2, 3, 1)
        .contiguous()[grid_batch, grid_x, grid_y]
        .permute(0, 3, 1, 2)
    )
    return x


def rand_cutout(x, param):
    ratio = param.ratio_cutout
    cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    set_seed_DiffAug(param)
    offset_x = torch.randint(
        0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device
    )
    set_seed_DiffAug(param)
    offset_y = torch.randint(
        0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device
    )
    if param.batchmode:  # batch-wise:
        offset_x[:] = offset_x[0]
        offset_y[:] = offset_y[0]
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
        torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(
        grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1
    )
    grid_y = torch.clamp(
        grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1
    )
    mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    mask[grid_batch, grid_x, grid_y] = 0
    x = x * mask.unsqueeze(1)
    return x


AUGMENT_FNS = {
    'color': [rand_brightness, rand_saturation, rand_contrast],
    'crop': [rand_crop],
    'cutout': [rand_cutout],
    'flip': [rand_flip],
    'scale': [rand_scale],
    'rotate': [rand_rotate],
}



def seed_torch(seed=3407):    
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
