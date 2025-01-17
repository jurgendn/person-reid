import timm
import torch
import torch.nn as nn
from torchvision import models
from torch.nn import init
from torchvision.models._utils import IntermediateLayerGetter

from configs.factory import FTNetConfig
from src.models.utils import get_num_channels, pooling


######################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find("Conv") != -1:
        init.kaiming_normal_(
            m.weight.data, a=0, mode="fan_in"
        )  # For old pytorch, you may use kaiming_normal.
    elif classname.find("Linear") != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode="fan_out")
    elif classname.find("BatchNorm1d") != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
    if hasattr(m, "bias") and m.bias is not None:
        init.constant_(m.bias.data, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)


def activate_drop(m):
    classname = m.__class__.__name__
    if classname.find("Drop") != -1:
        m.p = 0.1
        m.inplace = True


# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class ClassBlock(nn.Module):
    def __init__(
        self,
        input_dim,
        class_num,
        droprate,
        relu=True,
        bnorm=True,
        linear=512,
        return_f=False,
    ):
        super(ClassBlock, self).__init__()
        self.return_f = return_f
        add_block = []
        if linear > 0:
            add_block += [nn.Linear(input_dim, linear)]
        else:
            linear = input_dim
        if bnorm:
            add_block += [nn.BatchNorm1d(linear)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate > 0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(linear, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        x = self.add_block(x)
        if self.return_f:
            f = x
            return f
        else:
            x = self.classifier(x)
            return x


# Define the ResNet50-based Model
class FTNet(nn.Module):
    def __init__(self, target_layer: str, output_layer_name: str):
        super(FTNet, self).__init__()
        model_ft = models.resnet50(weights="IMAGENET1K_V1")
        self.model = IntermediateLayerGetter(
            model=model_ft,
            return_layers={target_layer: output_layer_name},
        )
        num_channels = get_num_channels(
            backbone=self.model,
            in_channels=3,
            output_name=output_layer_name,
        )
        # avg pooling to global pooling
        self.max_avg_pooling = pooling.MaxAvgPooling()
        self.bn = nn.BatchNorm1d(num_features=2 * num_channels)

        init.normal_(self.bn.weight.data, 1.0, 0.02)
        init.constant_(self.bn.bias.data, 0.0)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.max_avg_pooling(x)
        x = x.view(x.size(0), x.size(1))
        x = self.bn(x)
        return x


# Define the swin_base_patch4_window7_224 Model
# pytorch > 1.6
class FTNet_Swin(nn.Module):
    def __init__(
        self,
        class_num: int = 77,
        droprate: float = 0.5,
        stride: int = 2,
        linear_num: int = 512,
        return_f: bool = True,
    ):
        super(FTNet_Swin, self).__init__()
        model_ft = timm.create_model(
            "swin_base_patch4_window7_224", pretrained=True, drop_path_rate=0.2
        )
        # avg pooling to global pooling
        # model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        model_ft.head = nn.Sequential()  # save memory
        self.model = model_ft
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.classifier = ClassBlock(
            1024, class_num, droprate, linear=linear_num, return_f=return_f
        )

    def forward(self, x):
        x = self.model.forward_features(x)
        # swin is update in latest timm>0.6.0, so I add the following two lines.
        x = self.avgpool(x.permute((0, 2, 1)))
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        return x


# Define the HRNet18-based Model
class FTNetHR(nn.Module):
    def __init__(
        self,
        class_num: int = 77,
        droprate: float = 0.5,
        linear_num: int = 512,
        return_f: bool = True,
    ):
        super().__init__()
        model_ft = timm.create_model("hrnet_w18", pretrained=True)
        # avg pooling to global pooling
        # model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        model_ft.classifier = nn.Sequential()  # save memory
        self.model = model_ft
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = ClassBlock(
            2048, class_num, droprate, linear=linear_num, return_f=return_f
        )

    def forward(self, x):
        x = self.model.forward_features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        return x


class PCB(nn.Module):
    def __init__(self, class_num):
        super(PCB, self).__init__()

        self.part = 6  # We cut the pool5 to 6 parts
        resnet_weights = models.ResNet50_Weights.DEFAULT
        model_ft = models.resnet50(resnet_weights)
        self.model = model_ft
        self.avgpool = nn.AdaptiveAvgPool2d((self.part, 1))
        self.dropout = nn.Dropout(p=0.5)
        # remove the final downsample
        self.model.layer4[0].downsample[0].stride = (1, 1)
        self.model.layer4[0].conv2.stride = (1, 1)
        # define 6 classifiers
        for i in range(self.part):
            name = "classifier" + str(i)
            setattr(
                self,
                name,
                ClassBlock(
                    2048, class_num, droprate=0.5, linear=256, relu=False, bnorm=True
                ),
            )

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.avgpool(x)
        x = self.dropout(x)
        part = {}
        predict = {}
        # get six part feature batchsize*2048*6
        for i in range(self.part):
            part[i] = x[:, :, i].view(x.size(0), x.size(1))
            name = "classifier" + str(i)
            c = getattr(self, name)
            predict[i] = c(part[i])

        # sum prediction
        # y = predict[0]
        # for i in range(self.part-1):
        #    y += predict[i+1]
        y = []
        for i in range(self.part):
            y.append(predict[i])
        return y


class PCBTest(nn.Module):
    def __init__(self, model):
        super(PCBTest, self).__init__()
        self.part = 6
        self.model = model.model
        self.avgpool = nn.AdaptiveAvgPool2d((self.part, 1))
        # remove the final downsample
        self.model.layer4[0].downsample[0].stride = (1, 1)
        self.model.layer4[0].conv2.stride = (1, 1)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.avgpool(x)
        y = x.view(x.size(0), x.size(1), x.size(2))
        return y


class ResNet50(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT
        resnet50 = models.resnet50(weights=weights)
        if config.MODEL.RES4_STRIDE == 1:
            resnet50.layer4[0].conv2.stride = (1, 1)
            resnet50.layer4[0].downsample[0].stride = (1, 1)
        self.base = nn.Sequential(*list(resnet50.children())[:-2])

        if config.MODEL.POOLING.NAME == "avg":
            self.globalpooling = nn.AdaptiveAvgPool2d(1)
        elif config.MODEL.POOLING.NAME == "max":
            self.globalpooling = nn.AdaptiveMaxPool2d(1)
        elif config.MODEL.POOLING.NAME == "gem":
            self.globalpooling = pooling.GeMPooling(p=config.MODEL.POOLING.P)
        elif config.MODEL.POOLING.NAME == "maxavg":
            self.globalpooling = pooling.MaxAvgPooling()
        else:
            raise KeyError("Invalid pooling: '{}'".format(config.MODEL.POOLING.NAME))

        self.bn = nn.BatchNorm1d(config.MODEL.FEATURE_DIM)
        init.normal_(self.bn.weight.data, 1.0, 0.02)
        init.constant_(self.bn.bias.data, 0.0)

    def forward(self, x):
        x = self.base(x)
        x = self.globalpooling(x)
        x = x.view(x.size(0), -1)
        f = self.bn(x)

        return f


"""
# debug model structure
# Run this code with:
python model.py
"""
if __name__ == "__main__":
    # Here I left a simple forward function.
    # Test the model, before you train it.
    net = FTNet(751)
    # net = FTNet_swin(751, stride=1)
    net.classifier = nn.Sequential()
    print(net)
    input = torch.FloatTensor(8, 3, 224, 224)
    output = net(input)
