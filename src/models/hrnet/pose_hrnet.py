"""
Source: https://github.com/ChenyanWu/MEBOW/tree/master

Raises:
    ValueError: _description_
    ValueError: _description_
    ValueError: _description_
    ValueError: _description_

Returns:
    _type_: _description_

Yields:
    _type_: _description_
"""

from __future__ import absolute_import, division, print_function

import logging
import os
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.factory import HRNetConfig, HRNetStageConfig

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
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


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class HighResolutionModule(nn.Module):
    def __init__(
        self,
        num_branches,
        blocks,
        num_blocks,
        num_inchannels,
        num_channels,
        fuse_method,
        multi_scale_output=True,
    ):
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels
        )

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels
        )
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(True)

    def _check_branches(
        self, num_branches, blocks, num_blocks, num_inchannels, num_channels
    ):
        if num_branches != len(num_blocks):
            error_msg = "NUM_BRANCHES({}) <> NUM_BLOCKS({})".format(
                num_branches, len(num_blocks)
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = "NUM_BRANCHES({}) <> NUM_CHANNELS({})".format(
                num_branches, len(num_channels)
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = "NUM_BRANCHES({}) <> NUM_INCHANNELS({})".format(
                num_branches, len(num_inchannels)
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels, stride=1):
        downsample = None
        if (
            stride != 1
            or self.num_inchannels[branch_index]
            != num_channels[branch_index] * block.expansion
        ):
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index] * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(
                    num_channels[branch_index] * block.expansion, momentum=BN_MOMENTUM
                ),
            )

        layers = []
        layers.append(
            block(
                self.num_inchannels[branch_index],
                num_channels[branch_index],
                stride,
                downsample,
            )
        )
        self.num_inchannels[branch_index] = num_channels[branch_index] * block.expansion
        for _ in range(1, num_blocks[branch_index]):
            layers.append(
                block(self.num_inchannels[branch_index], num_channels[branch_index])
            )

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self) -> nn.ModuleList:
        """
        Creates the fuse layers for the multi-branch architecture.

        This function constructs the necessary layers to fuse the outputs
        from different branches of a multi-scale feature pyramid. It handles
        upsampling and downsampling between branches based on the specified
        architecture.

        Returns:
            nn.ModuleList: A list of fuse layers as ModuleLists, where each
            ModuleList contains the appropriate operations (upsample,
            downsample, or None) for the respective branches.
        """

        # Helper function to create a Conv2d -> BatchNorm -> (ReLU) block.
        def _conv_bn_relu(
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int,
            padding: int,
            use_relu: bool = True,
        ) -> nn.Sequential:
            layers = [
                nn.Conv2d(
                    in_channels, out_channels, kernel_size, stride, padding, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            ]
            if use_relu:
                layers.append(nn.ReLU(inplace=True))
            return nn.Sequential(*layers)

        # Helper function to create an upsampling block.
        def _create_upsample_block(
            in_channels: int, out_channels: int, scale_factor: float
        ) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.Upsample(scale_factor=scale_factor, mode="nearest"),
            )

        # Helper function to create downsampling blocks using 3x3 Conv2d.
        def _create_downsample_blocks(
            in_channels: int, out_channels: int, num_downsamples: int
        ) -> nn.Sequential:
            return nn.Sequential(
                *[
                    _conv_bn_relu(
                        in_channels=in_channels,
                        out_channels=out_channels
                        if k == num_downsamples - 1
                        else in_channels,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        use_relu=(k != num_downsamples - 1),
                    )
                    for k in range(num_downsamples)
                ]
            )

        # Helper function to create the appropriate fuse layer (upsample, downsample, or None).
        def _create_fuse_layer(
            i: int, j: int, num_inchannels: List[int]
        ) -> nn.Sequential | None:
            if j > i:
                # Upsample from higher resolution (j) to lower resolution (i)
                return _create_upsample_block(
                    num_inchannels[j], num_inchannels[i], scale_factor=2 ** (j - i)
                )
            elif j == i:
                # No operation needed when i == j
                return None
            else:
                # Downsample from lower resolution (j) to higher resolution (i)
                return _create_downsample_blocks(
                    num_inchannels[j], num_inchannels[i], num_downsamples=i - j
                )

        # Main logic of the fuse layers
        if self.num_branches == 1:
            return nn.ModuleList()

        # Generate the fuse layers using functional-style list comprehensions.
        fuse_layers = [
            nn.ModuleList(
                modules=[
                    _create_fuse_layer(i, j, self.num_inchannels)
                    for j in range(self.num_branches)
                ]
            )
            for i in range(self.num_branches if self.multi_scale_output else 1)
        ]

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []

        for i, layer in enumerate(self.fuse_layers):
            y = x[0] if i == 0 else layer[0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + layer[j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


blocks_dict = {"BASIC": BasicBlock, "BOTTLENECK": Bottleneck}


class PoseHighResolutionNet(nn.Module):
    def __init__(self, cfg: HRNetConfig, **kwargs):
        self.inplanes = 64
        super(PoseHighResolutionNet, self).__init__()
        self.num_joints = cfg.model.num_joints
        self.use_featuremap = cfg.model.use_featuremap
        self.return_hoe = cfg.model.return_hoe
        # stem net
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(Bottleneck, 64, 4)

        self.stage2_cfg = cfg.model.extra.stage2
        num_channels = self.stage2_cfg.num_channels
        block = blocks_dict[self.stage2_cfg.block]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition1 = self._make_transition_layer([256], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels
        )

        self.stage3_cfg = cfg.model.extra.stage3
        num_channels = self.stage3_cfg.num_channels
        block = blocks_dict[self.stage3_cfg.block]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition2 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels
        )

        self.stage4_cfg = cfg.model.extra.stage4
        num_channels = self.stage4_cfg.num_channels
        block = blocks_dict[self.stage4_cfg.block]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition3 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, multi_scale_output=False
        )

        self.final_layer = nn.Conv2d(
            in_channels=pre_stage_channels[0],
            out_channels=cfg.model.num_joints,
            kernel_size=cfg.model.extra.final_conv_kernel,
            stride=1,
            padding=1 if cfg.model.extra.final_conv_kernel == 3 else 0,
        )

        self.pretrained_layers = cfg.model.extra.pretrained_layers
        if self.use_featuremap:
            self.inplanes = 96
        else:
            self.inplanes = self.num_joints
        if self.return_hoe is True:
            self.hoe_layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
            self.hoe_layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
            self.hoe_layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)
            self.hoe_avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.hoe_fc = nn.Linear(512, 72)

    def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_channels_pre_layer[i],
                                num_channels_cur_layer[i],
                                3,
                                1,
                                1,
                                bias=False,
                            ),
                            nn.BatchNorm2d(num_channels_cur_layer[i]),
                            nn.ReLU(inplace=True),
                        )
                    )
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = (
                        num_channels_cur_layer[i]
                        if j == i - num_branches_pre
                        else inchannels
                    )
                    conv3x3s.append(
                        nn.Sequential(
                            nn.Conv2d(inchannels, outchannels, 3, 2, 1, bias=False),
                            nn.BatchNorm2d(outchannels),
                            nn.ReLU(inplace=True),
                        )
                    )
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_stage(
        self,
        layer_config: HRNetStageConfig,
        num_inchannels: List[int],
        multi_scale_output: bool = True,
    ):
        num_modules = layer_config.num_modules
        num_branches = layer_config.num_branches
        num_blocks = layer_config.num_blocks
        num_channels = layer_config.num_channels
        block = blocks_dict[layer_config.block]
        fuse_method = layer_config.fuse_method

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                HighResolutionModule(
                    num_branches=num_branches,
                    blocks=block,
                    num_blocks=num_blocks,
                    num_inchannels=num_inchannels,
                    num_channels=num_channels,
                    fuse_method=fuse_method,
                    multi_scale_output=reset_multi_scale_output,
                )
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg.num_branches):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)
        feature_map_2 = y_list[0]

        x_list = []
        for i in range(self.stage3_cfg.num_branches):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)
        feature_map_3 = y_list[0]

        x_list = []
        for i in range(self.stage4_cfg.num_branches):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)
        feature_map_4 = y_list[0]
        x = self.final_layer(y_list[0])
        if self.use_featuremap:
            x_cat = torch.cat([feature_map_2, feature_map_3, feature_map_4], 1)
        else:
            x_cat = x
        if self.return_hoe is True:
            y = self.hoe_layer2(x_cat)
            y = self.hoe_layer3(y)
            y = self.hoe_layer4(y)
            y = self.hoe_avgpool(y)
            y = y.view(y.size(0), -1)
            y = self.hoe_fc(y)
            y = F.softmax(y, dim=1)
            return x, y
        return x

    def init_weights(self, pretrained: str = "") -> None:
        """
        Initializes the weights of the model.

        This function initializes the weights of convolutional layers and
        batch normalization layers. It can also load weights from a
        pretrained model if a valid path is provided.

        Args:
            pretrained (str): Path to the pretrained model weights.
                            If not provided, only initializes weights.

        Raises:
            ValueError: If the specified pretrained model file does not exist.
        """

        def _init_conv_layer(layer: nn.Conv2d) -> None:
            """Initializes weights for Conv2d layers."""
            nn.init.normal_(layer.weight, std=0.001)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

        def _init_batch_norm_layer(layer: nn.BatchNorm2d) -> None:
            """Initializes weights for BatchNorm2d layers."""
            nn.init.constant_(layer.weight, 1)
            nn.init.constant_(layer.bias, 0)

        def _init_trans_conv_layer(layer: nn.ConvTranspose2d) -> None:
            """Initializes weights for ConvTranspose2d layers."""
            nn.init.normal_(layer.weight, std=0.001)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

        logger.info("=> Initializing weights from normal distribution")

        # Initialize weights for each layer
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                _init_conv_layer(module)
            elif isinstance(module, nn.BatchNorm2d):
                _init_batch_norm_layer(module)
            elif isinstance(module, nn.ConvTranspose2d):
                _init_trans_conv_layer(module)

        # Load pretrained weights if a valid path is given
        if os.path.isfile(pretrained):
            pretrained_state_dict = torch.load(pretrained)
            logger.info(f"=> Loading pretrained model {pretrained}")
            self.load_state_dict(pretrained_state_dict, strict=False)
        elif pretrained:
            logger.error("=> Please download pretrained models first!")
            raise ValueError(f"{pretrained} does not exist!")

    def get_hoe_params(self):
        for m in self.named_modules():
            if "hoe" in m[0] or "final_layer" in m[0]:
                print(m[0])
                for p in m[1].parameters():
                    if p.requires_grad:
                        yield p


def get_pose_net(cfg: HRNetConfig, is_train: bool, **kwargs):
    model = PoseHighResolutionNet(cfg=cfg, **kwargs)

    if is_train and cfg.model.init_weights:
        model.init_weights(cfg.model.pretrained)
    return model
