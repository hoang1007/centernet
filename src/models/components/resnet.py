import os
from typing import List, Tuple, Union, Type
import torch
from torch import nn
from .dcnv2 import DeformableConv2d
from src.utils import fill_upsample_weights, fill_fc_weights


def conv3x3(in_planes: int, out_planes: int, stride=1):
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module = None,
    ):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module = None,
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor):
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


class PoseResNet(nn.Module):
    def __init__(
        self,
        block: Union[Type[BasicBlock], Type[Bottleneck]],
        head_conv: int,
        num_layers: Tuple[int],
        num_classes: int,
        from_pretrained: str = None
    ):
        super().__init__()

        self.inplanes = 64
        self.planes = (64, 128, 256, 512)
        self.deconv_filters_size = (256, 128, 64)
        self.devonv_kernels_size = (4, 4, 4)
        self.deconv_with_bias = False
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(
            block, self.planes[0], num_layers[0], stride=1)
        self.layer2 = self._make_layer(
            block, self.planes[1], num_layers[1], stride=2)
        self.layer3 = self._make_layer(
            block, self.planes[2], num_layers[2], stride=2)
        self.layer4 = self._make_layer(
            block, self.planes[3], num_layers[3], stride=2)

        self.deconv_layers = self._make_deconv_layer(
            num_layers=3,
            filters_size=self.deconv_filters_size,
            kernels_size=self.devonv_kernels_size
        )

        deconv_inplanes = self.deconv_filters_size[-1]
        self.heatmap_layer = self._make_sub_layer(
            deconv_inplanes, head_conv, self.num_classes)
        self.offset_layer = self._make_sub_layer(deconv_inplanes, head_conv, 2)
        self.dimension_layer = self._make_sub_layer(
            deconv_inplanes, head_conv, 2)

        if from_pretrained:
            self.init_weights(from_pretrained)
        else:
            fill_fc_weights(self.offset_layer)
            fill_fc_weights(self.dimension_layer)

    def init_weights(self, pretrained_path: str):
        print("Loading pretrained model...")

        if pretrained_path.startswith('http'):
            pretrained = torch.hub.load_state_dict_from_url(
                pretrained_path, progress=True)
        elif os.path.isfile(pretrained_path):
            pretrained = torch.load(pretrained_path)
        else:
            print(f"Pretrained model not found at {pretrained_path}")

        try:
            self.load_state_dict(pretrained, strict=False)

            for _, m in self.deconv_layers.named_modules():
                if isinstance(m, nn.BatchNorm2d):
                    torch.nn.init.constant_(m.weight, 1)
                    torch.nn.init.constant_(m.bias, 0)

            def freeze_weight(model: nn.Module):
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        param.requires_grad = False

            freeze_weight(self.conv1)
            freeze_weight(self.bn1)
            freeze_weight(self.layer1)
            freeze_weight(self.layer2)
            freeze_weight(self.layer3)
        except:
            print("Pretrained model not loaded")


    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.deconv_layers(x)

        heatmap = self.heatmap_layer(x)
        offset = self.offset_layer(x)
        dimension = self.dimension_layer(x)

        # (B, 2, H, W) -> (B, H, W, 2)
        offset = offset.moveaxis(1, -1)
        dimension = dimension.moveaxis(1, -1)

        return heatmap, offset, dimension

    def _make_sub_layer(
        self,
        inplanes: int,
        head_conv: int,
        planes: int,
    ):
        """
        Make a sub layer that produces a heatmap, offset, or dimension.
        """
        return nn.Sequential(
            nn.Conv2d(inplanes, head_conv, kernel_size=3,
                      stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, planes, kernel_size=1, bias=True),
        )

    def _make_layer(
        self,
        block: Union[Type[BasicBlock], Type[Bottleneck]],
        planes: int,
        num_blocks: int,
        stride: int = 1,
    ):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers: List[Union[BasicBlock, Bottleneck]] = []

        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel: int, index: int):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0
        else:
            raise NotImplementedError

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(
        self,
        num_layers: int,
        filters_size: Tuple[int],
        kernels_size: Tuple[int],
    ):
        assert num_layers == len(
            filters_size), "ERROR: num_deconv_layers is different len(num_deconv_filters)"
        assert num_layers == len(
            kernels_size), "ERROR: num_deconv_layers is different len(num_deconv_filters)"

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = self._get_deconv_cfg(
                kernels_size[i], i)

            planes = filters_size[i]
            fc = DeformableConv2d(
                in_channels=self.inplanes,
                out_channels=planes,
                kernel_size=3
            )

            up = nn.ConvTranspose2d(
                in_channels=planes,
                out_channels=planes,
                kernel_size=kernel,
                stride=2,
                padding=padding,
                output_padding=output_padding,
                bias=self.deconv_with_bias,
            )

            fill_upsample_weights(up)

            self.inplanes = planes
            layers.extend((
                fc,
                nn.BatchNorm2d(planes),
                nn.ReLU(inplace=True),
                up,
                nn.BatchNorm2d(planes),
                nn.ReLU(inplace=True),
            ))

        return nn.Sequential(*layers)
