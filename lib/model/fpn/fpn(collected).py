# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn.functional as F
from torch import nn


class FPN(nn.Module):
    """
    Module that adds FPN on top of a list of feature maps.
    The feature maps are currently supposed to be in increasing depth
    order, and must be consecutive
    """

    def __init__(
        self, in_channels_list, out_channels, conv_block, top_blocks=None
    ):
        """
        Arguments:
            in_channels_list (list[int]): number of channels for each feature map that
                will be fed
            out_channels (int): number of channels of the FPN representation
            top_blocks (nn.Module or None): if provided, an extra operation will
                be performed on the output of the last (smallest resolution)
                FPN output, and the result will extend the result list
        """
        super(FPN, self).__init__()
        self.inner_blocks = []
        self.layer_blocks = []
        for idx, in_channels in enumerate(in_channels_list, 1):
            #print(idx,in_channels) # 256 512 1024 2048
            inner_block = "fpn_inner{}".format(idx)
            layer_block = "fpn_layer{}".format(idx)

            if in_channels == 0:
                continue
            inner_block_module = conv_block(in_channels, out_channels, 1)#变换通道数
            layer_block_module = conv_block(out_channels, out_channels, 3, 1)#每个层加3X3卷积来避免混叠效应？
            #print("inner_block_module,layer_block_module:\n",inner_block_module,layer_block_module,'\n')
            self.add_module(inner_block, inner_block_module)
            self.add_module(layer_block, layer_block_module)
            self.inner_blocks.append(inner_block)
            self.layer_blocks.append(layer_block)
        self.top_blocks = top_blocks

    def forward(self, x):
        """
        Arguments:
            x (list[Tensor]): feature maps for each feature level.
        Returns:
            results (tuple[Tensor]): feature maps after FPN layers.
                They are ordered from highest resolution first.
        """
        
        #从最后一层开始 last_inner:torch.Size([1, 2048, 25, 34])>>([1, 256, 25, 34])
        last_inner = getattr(self, self.inner_blocks[-1])(x[-1])#getattr用于返回一个对象属性值。
        print("\nlast_inner:\n",x[-1].shape,last_inner.shape)
        results = []
        results.append(getattr(self, self.layer_blocks[-1])(last_inner))#3X3卷积不改通道数256

        for feature, inner_block, layer_block in zip(
            x[:-1][::-1], self.inner_blocks[:-1][::-1], self.layer_blocks[:-1][::-1]
        ):
            print("\nfeature, inner_block, layer_block:\n",feature.shape, inner_block, layer_block,'\n')
            if not inner_block:
                continue
            inner_top_down = F.interpolate(last_inner, scale_factor=2, mode="nearest")#从上到下，上采样（scale尺寸上采倍数）
            inner_lateral = getattr(self, inner_block)(feature)#1x1卷积改变通道数（调用inner_block_module）
            # TODO use size instead of scale to make it robust to different sizes
            # inner_top_down = F.upsample(last_inner, size=inner_lateral.shape[-2:], mode='bilinear', align_corners=False)
            last_inner = inner_lateral + inner_top_down #像素点相加
            results.insert(0, getattr(self, layer_block)(last_inner)) #3x3卷积消除混叠效应（调用layer_block_module）

        if isinstance(self.top_blocks, LastLevelP6P7):
            last_results = self.top_blocks(x[-1], results[-1])
            results.extend(last_results)
        elif isinstance(self.top_blocks, LastLevelMaxPool):
            last_results = self.top_blocks(results[-1])
            results.extend(last_results)

        print("\nresults:",len(results),results[0].shape,results[1].shape,results[2].shape,results[3].shape,results[4].shape)
        return tuple(results)


class LastLevelMaxPool(nn.Module):
    def forward(self, x):
        return [F.max_pool2d(x, 1, 2, 0)]#kernel_size=1,stride=2,pad=0

