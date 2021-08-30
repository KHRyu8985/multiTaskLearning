"""
based on code from fastMRI
https://github.com/facebookresearch/fastMRI/tree/master/fastmri/models
"""

import torch
from torch import nn
from torch.nn import functional as F

from typing import List


class AUnet(nn.Module):
    """
    PyTorch implementation of a U-Net model with attention.
    attention in downsampling, bottleneck, and upsampling
    number of attentional gates = number of contrasts
    """

    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        chans: int = 32,
        num_pool_layers: int = 4,
        drop_prob: float = 0.0,
        decoder_heads = None,
    ):
        """
        Args:
            in_chans: Number of channels in the input to the U-Net model.
            out_chans: Number of channels in the output to the U-Net model.
            chans: Number of output channels of the first convolution layer.
            num_pool_layers: Number of down-sampling and up-sampling layers.
            drop_prob: Dropout probability.
            decoder_heads: number of contrasts in dataset
        """
        super().__init__()

        # parameters / sizes
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob

        filters = [
            chans * 2 ** layer for layer in range(num_pool_layers + 1)
        ]

        ################################################
        ############# shared unet layers ###############
        ################################################

        # down sample layers
        self.global_downsample = nn.ModuleList()
        self.global_downsample_conv = nn.ModuleList() 
        # first pooling layer
        self.global_downsample.append(
            ConvBlock(in_chans, chans, drop_prob)
            )
        self.global_downsample_conv.append(
            ConvBlock(chans, chans, drop_prob)
        )

        # rest of the pooling layers
        for idx_layer in range(0, num_pool_layers - 1):
            self.global_downsample.append(
                ConvBlock(filters[idx_layer], filters[idx_layer + 1], drop_prob)
            )
            self.global_downsample_conv.append(
                ConvBlock(filters[idx_layer + 1], filters[idx_layer + 1], drop_prob)
            )


        # bottleneck
        self.global_bottleneck = nn.ModuleList([
            ConvBlock(filters[-2], filters[-1], drop_prob)
            ])
        self.global_bottleneck_conv = nn.ModuleList([
            ConvBlock(filters[-1], filters[-1], drop_prob)
        ])

        # up sample layers
        self.global_uptranspose = nn.ModuleList()
        # post-skip connection concat conv layers
        self.global_upsample_conv = nn.ModuleList()

        for idx_layer in reversed(range(1, num_pool_layers)):
            self.global_uptranspose.append(
                TransposeConvBlock(filters[idx_layer + 1], filters[idx_layer])
                )
            # ConvBlock is one unit block at a time now
            self.global_upsample_conv.append(nn.Sequential(
                ConvBlock(filters[idx_layer + 1], filters[idx_layer], drop_prob),
                ConvBlock(filters[idx_layer], filters[idx_layer], drop_prob),
                ))

        self.global_uptranspose.append(
            TransposeConvBlock(filters[1], filters[0])
            )
        self.global_upsample_conv.append(
            nn.Sequential(
                ConvBlock(filters[1], filters[0], drop_prob),
                ConvBlock(filters[0], filters[0], drop_prob),
                nn.Conv2d(filters[0], self.out_chans, kernel_size = 1, stride = 1),
            )
        )

        ################################################
        ############ task attention layers #############
        ################################################
        self.downsample_att = nn.ModuleList()
        self.upsample_att = nn.ModuleList()
        self.downsample_att_conv = nn.ModuleList()
        self.upsample_att_conv = nn.ModuleList()

        for idx_task in range(decoder_heads):
            # att layers are lists within a list [idx_task][idx_layer]
            self.downsample_att.append(
                nn.ModuleList([AttBlock([filters[0], filters[0], filters[0]])])
                )
            self.upsample_att.append(
                nn.ModuleList([AttBlock([2 * filters[0], filters[0], filters[0]])])
                )

            for idx_layer in range(num_pool_layers):
                # att layers
                self.downsample_att[idx_task].append(self.att_layer([2 * filters[i + 1], filters[i + 1], filters[i + 1]]))
                self.upsample_att[idx_task].append(self.att_layer([filters[i + 1] + filters[i], filters[i], filters[i]]))

                # att conv block layers

        for i in range(4):
            if i < 3:
                self.encoder_block_att.append(self.conv_layer([filter[i + 1], filter[i + 2]]))
                self.decoder_block_att.append(self.conv_layer([filter[i + 1], filter[i]]))
            else:
                self.encoder_block_att.append(self.conv_layer([filter[i + 1], filter[i + 1]]))
                self.decoder_block_att.append(self.conv_layer([filter[i + 1], filter[i + 1]]))


    def forward(
        self, 
        image: torch.Tensor,
        int_contrast: int,
        ) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.
            int_contrast: i.e. 0 for div_coronal_pd_fs, 1 for div_coronal_pd
        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        
        stack = []
        output = image

        # apply down-sampling layers
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            
            output = F.avg_pool2d(output, kernel_size=2, stride=2, padding=0)

        for layer in self.conv:
            output = layer(output)

        # apply up-sampling layers
        for idx_upsample, (transpose_conv, conv) in enumerate(zip(
            self.up_transpose_convs[int_contrast], 
            self.up_convs[int_contrast],
        )):
            downsample_layer = stack[-(idx_upsample + 1)]
            output = transpose_conv(output)

            # reflect pad on the right/botton if needed to handle odd input dimensions
            padding = [0, 0, 0, 0]
            if output.shape[-1] != downsample_layer.shape[-1]:
                padding[1] = 1  # padding right
            if output.shape[-2] != downsample_layer.shape[-2]:
                padding[3] = 1  # padding bottom
            if torch.sum(torch.tensor(padding)) != 0:
                output = F.pad(output, padding, "reflect")
            
            output = torch.cat([output, downsample_layer], dim=1)
            output = conv(output)

        return output



class ConvBlock(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, LeakyReLU activation and dropout.
    """

    def __init__(self, in_chans: int, out_chans: int, drop_prob: float):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
            drop_prob: Dropout probability.
        Differs from fastmri unet ConvBlock in that there is only one "block"
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob

        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size = 3, padding = 1, bias = False),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope = 0.2, inplace = True),
            nn.Dropout2d(drop_prob),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.
        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        return self.layers(image)


class TransposeConvBlock(nn.Module):
    """
    A Transpose Convolutional Block that consists of one convolution transpose
    layers followed by instance normalization and LeakyReLU activation.
    """

    def __init__(self, in_chans: int, out_chans: int):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans

        self.layers = nn.Sequential(
            nn.ConvTranspose2d(
                in_chans, out_chans, kernel_size=2, stride=2, bias=False
            ),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.
        Returns:
            Output tensor of shape `(N, out_chans, H*2, W*2)`.
        """
        image = image.clone()
        return self.layers(image)


class AttBlock(nn.Module):
    """
    attentional block 
    1x1 conv, Norm, ReLU
    1x1 conv, Norom, Sigmoid
    """

    def __init__(self, in_chans: int, chans: int, out_chans: int):
        """
        Args:
            in_chans: Number of channels in the input.
            chans: Number of channels after first layter
            out_chans: Number of channels in the output.
        """
        super().__init__()

        self.in_chans = in_chans
        self.chans = chans
        self.out_chans = out_chans
        
        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, chans, kernel_size = 1, padding = 0),
            nn.InstanceNorm2d(chans),
            nn.LeakyReLU(negative_slope = 0.2, inplace = True),
            nn.Conv2d(chans, out_chans, kernel_size = 1, padding = 0),
            nn.InstanceNorm2d(chans),
            nn.Sigmoid(),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.
        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        image = image.clone()
        return self.layers(image)