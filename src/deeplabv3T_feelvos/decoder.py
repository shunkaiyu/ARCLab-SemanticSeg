"""
BSD 3-Clause License

Copyright (c) Soumith Chintala 2016,
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import torch
from torch import nn
from torch.nn import functional as F

__all__ = ["DeepLabV3Decoder"]


class DeepLabV3Decoder(nn.Sequential):
    def __init__(self, in_channels, out_channels=256, atrous_rates=(12, 24, 36)):
        super().__init__(
            ASPP(in_channels, out_channels, atrous_rates),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.out_channels = out_channels

    def forward(self, *features):
        return super().forward(features[-1])


class DeepLabV3PlusDecoder(nn.Module):
    def __init__(
        self,
        use_high_level,
        use_low_level,
        encoder_channels,
        out_channels=256,
        atrous_rates=(12, 24, 36),
        output_stride=16,

    ):
        super().__init__()
        if output_stride not in {8, 16}:
            raise ValueError("Output stride should be 8 or 16, got {}.".format(output_stride))

        self.out_channels = out_channels
        self.output_stride = output_stride
        self.use_high_level = use_high_level
        self.use_low_level = use_low_level

        self.aspp = nn.Sequential(
            ASPP(encoder_channels[-1], out_channels, atrous_rates, separable=True),
            SeparableConv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

        scale_factor = 2 if output_stride == 8 else 4
        self.up = nn.UpsamplingBilinear2d(scale_factor=scale_factor)

        highres_in_channels = encoder_channels[-4]
        highres_out_channels = 48   # proposed by authors of paper
        self.block1 = nn.Sequential(
            nn.Conv2d(highres_in_channels, highres_out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(highres_out_channels),
            nn.ReLU(),
        )
        self.block2 = nn.Sequential(
            SeparableConv2d(
                highres_out_channels + out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(768, 256, kernel_size=1, bias=False),
            # nn.BatchNorm2d(256),
            # nn.ReLU(),
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(highres_in_channels*3, highres_out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(highres_out_channels),
            nn.ReLU(),
        )

    
    def forward(self, *features):
        aspp_features = self.aspp(features[-1])
        aspp_features = self.up(aspp_features)
        high_res_features = self.block1(features[-4])
        concat_features = torch.cat([aspp_features, high_res_features], dim=1)
        fused_features = self.block2(concat_features)
        return fused_features    

    # def forward(self, *features):
    #     # features[3] : local feature low level high res (t=t-1)
    #     # features[2] : globle feature low level high res(t=0)
    #     # features[1] : local feature high level (t=t-1)
    #     # features[0] : globle feature high level (t=0)
    #     aspp_features = self.aspp(features[-1])
    #     # print("***********")
    #     # print(self.use_high_level)
    #     # print(aspp_features.shape)
    #     # print(aspp_features_combined.shape)
    #     # test concat before upsampling
    #     if self.use_high_level:
    #         # print("enter high level ****")
    #         aspp_features_globle = self.aspp(features[0])
    #         aspp_features_local = self.aspp(features[1])

    #         # #_____________________________________________________________________
    #         # # local matching
    #         # aspp_features_local = aspp_features_local - aspp_features
    #         # # globle matching
    #         # aspp_features_globle = aspp_features_globle - aspp_features
    #         # #_____________________________________________________________________
    #         print('test local matching')
    #         temptest = aspp_features_local - aspp_features
    #         print('local matching size: ', temptest.shape,' || ',aspp_features.shape)

    #         aspp_features_combined = torch.cat([aspp_features, aspp_features_local, aspp_features_globle], dim=1)
    #         aspp_features_combined = self.block3(aspp_features_combined)
    #         aspp_features = self.up(aspp_features_combined)
    #     else:
    #         aspp_features = self.up(aspp_features)

    #     #aspp_features = self.up(aspp_features)
    #     high_res_features = features[-4]


    #     #high_res_features = self.block1(features[-4])
        
    #     if self.use_low_level:
    #         high_res_features_globle = features[2]
    #         high_res_features_local = features[3]
    #         # #_____________________________________________________________________
    #         # # local matching
    #         # high_res_features_local = high_res_features_local - high_res_features
    #         # # globle matching
    #         # high_res_features_globle = high_res_features_globle - high_res_features
    #         # #_____________________________________________________________________

    #         high_res_features_combined = torch.cat([high_res_features, high_res_features_local, high_res_features_globle], dim=1)
            
    #         high_res_features = self.block4(high_res_features_combined)
    #     else:
    #         high_res_features = self.block1(high_res_features)

    #     # max_dim_2 = max(high_res_features.size(2), aspp_features.size(2))
    #     # aspp_features = F.pad(aspp_features, (0, 0, 0, max_dim_2-aspp_features.size(2), 0, 0), "constant", 0)



    #     concat_features = torch.cat([aspp_features, high_res_features], dim=1)
    #     fused_features = self.block2(concat_features)

    #     aspp_features = self.aspp(features[-1])
    #     aspp_features = self.up(aspp_features)
    #     high_res_features = self.block1(features[-4])



    #     concat_features = torch.cat([aspp_features, high_res_features], dim=1)
    #     fused_features = self.block2(concat_features)
    #     return fused_features    



        # return fused_features


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )


class ASPPSeparableConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__(
            SeparableConv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates, separable=False):
        super(ASPP, self).__init__()
        modules = []
        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            )
        )

        rate1, rate2, rate3 = tuple(atrous_rates)
        ASPPConvModule = ASPPConv if not separable else ASPPSeparableConv

        modules.append(ASPPConvModule(in_channels, out_channels, rate1))
        modules.append(ASPPConvModule(in_channels, out_channels, rate2))
        modules.append(ASPPConvModule(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


class SeparableConv2d(nn.Sequential):

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            bias=True,
    ):
        dephtwise_conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=False,
        )
        pointwise_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=bias,
        )
        super().__init__(dephtwise_conv, pointwise_conv)

# """
# BSD 3-Clause License

# Copyright (c) Soumith Chintala 2016,
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.

# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# """

# import torch
# from torch import nn
# from torch.nn import functional as F

# __all__ = ["DeepLabV3Decoder"]


# class DeepLabV3Decoder(nn.Sequential):
#     def __init__(self, in_channels, out_channels=256, atrous_rates=(12, 24, 36)):
#         super().__init__(
#             ASPP(in_channels, out_channels, atrous_rates),
#             nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(),
#         )
#         self.out_channels = out_channels

#     def forward(self, *features):
#         return super().forward(features[-1])


# class DeepLabV3PlusDecoder(nn.Module):
#     def __init__(
#         self,
#         encoder_channels,
#         out_channels=256,
#         atrous_rates=(12, 24, 36),
#         output_stride=16,
#     ):
#         super().__init__()
#         if output_stride not in {8, 16}:
#             raise ValueError("Output stride should be 8 or 16, got {}.".format(output_stride))

#         self.out_channels = out_channels
#         self.output_stride = output_stride

#         self.aspp = nn.Sequential(
#             ASPP(encoder_channels[-1], out_channels, atrous_rates, separable=True),
#             SeparableConv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(),
#         )

#         scale_factor = 2 if output_stride == 8 else 4
#         self.up = nn.UpsamplingBilinear2d(scale_factor=scale_factor)

#         highres_in_channels = encoder_channels[-4]
#         highres_out_channels = 48   # proposed by authors of paper
#         self.block1 = nn.Sequential(
#             nn.Conv2d(highres_in_channels, highres_out_channels, kernel_size=1, bias=False),
#             nn.BatchNorm2d(highres_out_channels),
#             nn.ReLU(),
#         )
#         self.block2 = nn.Sequential(
#             SeparableConv2d(
#                 highres_out_channels + out_channels,
#                 out_channels,
#                 kernel_size=3,
#                 padding=1,
#                 bias=False,
#             ),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(),
#         )

#     def forward(self, *features):
#         aspp_features = self.aspp(features[-1])
#         aspp_features = self.up(aspp_features)
#         high_res_features = self.block1(features[-4])
#         # print(aspp_features.shape)
#         # print(high_res_features.shape)
#         concat_features = torch.cat([aspp_features, high_res_features], dim=1)
#         fused_features = self.block2(concat_features)
#         return fused_features


# class ASPPConv(nn.Sequential):
#     def __init__(self, in_channels, out_channels, dilation):
#         super().__init__(
#             nn.Conv2d(
#                 in_channels,
#                 out_channels,
#                 kernel_size=3,
#                 padding=dilation,
#                 dilation=dilation,
#                 bias=False,
#             ),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(),
#         )


# class ASPPSeparableConv(nn.Sequential):
#     def __init__(self, in_channels, out_channels, dilation):
#         super().__init__(
#             SeparableConv2d(
#                 in_channels,
#                 out_channels,
#                 kernel_size=3,
#                 padding=dilation,
#                 dilation=dilation,
#                 bias=False,
#             ),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(),
#         )


# class ASPPPooling(nn.Sequential):
#     def __init__(self, in_channels, out_channels):
#         super().__init__(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(),
#         )

#     def forward(self, x):
#         size = x.shape[-2:]
#         for mod in self:
#             x = mod(x)
#         return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


# class ASPP(nn.Module):
#     def __init__(self, in_channels, out_channels, atrous_rates, separable=False):
#         super(ASPP, self).__init__()
#         modules = []
#         modules.append(
#             nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels, 1, bias=False),
#                 nn.BatchNorm2d(out_channels),
#                 nn.ReLU(),
#             )
#         )

#         rate1, rate2, rate3 = tuple(atrous_rates)
#         ASPPConvModule = ASPPConv if not separable else ASPPSeparableConv

#         modules.append(ASPPConvModule(in_channels, out_channels, rate1))
#         modules.append(ASPPConvModule(in_channels, out_channels, rate2))
#         modules.append(ASPPConvModule(in_channels, out_channels, rate3))
#         modules.append(ASPPPooling(in_channels, out_channels))

#         self.convs = nn.ModuleList(modules)

#         self.project = nn.Sequential(
#             nn.Conv2d(5 * out_channels, out_channels, kernel_size=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#         )

#     def forward(self, x):
#         res = []
#         for conv in self.convs:
#             res.append(conv(x))
#         res = torch.cat(res, dim=1)
#         return self.project(res)


# class SeparableConv2d(nn.Sequential):

#     def __init__(
#             self,
#             in_channels,
#             out_channels,
#             kernel_size,
#             stride=1,
#             padding=0,
#             dilation=1,
#             bias=True,
#     ):
#         dephtwise_conv = nn.Conv2d(
#             in_channels,
#             in_channels,
#             kernel_size,
#             stride=stride,
#             padding=padding,
#             dilation=dilation,
#             groups=in_channels,
#             bias=False,
#         )
#         pointwise_conv = nn.Conv2d(
#             in_channels,
#             out_channels,
#             kernel_size=1,
#             bias=bias,
#         )
#         super().__init__(dephtwise_conv, pointwise_conv)