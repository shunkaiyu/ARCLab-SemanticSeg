import torch
from . import initialization as init
import torch.nn as nn

class SegmentationModel(torch.nn.Module):

    def initialize(self):
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.segmentation_head)
        if self.classification_head is not None:
            init.initialize_head(self.classification_head)


    def forward(self, x, globlex=None,localx=None):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        if True:
            features = self.encoder(x)
            # print(features[-1].shape)
            # print(features[-2].shape)
            # print(features[-3].shape)
            useLocalGloble = True
            if useLocalGloble:
                globleFeatures = self.encoder(globlex)
                localFeatures = self.encoder(localx)
                features = [globleFeatures[-1],localFeatures[-1],globleFeatures[-4],localFeatures[-4]] + features

            # concat_features = torch.cat([features[-1], globleFeatures[-1], localFeatures[-1]], dim=1) # num channel * 3
            # #print(str(self.encoder.out_channels))
            # #self.encoder.out_channels[-1] = self.encoder.out_channels[-1] * 3
            # self.convblock = nn.Sequential(
            #     nn.Conv2d(self.encoder.out_channels[-1] * 3, self.encoder.out_channels[-1], kernel_size=1, bias=False),
            #     # nn.BatchNorm2d(self.encoder.out_channels[-1]),
            #     # nn.ReLU(),
            # )
            #self.convblock.cuda()
            # print('before conv ',features[-1].size)
            #features[-1] = self.convblock(concat_features)
            #features[-1] = concat_features

            # feature[3] : local feature low level (t=t-1)
            # feature[2] : globle feature low level (t=0)
            # feature[1] : local feature high level (t=t-1)
            # feature[0] : globle feature high level (t=0)



            #features = [globleFeatures[-1],localFeatures[-1],globleFeatures[-4],localFeatures[-4]] + features

            # features[-1] = (features[-1]+globleFeatures[-1]+localFeatures[-1])/3
            # print('after conv ',features[-1].size)
        else:
            features = self.encoder(x)

        decoder_output = self.decoder(*features)

        masks = self.segmentation_head(decoder_output)

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels

        return masks


    # def forward(self, x):
    #     """Sequentially pass `x` trough model`s encoder, decoder and heads"""

    #     #self.check_input_shape(x)

    #     features = self.encoder(x)
    #     decoder_output = self.decoder(*features)

    #     masks = self.segmentation_head(decoder_output)

    #     if self.classification_head is not None:
    #         labels = self.classification_head(features[-1])
    #         return masks, labels

    #     return masks


    def predict(self, x, globlex=None,localx=None):
        """Inference method. Switch model to `eval` mode, call `.forward(x)` with `torch.no_grad()`

        Args:
            x: 4D torch tensor with shape (batch_size, channels, height, width)

        Return:
            prediction: 4D torch tensor with shape (batch_size, classes, height, width)

        """
        if self.training:
            self.eval()

        with torch.no_grad():
            x = self.forward(x,globlex,localx)

        return x

# import torch
# from . import initialization as init


# class SegmentationModel(torch.nn.Module):

#     def initialize(self):
#         init.initialize_decoder(self.decoder)
#         init.initialize_head(self.segmentation_head)
#         if self.classification_head is not None:
#             init.initialize_head(self.classification_head)

#     def forward(self, x):
#         """Sequentially pass `x` trough model`s encoder, decoder and heads"""
#         features = self.encoder(x)
#         decoder_output = self.decoder(*features)

#         masks = self.segmentation_head(decoder_output)

#         if self.classification_head is not None:
#             labels = self.classification_head(features[-1])
#             return masks, labels

#         return masks

#     def predict(self, x):
#         """Inference method. Switch model to `eval` mode, call `.forward(x)` with `torch.no_grad()`

#         Args:
#             x: 4D torch tensor with shape (batch_size, channels, height, width)

#         Return:
#             prediction: 4D torch tensor with shape (batch_size, classes, height, width)

#         """
#         if self.training:
#             self.eval()

#         with torch.no_grad():
#             x = self.forward(x)

#         return x