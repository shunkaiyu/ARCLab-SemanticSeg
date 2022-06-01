from json import decoder
from re import I
import torch
#from Embeddings import PixelwiseEmbedding
from . import initialization as init
import torch.nn as nn
import torch.nn.functional as F
import torchshow as ts

def distance(p, q):
    # ps = torch.sum(p * p)
    # qs = torch.sum(q * q)
    norm = torch.norm(p-q, p=2, dim=1)
    res = 1 - (2 / (1 + torch.exp(norm)))
    return res

def local_matching(x,y):
    N,C,H,W = x.shape
    #print("in local m ", x.shape)
    x1 = x.reshape(N,C,H*W)
    x2 = torch.unsqueeze(x1,dim=-1)
    y1 = y.reshape(N,C,H*W)
    y2 = torch.unsqueeze(y1,dim=-2)
    # print(x1.shape)
    # print(y1.shape)
    # xt = x1
    # for i in range(0,H*W):
    #     #print(x1[:,:,0].shape)
    #     m1=torch.min(distance(torch.unsqueeze(x1[:,:,i],dim=-1),y1[:,:,:H*W//2]),dim=-1,keepdim=True)
    #     m2=torch.min(distance(torch.unsqueeze(x1[:,:,i],dim=-1),y1[:,:,H*W//2:]),dim=-1,keepdim=True) 
    #     #print(m[0].shape)
    #     xt[:,:,i] = torch.minimum(m1[0],m2[0])
    # ps = torch.matmul(x1.transpose(1,2),y1)
    # print(ps.shape)
    # qs = torch.matmul(y1.transpose(1,2),x1)
    # dif = ps-qs
    dif = x2 - y2
    print("dif shape: ", dif.shape)
    norm = torch.norm(dif,p=2,dim=1)
    #print(norm.shape)
    res = 1 - (2 / (1 + torch.exp(norm)))
    print("res shape: ",res.shape)
    findmin = torch.min(res,dim=-1)
    print("findmin: ", findmin[0].shape)
    return findmin[0].view(N,H,W)

# def lmm_slices(x,y):
#     N,C,H,W = x.shape
#     x1 = x[:,:,:H//2,:W//2]
#     y1 = y[:,:,:H//2,:W//2]
#     print(x1.shape)
#     print(H//2*W//2)
#     upleft = local_matching(x1,y1)
#     print(upleft.shape)
#     upleft1 = upleft.view(N,C,H//2*W//2)
#     x2 = x[:,:,:H//2,W//2:]
#     y2 = y[:,:,:H//2,W//2:]
#     print(x2.shape)
#     upright = local_matching(x2,y2)
#     upright1 = upright.view(N,C,H//2*W//2)
#     x3 = x[:,:,H//2:,:W//2]
#     y3 = y[:,:,H//2:,:W//2]
#     lowleft = local_matching(x3,y3)
#     lowleft1 = lowleft.view(N,C,H//2*W//2)
#     x4 = x[:,:,H//2:,W//2:]
#     y4 = y[:,:,H//2:,W//2:]
#     lowright = local_matching(x4,y4)
#     lowright1 = lowright.view(N,C,H//2*W//2)
#     concat = x.view(N,C,H*W)
#     concat[:,:,:H//2*W//2] = upleft1
#     concat[:,:,H//2*W//2:2*H//2*W//2] = upright1
#     concat[:,:,2*H//2*W//2:3*H//2*W//2] = lowleft1
#     concat[:,:,3*H//2*W//2:4*H//2*W//2] = lowright1
#     res = concat.view(N,C,H,W)
#     print(concat.shape)
#     return res


# def global_matching(x, y):
#     output = torch.zeros(x.size(0), 1, x.size(2), x.size(3))
#     for i in range(x.size(0)):
#         for j in range(x.size(2)):
#             for k in range(x.size(3)):
#                 output[i, :, j, k] = distance(x[i, :, j, k], y[i, :, j, k])
#     return output
def global_matching(x, y):
    output = torch.zeros(x.size(0), x.size(1), x.size(2))
    for i in range(x.size(0)):
        for j in range(x.size(1)):
            for k in range(x.size(2)):
                output[i, j, k] = distance(x[i, j, k], y[i, j, k])
    return output


class SegmentationModel(torch.nn.Module):

    def initialize(self):
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.segmentation_head)
        if self.classification_head is not None:
            init.initialize_head(self.classification_head)
        

    def forward(self, x, globlex=None,localx=None):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        if False:
            features = self.encoder(x)
            print('feature size: ')
            print(features[-1].shape)
            print(features[-2].shape)
            print(features[-3].shape)
            print(features[-4].shape)
            print(features[-5].shape)
            print(features[-6].shape)

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
        #print('encoder output size: ',features.shape)

        decoder_output = self.decoder(*features)
        # print('decoder output size: ',decoder_output.shape)

        #print(x == localx)
        addAfterDecoder = True
        if (addAfterDecoder):
            globleFeatures = self.encoder(globlex)
            decoder_output_globle = self.decoder(*globleFeatures)

            localFeatures = self.encoder(localx)
            decoder_output_local = self.decoder(*localFeatures)

            # feature_map = F.interpolate(decoder_output,size=[60,106])
            # local_map = F.interpolate(decoder_output_local,size=[60,106])
            m = nn.MaxPool2d(4, stride = 4)
            # feature_map =m(decoder_output)
            # local_map = m(decoder_output_local)
            
            #print(feature_map.shape)

            #local_stack = decoder_output_local
            #print(features==localFeatures)
            #print(features[0] == localFeatures[0])
            #local_stack = local_matching(feature_map[:,:,:,:], local_map[:,:,:,:])
            local_mask = self.segmentation_head(decoder_output_local)
            #ts.show(local_stack)
            local_mask = m(local_mask)
            # print(local_mask.shape)
            #local_stack=torch.unsqueeze(local_stack, 1)
            #print(local_stack.shape)
            # for i in range(0,4):
            #     k = 64*i
            #     local_dm = local_matching(decoder_output[:,k:k+64,:,:], decoder_output_local[:,k:k+64,:,:])
            #     local_stack[:,k:k+64,:,:] = local_dm

            # m = nn.Upsample(scale_factor=4, mode='nearest')
            # local_stack = m(local_stack)

            decoder_comb = torch.cat([decoder_output,local_mask], dim=1)
            # print(decoder_comb.shape)

            # decoder_comb = torch.cat([decoder_output, decoder_output, decoder_output], dim=1)
            # for c_i in range((decoder_output.shape)[1]):
            #     decoder_comb[:,c_i*3,:,:] = decoder_output[:,c_i,:,:]
            #     decoder_comb[:,c_i*3+1,:,:] = decoder_output[:,c_i,:,:]
            #     decoder_comb[:,c_i*3+2,:,:] = decoder_output[:,c_i,:,:]

            decoder_output = decoder_comb
            decoder_output = self.testConvAfterDecoder(decoder_comb)

        
        masks = self.segmentation_head(decoder_output)
        # print('seghead output size: ',masks.shape)
        # print('_______________________________________________________________________')

        addAfterSegHead = False
        if (addAfterSegHead):
            globleFeatures = self.encoder(globlex)
            decoder_output_globle = self.decoder(*globleFeatures)

            localFeatures = self.encoder(localx)
            decoder_output_local = self.decoder(*localFeatures)

            masks_globle = self.segmentation_head(decoder_output_globle)
            masks_local = self.segmentation_head(decoder_output_local)



#______________feelvos____________________________________________
            # x1_l = []; x1_e = []
            # x2_l = []; x2_e = []
            # x3_l = []; x3_e = []
            # gm = []; lm = []
            # logits = []

            # x1 = masks_globle
            # x2 = masks_local
            # x3 = masks
            
            # x1 = F.interpolate(x1, 32)
            # x2 = F.interpolate(x2, 32)
            # x3 = F.interpolate(x3, 32)
            

            # for i in range(13):
            #     print(x1.shape)
            #     x1_l.append(x1[:, i, :, :].unsqueeze(1))
            #     print(x1_l[0].shape)
            #     x1_e.append(self.embedding(x1_l[i]))
            #     x2_l.append(x2[:, i, :, :].unsqueeze(1))
            #     x2_e.append(self.embedding(x2_l[i]))
            #     x3_l.append(x3[:, i, :, :].unsqueeze(1))
            #     x3_e.append(self.embedding(x3_l[i]))
            #     with torch.no_grad():
            #         gm.append(global_matching(x1_e[i], x3_e[i]))
            #         lm.append(global_matching(x2_e[i], x3_e[i]))
            #     x_t = torch.cat((x3, gm[i].cuda(), lm[i].cuda(), x2_l[i]), dim=1)
            #     print(x_t.shape)
            #     logits.append(self.dsh(x_t))
            # print(len(logits))
            # print(logits[0].shape)
            # x = None
            # for i in range(13):
            #     if i == 0:
            #         x = logits[i]
            #     else:
            #         x = torch.cat((logits[i-1], logits[i]), dim=1)
            # print('x shape: ',x.shape)
#__________________________________________________________

            # self.testConvAfterSeg = nn.Sequential(
            #     nn.Conv2d(13 * 3, 128, kernel_size=1, bias=False),
            #     nn.Conv2d(128, 256, kernel_size=1, bias=False),
            #     nn.Conv2d(256, 128, kernel_size=1, bias=False),
            #     nn.Conv2d(128, 13, kernel_size=1, bias=False),
            #     # nn.BatchNorm2d(13),
            #     # nn.ReLU(),
            #     )
            # self.testConvAfterSeg.cuda()

            masks_comb = torch.cat([masks, masks_globle, masks_local], dim=1)
            for c_i in range((masks.shape)[1]):
                #print(masks_globle[:,c_i,:,:].shape)
                masks_comb[:,c_i*3,:,:] = masks[:,c_i,:,:]
                masks_comb[:,c_i*3+1,:,:] = global_matching(masks_globle[:,c_i,:,:], masks[:,c_i,:,:])
                masks_comb[:,c_i*3+2,:,:] = global_matching(masks_local[:,c_i,:,:], masks[:,c_i,:,:])


            masks = self.testConvAfterSeg(masks_comb)

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

class Double33Conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.double33conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.double33conv(x)

class Up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:            
            self.up = nn.ConvTranspose2d(in_ch//2, out_ch//2, kernel_size=2, stride=2)
        
        self.conv = Double33Conv(in_ch, out_ch)
                                         
    
    def forward(self, x1, x2):
        x1 = self.up(x1)

        dy = x2.size()[2]-x1.size()[2]
        dx = x2.size()[3]-x1.size()[3]
        """ Caution: Padding dimension
        N, C, H, W, dx=diffence of W-value
        pad=(w_left,w_right,h_top,h_bottom)
        """
        x1 = F.pad(input=x1, pad=(dx//2, dx-dx//2, dy//2, dy-dy//2))
        # print('sizes',x1.size(),x2.size(),dx // 2, dx - dx//2, dy // 2, dy - dy//2)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)