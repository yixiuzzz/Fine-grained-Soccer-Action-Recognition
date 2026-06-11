import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pdb


class Action(nn.Module):
    def __init__(self, in_channels):
        super(Action, self).__init__()

        self.in_channels = in_channels
        # self.out_channels = out_channels
        # self.kernel_size = kernel_size
        # self.stride = stride
        # self.padding = padding
        self.reduced_channels = self.in_channels//4
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        # self.fold = self.in_channels // shift_div
        self.fold = self.in_channels // 4

        #shifting
        # self.action_shift = nn.Conv1d(
        #                             self.in_channels, self.in_channels,
        #                             kernel_size=3, padding=1, groups=self.in_channels,
        #                             bias=False)      
        # self.action_shift.weight.requires_grad = True
        # self.action_shift.weight.data.zero_()
        # self.action_shift.weight.data[:self.fold, 0, 2] = 1 # shift left
        # self.action_shift.weight.data[self.fold: 2 * self.fold, 0, 0] = 1 # shift right  


        # if 2*self.fold < self.in_channels:
        #     self.action_shift.weight.data[2 * self.fold:, 0, 1] = 1 # fixed


        # # spatial temporal excitation
        self.action_p1_conv1 = nn.Conv3d(1, 1, kernel_size=(3, 3, 3), 
                                    stride=(1, 1 ,1), bias=False, padding=(1, 1, 1))

        # # # channel excitation
        # self.action_p2_squeeze = nn.Conv2d(self.in_channels, self.reduced_channels, kernel_size=(1, 1), stride=(1 ,1), bias=False, padding=(0, 0))
        # self.action_p2_conv1 = nn.Conv1d(self.reduced_channels, self.reduced_channels, kernel_size=3, stride=1, bias=False, padding=1, 
        #                                groups=1)
        # self.action_p2_expand = nn.Conv2d(self.reduced_channels, self.in_channels, kernel_size=(1, 1), stride=(1 ,1), bias=False, padding=(0, 0))
    


        # motion excitation
        # groups=self.reduced_channels: use depthwise conv
        self.pad = (0,0,0,0,0,0,0,1)
        self.action_p3_squeeze = nn.Conv2d(self.in_channels, self.reduced_channels, kernel_size=(1, 1), stride=(1 ,1), bias=False, padding=(0, 0))
        self.action_p3_bn1 = nn.BatchNorm2d(self.reduced_channels)
        self.action_p3_conv1 = nn.Conv2d(self.reduced_channels, self.reduced_channels, kernel_size=(3, 3), 
                                    stride=(1 ,1), bias=False, padding=(1, 1), groups=self.reduced_channels)
        self.action_p3_expand = nn.Conv2d(self.reduced_channels, self.in_channels, kernel_size=(1, 1), stride=(1 ,1), bias=False, padding=(0, 0))
        # print('=> Using ACTION')

        


    def forward(self, x):
        b, c, t, h, w = x.shape #[16, 24, 16, 112, 112]
        nt = b * t

        # # 2D convolution: motion excitation
        x = x.view(nt, c, h, w) 
        x3 = self.action_p3_squeeze(x)
        x3 = self.action_p3_bn1(x3) #torch.Size([256, 1, 56, 56])
        nt, c, h, w = x3.size()

        x3_plus0, _ = x3.view(b, t, c, h, w).split([t-1, 1], dim=1)
        x3_plus1 = self.action_p3_conv1(x3)
        _ , x3_plus1 = x3.view(b, t, c, h, w).split([1, t-1], dim=1) # torch.Size([16, 15, 6, 56, 56])
        x_p3 = x3_plus1 - x3_plus0 
        x_p3 = F.pad(x_p3, self.pad, mode="constant", value=0)  #torch.Size([16, 6, 16, 56, 56])
       
        x_p3 = self.avg_pool(x_p3.view(nt, c, h, w))
        x_p3 = self.action_p3_expand(x_p3)
        x_p3 = self.sigmoid(x_p3)

        x_p3 = x * x_p3 + x # torch.Size([256, 24, 56, 56])
        nt, c, h, w = x_p3.size()
        x_p3 = x_p3.view(b, c, t, h, w) 

        return x_p3





# class TemporalPool(nn.Module):
#     def __init__(self, net, n_segment):
#         super(TemporalPool, self).__init__()
#         self.net = net
#         self.n_segment = n_segment

#     def forward(self, x):
#         x = self.temporal_pool(x, n_segment=self.n_segment)
#         return self.net(x)

#     @staticmethod
#     def temporal_pool(x, n_segment):
#         nt, c, h, w = x.size()
#         n_batch = nt // n_segment
#         x = x.view(n_batch, n_segment, c, h, w).transpose(1, 2)  # n, c, t, h, w
#         x = F.max_pool3d(x, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0))
#         x = x.transpose(1, 2).contiguous().view(nt // 2, c, h, w)
#         return x


# def make_temporal_shift(net, n_segment, n_div=8, place='blockres', temporal_pool=False):
#     if temporal_pool:
#         n_segment_list = [n_segment, n_segment // 2, n_segment // 2, n_segment // 2]
#     else:
#         n_segment_list = [n_segment] * 4
#     assert n_segment_list[-1] > 0
#     print('=> n_segment per stage: {}'.format(n_segment_list))


#     # pdb.set_trace()
#     import torchvision
#     if isinstance(net, torchvision.models.ResNet):
#         if place == 'block':
#             def make_block_temporal(stage, this_segment):
#                 blocks = list(stage.children())
#                 print('=> Processing stage with {} blocks'.format(len(blocks)))
#                 for i, b in enumerate(blocks):
#                     blocks[i].conv1 = Action(b.conv1, n_segment=this_segment, shift_div = n_div)
#                 return nn.Sequential(*(blocks))

#             pdb.set_trace()
#             net.layer1 = make_block_temporal(net.layer1, n_segment_list[0])
#             net.layer2 = make_block_temporal(net.layer2, n_segment_list[1])
#             net.layer3 = make_block_temporal(net.layer3, n_segment_list[2])
#             net.layer4 = make_block_temporal(net.layer4, n_segment_list[3])

#         elif 'blockres' in place:
#             n_round = 1
#             if len(list(net.layer3.children())) >= 23:
#                 n_round = 2
#                 print('=> Using n_round {} to insert temporal shift'.format(n_round))

#             def make_block_temporal(stage, this_segment):
#                 blocks = list(stage.children())
#                 print('=> Processing stage with {} blocks residual'.format(len(blocks)))
#                 for i, b in enumerate(blocks):
#                     if i % n_round == 0:
#                         blocks[i].conv1 = Action(b.conv1, n_segment=this_segment, shift_div = n_div)
#                         # pdb.set_trace()
#                 return nn.Sequential(*blocks)

#             # pdb.set_trace()
#             net.layer1 = make_block_temporal(net.layer1, n_segment_list[0])
#             net.layer2 = make_block_temporal(net.layer2, n_segment_list[1])
#             net.layer3 = make_block_temporal(net.layer3, n_segment_list[2])
#             net.layer4 = make_block_temporal(net.layer4, n_segment_list[3])
             
#     else:
#         raise NotImplementedError(place)


# def make_temporal_pool(net, n_segment):
#     import torchvision
#     if isinstance(net, torchvision.models.ResNet):
#         print('=> Injecting nonlocal pooling')
#         net.layer2 = TemporalPool(net.layer2, n_segment)
#     else:
#         raise NotImplementedError





