import torch
import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    __constants__ = ['downsample']
    # inplanes：输入通道数
    # planes：输出通道数
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=nn.BatchNorm1d):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.selu = nn.SELU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.selu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.selu(out)

        return out

class Bottleneck(nn.Module):
    __constants__ = ['downsample']
    # inplanes：输入通道数
    # planes：输出通道数
    def __init__(self, inplanes, planes1, planes2, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=nn.BatchNorm1d):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes1, stride)
        self.bn1 = norm_layer(planes1)
        self.selu = nn.SELU(inplace=True)
        self.conv2 = conv3x3(planes1, planes1)
        self.bn2 = norm_layer(planes1)
        self.conv3 = conv1x1(planes1, planes2)
        self.bn3 = norm_layer(planes2)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.selu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.selu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.selu(out)

        return out


'''Attention: 
Every model only has a channel adjustment layer to be the former processing layer, 
which does not have any downsampling'''

# resnet18 w/ 1d conv
# output: //8, c512
class resnet18_1d(nn.Module):
    def __init__(self,
                 n_in,
                 require_multi = False) -> None:
        super(resnet18_1d, self).__init__()
        self.multi = require_multi
        self.chan_adj = nn.Conv1d(n_in, 64, 7, 1, 3, bias=False)
        self.block1 = nn.Sequential(
            BasicBlock(64,64),
            BasicBlock(64,64)
        )
        self.block2 = nn.Sequential(
            BasicBlock(64,128,2,conv1x1(64,128,2)),
            BasicBlock(128,128)
        )
        self.block3 = nn.Sequential(
            BasicBlock(128,256,2,conv1x1(128,256,2)),
            BasicBlock(256,256)
        )
        self.block4 = nn.Sequential(
            BasicBlock(256,512,2,conv1x1(256,512,2)),
            BasicBlock(512,512)
        )
        self.res18 = nn.Sequential(
            self.chan_adj,
            self.block1,
            self.block2,
            self.block3,
            self.block4,
        )
        # self.res18 = nn.DataParallel(self.res18)
    def forward(self, input):
        if self.multi:
            out1 = self.block1(self.chan_adj(input))
            out2 = self.block2(out1)
            out3 = self.block3(out2)
            out4 = self.block4(out3)
            return out1, out2, out3, out4
        else:
            return self.res18(input)

# resnet34 w/ 1d conv
# output: //8, c512
class resnet34_1d(nn.Module):
    def __init__(self,
                 n_in, 
                 require_multi = False) -> None:
        super(resnet34_1d, self).__init__()
        self.multi = require_multi
        self.chan_adj = nn.Conv1d(n_in, 64, 7, 1, 3, bias=False)
        self.block1 = nn.Sequential(
            BasicBlock(64,64),
            BasicBlock(64,64),
            BasicBlock(64,64)
        )
        self.block2 = nn.Sequential(
            BasicBlock(64,128,2,conv1x1(64,128,2)),
            BasicBlock(128,128),
            BasicBlock(128,128),
            BasicBlock(128,128)
        )
        self.block3 = nn.Sequential(
            BasicBlock(128,256,2,conv1x1(128,256,2)),
            BasicBlock(256,256),
            BasicBlock(256,256),
            BasicBlock(256,256),
            BasicBlock(256,256),
            BasicBlock(256,256)
        )
        self.block4 = nn.Sequential(
            BasicBlock(256,512,2,conv1x1(256,512,2)),
            BasicBlock(512,512),
            BasicBlock(512,512)
        )
        self.res34 = nn.Sequential(
            self.chan_adj,
            self.block1,
            self.block2,
            self.block3,
            self.block4,
        )
        # self.res34 = nn.DataParallel(self.res34)
    def forward(self, input):
        if self.multi:
            out1 = self.block1(self.chan_adj(input))
            out2 = self.block2(out1)
            out3 = self.block3(out2)
            out4 = self.block4(out3)
            return out1, out2, out3, out4
        else:
            return self.res34(input)

# resnet50 w/ 1d conv
# output: //8, c2048
class resnet50_1d(nn.Module):
    def __init__(self,
                 n_in) -> None:
        super(resnet50_1d, self).__init__()
        self.chan_adj = nn.Conv1d(n_in, 64, 7, 1, 3, bias=False)
        self.block1 = nn.Sequential(
            Bottleneck(64,64,256,1,conv1x1(64,256)),
            Bottleneck(256,64,256),
            Bottleneck(256,64,256)
        )
        self.block2 = nn.Sequential(
            Bottleneck(256,128,512,2,conv1x1(256,512,2)),
            Bottleneck(512,128,512),
            Bottleneck(512,128,512),
            Bottleneck(512,128,512)
        )
        self.block3 = nn.Sequential(
            Bottleneck(512,256,1024,2,conv1x1(512,1024,2)),
            Bottleneck(1024,256,1024),
            Bottleneck(1024,256,1024),
            Bottleneck(1024,256,1024),
            Bottleneck(1024,256,1024),
            Bottleneck(1024,256,1024)
        )
        self.block4 = nn.Sequential(
            Bottleneck(1024,512,2048,2,conv1x1(1024,2048,2)),
            Bottleneck(2048,256,2048),
            Bottleneck(2048,256,2048)
        )
        self.res50 = nn.Sequential(
            self.chan_adj,
            self.block1,
            self.block2,
            self.block3,
            self.block4,
        )
        # self.res50 = nn.DataParallel(self.res50)

    def forward(self, input):
        return self.res50(input)


if __name__=='__main__':
    input = torch.randn((64,16,1000)).to(torch.float16).cuda()     # (batch size, dimension, length)
    print("input shape:", input.shape)
    # res18 = nn.DataParallel(resnet18_1d(input.shape[1]).to(torch.float16)).cuda()
    # print("resnet18 output shape:", res18.module(input).shape)
    # del res18
    # res34 = nn.DataParallel(resnet34_1d(input.shape[1]).to(torch.float16)).cuda()
    # print("resnet34 output shape:", res34.module(input).shape)
    # del res34
    # res50 = nn.DataParallel(resnet50_1d(input.shape[1]).to(torch.float16)).cuda()
    # print("resnet50 output shape:", res50.module(input).shape)
    # del res50
    res34 = nn.DataParallel(resnet34_1d(input.shape[1],require_multi=True).to(torch.float16)).cuda()
    print("resnet18 output shape:", res34.module(input).shape)
    del res34
