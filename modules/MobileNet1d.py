'''MobileNetV3 1D
modified from <https://github.com/xiaolai-sqlai/mobilenetv3/>, we transfer the 2D model to 1D.
'''
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class LayerNorm_insert(nn.Module):
    def __init__(self, 
                 in_dim,
                 ):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
    
    def forward(self, x):
        return self.norm(x.permute(0,2,1)).permute(0,2,1)


class hswish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out


class hsigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out


class SeModule(nn.Module):
    def __init__(self, in_size, reduction=4):
        super(SeModule, self).__init__()
        expand_size =  max(in_size // reduction, 8)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_size, expand_size, kernel_size=1, bias=False),
            LayerNorm_insert(expand_size),
            nn.ReLU(inplace=True),
            nn.Conv1d(expand_size, in_size, kernel_size=1, bias=False),
            nn.Hardsigmoid()
        )

    def forward(self, x):
        return x * self.se(x)


class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, kernel_size, in_size, expand_size, out_size, act, se, stride):
        super(Block, self).__init__()
        self.stride = stride

        self.conv1 = nn.Conv1d(in_size, expand_size, kernel_size=1, bias=False)
        self.ln1 = LayerNorm_insert(expand_size)
        self.act1 = act(inplace=True)

        self.conv2 = nn.Conv1d(expand_size, expand_size, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=expand_size, bias=False)
        self.ln2 = LayerNorm_insert(expand_size)
        self.act2 = act(inplace=True)
        self.se = SeModule(expand_size) if se else nn.Identity()

        self.conv3 = nn.Conv1d(expand_size, out_size, kernel_size=1, bias=False)
        self.ln3 = LayerNorm_insert(out_size)
        self.act3 = act(inplace=True)

        self.skip = None
        if stride == 1 and in_size != out_size:
            self.skip = nn.Sequential(
                nn.Conv1d(in_size, out_size, kernel_size=1, bias=False),
                LayerNorm_insert(out_size)
            )

        if stride == 2 and in_size != out_size:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels=in_size, out_channels=in_size, kernel_size=3, groups=in_size, stride=2, padding=1, bias=False),
                LayerNorm_insert(in_size),
                nn.Conv1d(in_size, out_size, kernel_size=1, bias=True),
                LayerNorm_insert(out_size)
            )

        if stride == 2 and in_size == out_size:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels=in_size, out_channels=out_size, kernel_size=3, groups=in_size, stride=2, padding=1, bias=False),
                LayerNorm_insert(out_size)
            )

    def forward(self, x):
        skip = x

        out = self.act1(self.ln1(self.conv1(x)))
        out = self.act2(self.ln2(self.conv2(out)))
        out = self.se(out)
        out = self.ln3(self.conv3(out))
        
        if self.skip is not None:
            skip = self.skip(skip)
        return self.act3(out + skip)


'''
Differences from vanilla:
1. 1D instead of 2D
2. No last classifier, global average pooling and dropout
3. Output channel = 512 by conv2
4. input channel is made costumizable
5. stem conv (conv1) is 7k1s instead of 3k2s, cuz we experiential find no downsampling in stem is better
6. Block No.1 in self.bneck is changed to stride=1, so that the downsampling rate for the whole network is 2**3=8
7. LayerNorm instead of BatchNorm
'''
class MobileNetV3_Small_1D(nn.Module):
    def __init__(self, 
                 in_dim=3, 
                 act=nn.Hardswish):
        super(MobileNetV3_Small_1D, self).__init__()
        self.conv1 = nn.Conv1d(in_dim, 16, 7, 1, 3)
        self.ln1 = LayerNorm_insert(16)
        self.hs1 = act(inplace=True)

        self.bneck = nn.Sequential(
            Block(3, 16, 16, 16, nn.ReLU, True, 1),
            Block(3, 16, 72, 24, nn.ReLU, False, 2),
            Block(3, 24, 88, 24, nn.ReLU, False, 1),
            Block(5, 24, 96, 40, act, True, 2),
            Block(5, 40, 240, 40, act, True, 1),
            Block(5, 40, 240, 40, act, True, 1),
            Block(5, 40, 120, 48, act, True, 1),
            Block(5, 48, 144, 48, act, True, 1),
            Block(5, 48, 288, 96, act, True, 2),
            Block(5, 96, 576, 96, act, True, 1),
            Block(5, 96, 576, 96, act, True, 1),
        )

        self.conv2 = nn.Conv1d(96, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.ln2 = LayerNorm_insert(512)
        self.hs2 = act(inplace=True)

        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.hs1(self.ln1(self.conv1(x)))
        out = self.bneck(out)
        out = self.hs2(self.ln2(self.conv2(out)))
        return out



# TODO: transfer Large model to 1D
class MobileNetV3_Large(nn.Module):
    def __init__(self, num_classes=1000, act=nn.Hardswish):
        super(MobileNetV3_Large, self).__init__()
        self.conv1 = nn.Conv1d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(16)
        self.hs1 = act(inplace=True)

        self.bneck = nn.Sequential(
            Block(3, 16, 16, 16, nn.ReLU, False, 1),
            Block(3, 16, 64, 24, nn.ReLU, False, 2),
            Block(3, 24, 72, 24, nn.ReLU, False, 1),
            Block(5, 24, 72, 40, nn.ReLU, True, 2),
            Block(5, 40, 120, 40, nn.ReLU, True, 1),
            Block(5, 40, 120, 40, nn.ReLU, True, 1),
            Block(3, 40, 240, 80, act, False, 2),
            Block(3, 80, 200, 80, act, False, 1),
            Block(3, 80, 184, 80, act, False, 1),
            Block(3, 80, 184, 80, act, False, 1),
            Block(3, 80, 480, 112, act, True, 1),
            Block(3, 112, 672, 112, act, True, 1),
            Block(5, 112, 672, 160, act, True, 2),
            Block(5, 160, 672, 160, act, True, 1),
            Block(5, 160, 960, 160, act, True, 1),
        )


        self.conv2 = nn.Conv1d(160, 960, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm1d(960)
        self.hs2 = act(inplace=True)
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        self.linear3 = nn.Linear(960, 1280, bias=False)
        self.bn3 = nn.BatchNorm1d(1280)
        self.hs3 = act(inplace=True)
        self.drop = nn.Dropout(0.2)

        self.linear4 = nn.Linear(1280, num_classes)
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.hs1(self.bn1(self.conv1(x)))
        out = self.bneck(out)

        out = self.hs2(self.bn2(self.conv2(out)))
        out = self.gap(out).flatten(1)
        out = self.drop(self.hs3(self.bn3(self.linear3(out))))
        
        return self.linear4(out)



if __name__ == '__main__':
    import torch
    B=64 ; L=200 ; C=3
    x = torch.randn(B, L, C).cuda()
    model = MobileNetV3_Small_1D(3).cuda()

    out = model(x.permute(0,2,1))
    print(out.shape)
    