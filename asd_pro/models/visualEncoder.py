##
# 基于 ResNet18 的视觉前端网络，用于提取嘴部/唇部特征。
# 核心架构由 3D 卷积层和 2D ResNet 构成，能够捕捉唇部的动态运动特征。
# 参考论文: TalkNet (Ruijie Tao et al., ACM MM 2021)
##

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNetLayer(nn.Module):
    """
    【ResNet 基础层定义】
    包含卷积、批量归一化 (BN)、ReLU 激活以及短路连接 (Shortcut Connection)。
    """
    def __init__(self, inplanes, outplanes, stride):
        super(ResNetLayer, self).__init__()
        self.conv1a = nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1a = nn.BatchNorm2d(outplanes, momentum=0.01, eps=0.001)
        self.conv2a = nn.Conv2d(outplanes, outplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.stride = stride
        self.downsample = nn.Conv2d(inplanes, outplanes, kernel_size=(1,1), stride=stride, bias=False)
        self.outbna = nn.BatchNorm2d(outplanes, momentum=0.01, eps=0.001)

        self.conv1b = nn.Conv2d(outplanes, outplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1b = nn.BatchNorm2d(outplanes, momentum=0.01, eps=0.001)
        self.conv2b = nn.Conv2d(outplanes, outplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.outbnb = nn.BatchNorm2d(outplanes, momentum=0.01, eps=0.001)

    def forward(self, inputBatch):
        # 第一层残差计算
        batch = F.relu(self.bn1a(self.conv1a(inputBatch)))
        batch = self.conv2a(batch)
        if self.stride == 1:
            residualBatch = inputBatch
        else:
            residualBatch = self.downsample(inputBatch)
        batch = batch + residualBatch
        intermediateBatch = batch
        batch = F.relu(self.outbna(batch))

        # 第二层残差计算
        batch = F.relu(self.bn1b(self.conv1b(batch)))
        batch = self.conv2b(batch)
        residualBatch = intermediateBatch
        batch = batch + residualBatch
        outputBatch = F.relu(self.outbnb(batch))
        return outputBatch

class ResNet(nn.Module):
    """
    【18层 ResNet 视觉主干网络】
    通过连续的特征下采样，将嘴部图像转化为 512 维的视觉语义嵌入。
    """
    def __init__(self):
        super(ResNet, self).__init__()
        # 定义 4 个阶段的 ResNet 层
        self.layer1 = ResNetLayer(64, 64, stride=1)
        self.layer2 = ResNetLayer(64, 128, stride=2)
        self.layer3 = ResNetLayer(128, 256, stride=2)
        self.layer4 = ResNetLayer(256, 512, stride=2)
        # 平均池化层，将空间特征图压缩为一维向量
        self.avgpool = nn.AvgPool2d(kernel_size=(4,4), stride=(1,1))

    def forward(self, inputBatch):
        batch = self.layer1(inputBatch)
        batch = self.layer2(batch)
        batch = self.layer3(batch)
        batch = self.layer4(batch)
        outputBatch = self.avgpool(batch)
        return outputBatch

class GlobalLayerNorm(nn.Module):
    """
    【全局层归一化】
    相比普通 BN，它在处理变长序列和长视频流推理时具有更好的鲁棒性。
    """
    def __init__(self, channel_size):
        super(GlobalLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.Tensor(1, channel_size, 1))
        self.beta = nn.Parameter(torch.Tensor(1, channel_size, 1))
        self.reset_parameters()

    def reset_parameters(self):
        self.gamma.data.fill_(1)
        self.beta.data.zero_()

    def forward(self, y):
        # 计算全局均值和标准差
        mean = y.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True)
        var = (torch.pow(y-mean, 2)).mean(dim=1, keepdim=True).mean(dim=2, keepdim=True)
        gLN_y = self.gamma * (y - mean) / torch.pow(var + 1e-8, 0.5) + self.beta
        return gLN_y

class visualFrontend(nn.Module):
    """
    【视觉前端总模块】
    结构: 3D 卷积层 (捕捉极短时运动) + 2D ResNet-18 (提取深度外观特征)。
    """
    def __init__(self):
        super(visualFrontend, self).__init__()
        # 初始 3D 卷积层，针对嘴唇开合动作进行特征提取
        self.frontend3D = nn.Sequential(
                            nn.Conv3d(1, 64, kernel_size=(5,7,7), stride=(1,2,2), padding=(2,3,3), bias=False),
                            nn.BatchNorm3d(64, momentum=0.01, eps=0.001),
                            nn.ReLU(),
                            nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1))
                        )
        self.resnet = ResNet()

    def forward(self, inputBatch):
        # inputBatch 形状: (Batch, Time, H, W)
        inputBatch = inputBatch.transpose(0, 1).transpose(1, 2)
        batchsize = inputBatch.shape[0]

        # 1. 运行 3D 卷积提取低级动态特征
        batch = self.frontend3D(inputBatch)

        # 2. 将 Time 维度与 Batch 维度合并，输入 2D ResNet 进行深度语义识别
        batch = batch.transpose(1, 2)
        batch = batch.reshape(batch.shape[0]*batch.shape[1], batch.shape[2], batch.shape[3], batch.shape[4])
        outputBatch = self.resnet(batch)

        # 3. 还原维度，输出 (Time, Batch, 512) 的时序特征序列
        outputBatch = outputBatch.reshape(batchsize, -1, 512)
        outputBatch = outputBatch.transpose(1 ,2)
        outputBatch = outputBatch.transpose(1, 2).transpose(0, 1)
        return outputBatch

class DSConv1d(nn.Module):
    """
    【深度可分离卷积 1D】 (Depthwise Separable Convolution)
    在不牺牲精度的情况下极大地降低了计算量，提升了在 3080ti 上的推理 FPS。
    """
    def __init__(self):
        super(DSConv1d, self).__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Conv1d(512, 512, 3, stride=1, padding=1,dilation=1, groups=512, bias=False),
            nn.PReLU(),
            GlobalLayerNorm(512),
            nn.Conv1d(512, 512, 1, bias=False),
            )

    def forward(self, x):
        # 包含残差连接，防止深度网络中的梯度消失
        out = self.net(x)
        return out + x

class visualTCN(nn.Module):
    """
    【视觉时间卷积网络】 (V-TCN)
    通过堆叠 5 层 1D 卷积，分析 4 秒内的视觉时序变化趋势。
    """
    def __init__(self):
        super(visualTCN, self).__init__()
        stacks = []
        for x in range(5):
            stacks += [DSConv1d()]
        self.net = nn.Sequential(*stacks)

    def forward(self, x):
        out = self.net(x)
        return out

class visualConv1D(nn.Module):
    """
    【视觉最终映射层】
    将视觉嵌入向量压缩至 128 维，以便与音频分支进行跨模态注意力交互。
    """
    def __init__(self):
        super(visualConv1D, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(512, 256, 5, stride=1, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 128, 1),
            )

    def forward(self, x):
        out = self.net(x)
        return out
