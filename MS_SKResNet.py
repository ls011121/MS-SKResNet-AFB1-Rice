import torch
import torch.nn as nn
import torch.nn.functional as F


# ==================== 轻量化 SK 模块 ====================
class SKConvLite(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, L=16, reduction=16, groups=1):
        super(SKConvLite, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.L = L
        self.M = 2
        self.reduction = reduction
        self.groups = groups

        self.d = max(in_channels // reduction, L)

        # 第一个分支：膨胀率为1
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, dilation=1, groups=groups, bias=False
        )

        # 第二个分支：膨胀率为2
        self.conv2 = nn.Conv1d(
            in_channels, out_channels, kernel_size=3,
            stride=stride, padding=2, dilation=2, groups=groups, bias=False
        )

        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Sequential(
            nn.Conv1d(out_channels, self.d, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.d, out_channels * self.M, kernel_size=1, bias=True)
        )

        self._init_weights()

    def _init_weights(self):
        for m in [self.conv1, self.conv2]:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

        nn.init.constant_(self.bn1.weight, 1)
        nn.init.constant_(self.bn1.bias, 0)
        nn.init.constant_(self.bn2.weight, 1)
        nn.init.constant_(self.bn2.bias, 0)

    def forward(self, x):
        u1 = self.relu(self.bn1(self.conv1(x)))
        u2 = self.relu(self.bn2(self.conv2(x)))

        u = u1 + u2
        s = self.global_avg_pool(u)
        z = self.fc(s)

        z = z.view(z.size(0), self.M, self.out_channels, 1)
        z = F.softmax(z, dim=1)

        z1 = z[:, 0, :, :]
        z2 = z[:, 1, :, :]

        out = u1 * z1 + u2 * z2

        return out


class SKBlockLite(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, reduction=16):
        super(SKBlockLite, self).__init__()

        self.sk_conv = SKConvLite(
            in_channels, out_channels, stride=stride,
            reduction=reduction, groups=1
        )

        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=3,
            padding=1, stride=1, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.sk_conv(x)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out


# ==================== 轻量化多尺度融合 ====================
class MultiScaleFusionLite(nn.Module):


    def __init__(self, in_channels, out_channels):
        super(MultiScaleFusionLite, self).__init__()

        if out_channels % 3 != 0:

           out_channels = ((out_channels + 2) // 3) * 3

        self.actual_out_channels = out_channels

        branch_channels = out_channels // 3

        self.branch1 = nn.Sequential(
            nn.Conv1d(in_channels, branch_channels, kernel_size=3,
                      padding=1, bias=False, groups=in_channels),
            nn.Conv1d(branch_channels, branch_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(branch_channels),
            nn.ReLU(inplace=True)
        )

        self.branch2 = nn.Sequential(
            nn.Conv1d(in_channels, branch_channels, kernel_size=7,
                      padding=3, bias=False, groups=in_channels),
            nn.Conv1d(branch_channels, branch_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(branch_channels),
            nn.ReLU(inplace=True)
        )

        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, branch_channels, kernel_size=11,
                      padding=5, bias=False, groups=in_channels),
            nn.Conv1d(branch_channels, branch_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(branch_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        return torch.cat([b1, b2, b3], dim=1)

class SpectralModel4ClassWithSKLite_V3(nn.Module):


    def __init__(self, input_length=196, num_classes=4):
        super(SpectralModel4ClassWithSKLite_V3, self).__init__()

        self.input_length = input_length
        self.num_classes = num_classes

        # ========== 输入层 ==========
        # 使用能被3整除的数：63
        self.input_conv = MultiScaleFusionLite(1, 63)
        input_out_channels = self.input_conv.actual_out_channels  # ✅ 获取实际输出通道

        # ========== Stem 层 ==========
        self.stem = nn.Sequential(
            nn.Conv1d(input_out_channels, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(96),
            nn.ReLU(inplace=True),
        )

        # ========== Stage 1: 只有1个Block ==========
        self.stage1 = nn.Sequential(
            SKBlockLite(96, 96, stride=1, reduction=16),
        )

        # ========== Stage 2 ==========
        self.stage2 = nn.Sequential(
            SKBlockLite(96, 192, stride=2, reduction=16),
            SKBlockLite(192, 192, stride=1, reduction=16),
        )


        # ========== 全局池化 ==========
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)

        # ========== 分类头 ==========
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(192 * 2, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.input_conv(x)
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        #x = self.stage3(x)
        #x = self.stage4(x)

        avg_pool = self.global_avg_pool(x).flatten(1)
        max_pool = self.global_max_pool(x).flatten(1)
        x = torch.cat([avg_pool, max_pool], dim=1)

        x = self.classifier(x)
        return x