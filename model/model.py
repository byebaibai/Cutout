from model.basic import *


class MobileUnet(nn.Module):
    def __init__(self, in_channels=3, n_classes=2):

        super(MobileUnet, self).__init__()

        self.mobilenet_block = MobileNetBlock(in_channels)

        self.unet_block1 = UetBlock(320, 96, dilation=2)
        self.unet_block2 = UetBlock(192, 32, dilation=6)
        self.unet_block3 = UetBlock(64, 24, dilation=12)
        self.unet_block4 = UetBlock(48, 16, dilation=18)

        self.last_conv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, n_classes, kernel_size=1, stride=1)
        )
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        e1, e2, e3, e4, feature_map = self.mobilenet_block(x)

        feature_map = self.unet_block1(e4, feature_map)
        feature_map = self.unet_block2(e3, feature_map)
        feature_map = self.unet_block3(e2, feature_map)
        feature_map = self.unet_block4(e1, feature_map)

        heat_map = self.last_conv(feature_map)

        heat_map = F.upsample(heat_map, scale_factor=2, mode='bilinear', align_corners=True)

        return F.sigmoid(heat_map)
