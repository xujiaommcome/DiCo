import torch
import torch.nn as nn


# 示例：定义鉴别器（Discriminator）
class ConvDiscriminator(nn.Module):
    def __init__(self, input_channels=2):
        super(ConvDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv3d(input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(0.2),
            nn.Conv3d(32, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(16),
            nn.LeakyReLU(0.2),
            nn.Conv3d(16, 1, kernel_size=3, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


class Change(nn.Module):
    def __init__(self, input_channels=2):
        super(Change, self).__init__()
        self.model = nn.Sequential(
            nn.Conv3d(input_channels, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):

    def __init__(self, ndf=64, n_channel=1):
        super(Discriminator, self).__init__()
        self.conv0 = nn.Conv2d(n_channel, ndf, kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.Conv2d(n_channel, ndf, kernel_size=4, stride=2, padding=1)

        self.conv2 = nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1)
        self.avgpool = nn.AvgPool2d(kernel_size=6)
        self.classifier = nn.Linear(ndf*8, 1)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.dropout = nn.Dropout2d(0.5)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, mapp, image):
        batch_size = mapp.shape[0]
        map_feature = self.conv0(mapp)
        image_feature = self.conv1(image)
        x = torch.add(map_feature, image_feature)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        x = self.conv4(x)
        x = self.leaky_relu(x)

        x = self.avgpool(x)

        x = x.view(batch_size, -1)
        x = self.classifier(x)
        # x = self.softmax(x)
        x = x.reshape((batch_size, 1))

        return x



class Discriminator_3d(nn.Module):

    def __init__(self, ndf=64, n_channel=2):
        super(Discriminator_3d, self).__init__()
        self.conv0 = nn.Conv3d(2, ndf, kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.Conv3d(1, ndf, kernel_size=4, stride=2, padding=1)

        self.conv2 = nn.Conv3d(ndf, ndf*2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv3d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv3d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1)
        self.avgpool = nn.AvgPool3d(kernel_size=6)
        self.classifier = nn.Linear(ndf*8, 1)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.dropout = nn.Dropout3d(0.5)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, maps, image,flag=1):
        # print(type(map))
        # print(mapp)
        batch_size = image.shape[0]
        if flag==1:
            map_feature = self.conv1(maps)
        else:
            map_feature = self.conv0(maps)
        image_feature = self.conv1(image)
        x = torch.add(map_feature, image_feature)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        x = self.conv4(x)
        x = self.leaky_relu(x)

        x = self.avgpool(x)

        x = x.view(batch_size, -1)
        x = self.classifier(x)
        # x = self.softmax(x)
        x = x.reshape((batch_size, 1))

        return x


def dice_loss(pred, target):
    pred = pred.view(-1)
    target = target.view(-1)
    smooth = 1e-5
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    loss = 1 - dice
    return loss


class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.bce_loss = torch.nn.BCELoss()

    def forward(self, pred, target):
        bce_loss = self.bce_loss(pred, target)
        # bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(pred, target)
        dice_loss_value = dice_loss(pred, target)
        loss = self.alpha * bce_loss + (1 - self.alpha) * dice_loss_value
        return loss
