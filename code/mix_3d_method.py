import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

def group_shuffle(x1, x2):
    size = (x1.size(2)) // 2
    sub_blocks1 = [x1[:, :, i * size:(i + 1) * size, j * size:(j + 1) * size, :] for i in range(2) for j in range(2)]
    sub_blocks2 = [x2[:, :, i * size:(i + 1) * size, j * size:(j + 1) * size, :] for i in range(2) for j in range(2)]

    # 从每个输入的四个子块中随机选择两个序号的子块
    all_indices = torch.tensor([0, 3, 1, 2])
    selected_indices1 = all_indices[:2]
    selected_indices2 = all_indices[2:4]
    half_1 = torch.cat([sub_blocks1[selected_indices1[0]], sub_blocks2[selected_indices2[1]]], dim=3)
    half_2 = torch.cat([sub_blocks2[selected_indices2[0]], sub_blocks1[selected_indices1[1]]], dim=3)
    half_3 = torch.cat([sub_blocks1[selected_indices2[0]], sub_blocks2[selected_indices1[1]]], dim=3)
    half_4 = torch.cat([sub_blocks2[selected_indices1[0]], sub_blocks1[selected_indices2[1]]], dim=3)
    # # 将两部分按照规则拼接
    combined_block1 = torch.cat([half_1, half_2], dim=2)
    combined_block2 = torch.cat([half_4, half_3], dim=2)

    return combined_block1, combined_block2


def generate_mip_projectional(picture, gt, axis=2):
    data = picture.squeeze(0).squeeze(0) #[96,96,96]
    gt = torch.argmax(gt, dim=1, keepdim=True)  #[1,1,96,96,96]
    label = gt.squeeze(0).squeeze(0) #[96,96,96]
    mip_projection = torch.max(data, dim=axis)[0] #[96,96]
    # 假设 max_indices 是通过 torch.argmax 得到的索引数组
    max_indices = torch.argmax(data, dim=axis) #[96,96]

    # 初始化一个与 mip_projection 相同大小的数组，用于存储结果
    label_project = torch.zeros_like(mip_projection) #【96,96】

    # 遍历 max_indices 中的每个元素
    for i in range(label_project.shape[0]):
        for j in range(label_project.shape[1]):
            # 获取对应位置的索引值
            index = max_indices[i, j].item()

            # 检查 label 中对应位置的值是否为 1
            num = label[i, j, index].item()
            if num != 0:
                # 如果是，将 label_project 对应位置标注为 1
                label_project[i, j] = 255
            else:
                # 否则，标注为 0
                label_project[i, j] = 0
    mip_projection = (mip_projection - torch.min(mip_projection)) * 255 / (
            torch.max(mip_projection) - torch.min(mip_projection))
    
    # has_negative = (label_project.long() < 0).any().item()

    # one_hot_label = F.one_hot(label_project.long(), num_classes=2).float()

    # # 调整形状
    # mip_projection = mip_projection.unsqueeze(0).unsqueeze(0)
    # one_hot_label = one_hot_label.unsqueeze(0).permute(0, 3, 1, 2)

    return  mip_projection.unsqueeze(0).unsqueeze(0),label_project.unsqueeze(0).unsqueeze(0)


    
    


def max_min_group(image1, image2):
    position_max = torch.where(image1 >= image2, torch.tensor(1), torch.tensor(2))
    position_min = torch.where(image1 <= image2, torch.tensor(1), torch.tensor(2))

    max_image = torch.max(image1, image2)
    min_image = torch.min(image1, image2)

    return max_image, min_image, position_max, position_min


def refresh_max_min_group(mix_feature_max, mix_feature_min, position_max, position_min):
    refresh_xL = torch.where(position_max == 1, mix_feature_max, mix_feature_min)
    refresh_xU = torch.where(position_min == 1, mix_feature_max, mix_feature_min)
    return refresh_xL, refresh_xU


def volume2blocks(x, stride_h=2, stride_w=2, stride_d=1):
    """b c (hd h) (wd w) (dd d) -> (hd wd dd b) c h w d"""
    x = rearrange(x, 'b c (hd h) (wd w) (dd d) -> (hd wd dd b) c h w d', hd=stride_h, wd=stride_w, dd=stride_d)
    # x = rearrange(x, 'b c (hd h) (wd w) (dd d) -> (hd wd dd b) c h w', hd=stride_h, wd=stride_w, dd=stride_d)

    return x


def blocks2volume(x):
    """(hg wg b) c h w -> b c (hg h) (wg w)"""
    x = rearrange(x, '(hd wd dd b) c h w d -> b c (hd h) (wd w) (dd d)', hd=2, wd=2, dd=1)
    return x

def mul_view(x):
    loc = volume2blocks(x, stride_h=2, stride_w=2, stride_d=1)
    # glb = F.interpolate(x, size=[48, 48, 32], mode='trilinear', align_corners=False)
    # return torch.cat((loc, glb), dim=0)
    return loc


def re_vol(x):
    x = F.interpolate(x, size=[96, 96, 64], mode='nearest')
    return x


class conv(nn.Module):
    def __init__(self, in_channels):
        super(conv, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, 16*in_channels, kernel_size=3, stride=1, padding=1).to('cuda')
        self.bn1 = nn.BatchNorm3d(16*in_channels).to('cuda')
        self.conv2 = nn.Conv3d(16*in_channels, in_channels, kernel_size=3, stride=1, padding=1).to('cuda')
        self.bn2 = nn.BatchNorm3d(in_channels).to('cuda')

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        return x


class MixedNetwork(nn.Module):
    def __init__(self, c):
        super(MixedNetwork, self).__init__()
        self.conv_up = nn.Conv3d(c, 16 * c, kernel_size=3, stride=1, padding=1).to('cuda')
        self.conv_down = nn.Conv3d(16 * c, c, kernel_size=3, stride=1, padding=1).to('cuda')
        self.norm2 = nn.LayerNorm(96).to('cuda')
        self.dropout = nn.Dropout(0.1).to('cuda')

    def forward(self, x):
        input_conv = self.conv_up(x)
        input_conv = self.norm2(input_conv)
        input_conv = self.dropout(input_conv)
        out_conv = self.conv_down(input_conv)
        return out_conv
