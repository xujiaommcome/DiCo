import argparse
import test_util
from no_updown.magicnet import VNet_Magic
from dataloaders.dataset import *

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=2, help='batch_size per gpu')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--txt_path', type=str, default='/home/prohibit/DiCo-code/data/ImageCAS.txt',
                    help='dataset_txt_position')
parser.add_argument('--datasets', type=str, default='imageCAS', help='datasets') # imageCAS,N_cas2023,parse2022,SEG_A
parser.add_argument('--number', type=int, default=850, help='number of train')
args = parser.parse_args()



def create_model(n_classes=14, ema=False):
    # Network definition
    net = VNet_Magic(n_channels=1, n_classes=n_classes)
    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)
    model = net.cuda()
    if ema:
        for param in model.parameters():
            param.detach_()
    return model



os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
batch_size = args.batch_size * len(args.gpu.split(','))

num_classes = 2
patch_size = (96, 96, 96)


model = create_model(n_classes=num_classes)


model.load_state_dict(torch.load(
    '/home/prohibit/DiCo-code/model/paper_output/ImageCAS/vnet_iter_20000_dice_0.7379206059173157.pth'))


print('导入成功')


dataset_name = args.datasets
txt_path = args.txt_path
number = args.number
with open(txt_path, 'r') as f:
    total_list = f.readlines()
total_list = ["{}".format(item.strip()) for item in total_list]

train_list, test_list = total_list[:number], total_list[number:]

# 用其他方法
dice_all,_,all_metrics = test_util.validation_all_case(model,
                                                       num_classes=num_classes,
                                                       dataset=args.datasets,
                                                       image_list=test_list,
                                                       patch_size=patch_size,
                                                       stride_xy=16,
                                                       stride_z=16)
metric_mean, metric_std = np.mean(all_metrics, axis=0), np.std(all_metrics, axis=0)
print(('Final Average DSC:{:.4f}, jc: {:.4f}, ASD: {:.4f}, NSD: {:.4f}'
                 .format(metric_mean[0].mean(), metric_mean[1].mean(), metric_mean[2].mean(),metric_mean[3].mean()
                         )))
print(dice_all.mean())
