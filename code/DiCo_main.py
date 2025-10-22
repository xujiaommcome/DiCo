import os
import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from utils import ramps, test_util
from dataloaders.dataset import *
from networks.dicovnet import VNet
from monai.networks.nets import UNETR
from mix_3d_method import  generate_mip_projectional,mul_view
from descriminator import Discriminator, Discriminator_3d
from monai.losses import DiceCELoss


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='lr_0.001_test_5%_Parse2022', help='dataset_name')
parser.add_argument('--root_path', type=str, default='/home/prohibit/DiCo', help='Name of Dataset')
parser.add_argument('--exp', type=str, default='C+V', help='exp_name')
parser.add_argument('--model', type=str, default='V-Net', help='model_name')
parser.add_argument('--max_iteration', type=int, default=40000, help='maximum iteration to train')
parser.add_argument('--total_samples', type=int, default=100, help='total samples of the dataset')
parser.add_argument('--max_train_samples', type=int, default=90, help='maximum samples to train')
parser.add_argument('--max_test_samples', type=int, default=10, help='maximum samples to test')
parser.add_argument('--labeled_bs', type=int, default=1, help='batch_size of labeled data per gpu')
parser.add_argument('--batch_size', type=int, default=2, help='batch_size per gpu')
parser.add_argument('--base_lr', type=float, default=0.001, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--labelnum', type=int, default=5, help='labeled trained samples')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--lamda', type=float, default=0.2, help='weight to balance all losses')
parser.add_argument('--ema_decay', type=float, default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str, default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float, default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float, default=200.0, help='consistency_rampup')
parser.add_argument('--T_dist', type=float, default=1.0, help='Temperature for organ-class distribution')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--txt_path', type=str,
                    default='/home/prohibit/DiCo/data/Parse2022.txt',
                    help='dataset_txt_position')
parser.add_argument('--datasets', type=str, default='parse2022', help='datasets')  # imageCAS,N_cas2023,parse2022,SEG_A
args = parser.parse_args()


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        # ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
        ema_param.data.mul_(param.data).add_(1 - alpha, param.data)


def create_model(n_classes=14, ema=False):
    # Network definition
    net = VNet(n_channels=1, n_classes=n_classes)
    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)
    model = net.cuda()
    if ema:
        for param in model.parameters():
            param.detach_()
    return model


snapshot_path = "../model" + "/{}_{}_{}labeled_cons{}_/{}".format(args.dataset_name, args.exp, args.labelnum,
                                                                             args.consistency,
                                                                             args.model)
txt_path = args.txt_path

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
batch_size = args.batch_size * len(args.gpu.split(','))

num_classes = 2
class_momentum = 0.999
patch_size = (96, 96, 96)
args.root_path = args.root_path

train_data_path = args.root_path
max_iterations = args.max_iteration
base_lr = args.base_lr
labeled_bs = args.labeled_bs

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)


def config_log(snapshot_path_tmp, typename):
    formatter = logging.Formatter(fmt='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    logging.getLogger().setLevel(logging.INFO)

    handler = logging.FileHandler(snapshot_path_tmp + "/log_{}.txt".format(typename), mode="w")
    handler.setFormatter(formatter)
    handler.setLevel(logging.INFO)
    logging.getLogger().addHandler(handler)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    sh.setLevel(logging.INFO)
    logging.getLogger().addHandler(sh)
    return handler, sh


def train(train_list, test_list, fold_id=1):
    snapshot_path_tmp = snapshot_path

    handler, sh = config_log(snapshot_path_tmp, 'fold' + str(fold_id))
    logging.info(str(args))

    model = create_model(n_classes=num_classes)
    model1 = UNETR(
        in_channels=1,
        out_channels=2,
        img_size=(48, 48, 96),
        feature_size=16,
        hidden_size=768,
        mlp_dim=3072,
        num_heads=12,
        pos_embed="perceptron",
        norm_name="instance",
        res_block=True,
        dropout_rate=0.0,
    ).cuda()
    D_2d = Discriminator().cuda()
    loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
    criterion = torch.nn.CrossEntropyLoss()  # 二元交叉熵损失

    # model.load_state_dict(torch.load(
    #     '/home/prohibit/DiCo-code/model/paper_output/vnet_iter_15000_dice_0.8605341066346446.pth'),strict=False)
    # model1.load_state_dict(torch.load(
    #     '/home/prohibit/DiCo-code/model/paper_output/unetr_iter_15000_dice_0.8605341066346446.pth'),strict=False)

    db_train = BTCV(train_list,
                    base_dir=train_data_path,
                    datasets=args.datasets,
                    transform=transforms.Compose([
                        RandomCrop(patch_size),
                        ToTensor(),
                    ])
                    )

    labelnum = args.labelnum
    labeled_idxs = list(range(labelnum))  # 5
    unlabeled_idxs = list(range(labelnum, len(train_list)))  #
    print(labeled_idxs, unlabeled_idxs)
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size,
                                          args.batch_size - args.labeled_bs)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=1, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    # optimizer = torch.optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    # optimizer1 = torch.optim.SGD(
    #     [
    #         {"params": model1.parameters()},
    #     ],
    #     lr=base_lr,
    #     momentum=0.9,
    #     weight_decay=0.0001
    # )
    optimizer = torch.optim.AdamW([
        {'params': model.parameters()},
    ], lr=base_lr)

    optimizer1 = torch.optim.AdamW([
        {'params': model.parameters()},
    ], lr=base_lr)

    Dopt = torch.optim.Adam(D_2d.parameters(), lr=1e-4, betas=(0.9, 0.99))


    writer = SummaryWriter(snapshot_path_tmp)
    logging.info("{} itertations per epoch".format(len(trainloader)))

    iter_num = 0
    best_dice_avg = 0
    metric_all_cases = None
    max_epoch = max_iterations // len(trainloader) + 1
    iterator = tqdm(range(max_epoch), ncols=70)
    print('iterator:', len(iterator))  # ----------max_iterations // len(trainloader) + 1
    print('trainloader:', len(trainloader))  # --------5

    for epoch_num in iterator:  # 80000/5+1
        for i_batch, sampled_batch in enumerate(trainloader):  # 25
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            label_batch = label_batch.squeeze().long()
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            xU = volume_batch[labeled_bs:]
            xL = volume_batch[:labeled_bs]

            y = label_batch[:labeled_bs].unsqueeze(0)

            model.train()
            model1.train()

            feature_xL = model(xL)
            feature_xU = model(xU)

            height = 48
            weight = 48
            deep = 96
            mul_xU = mul_view(xU)  # [4,1,48,48,64]
            mul_xL = mul_view(xL)
            # print('XL shape:',)

            min_xL = F.interpolate(xL, size=[height, weight, deep], mode='trilinear', align_corners=False)
            min_xU = F.interpolate(xU, size=[height, weight, deep], mode='trilinear', align_corners=False)
            glb_loc_xU = torch.cat([min_xU, mul_xU], dim=0)
            glb_loc_xL = torch.cat([min_xL, mul_xL], dim=0)

            vit_xL = model1(glb_loc_xL, 1)
            vit_xU = model1(glb_loc_xU, 1)


            """
            mip鉴别器
            """
            # 1.0
            real_labels = torch.tensor([[1]]).cuda().float()
            fake_labels = torch.tensor([[0]]).cuda().float()

            # 2.0得到标签和图片组合的mip投影-------2d特征[96,96]
            image_xL, label_project_xL_gt = generate_mip_projectional(xL, y, 2)  # [96,96]
            image_xU, label_project_xU1 = generate_mip_projectional(xU, feature_xU, 2)
            _, label_project_xU2 = generate_mip_projectional(xU, vit_xU, 2)
            _, label_project_xL1 = generate_mip_projectional(xL, feature_xL, 2)
            _, label_project_xL2 = generate_mip_projectional(xL, vit_xL, 2)

            with torch.no_grad():
                D_output_flase1 = D_2d(label_project_xU1, image_xU)
                D_output_flase2 = D_2d(label_project_xU2, image_xU)

            loss_adv1 = criterion(D_output_flase1, real_labels)
            loss_adv2 = criterion(D_output_flase2, real_labels)

            loss1 = loss_function(feature_xL, y)
            loss2 = loss_function(vit_xL, y)

            if loss1 > loss2:
                label_vit_xU = torch.argmax(vit_xU, dim=1, keepdim=True)
                mix_loss = loss_function(feature_xU, label_vit_xU)
                loss = loss1 + mix_loss + loss_adv1
                loss_vit = loss2 + loss_adv2


            else:
                label_feature_xU = torch.argmax(feature_xU, dim=1, keepdim=True)
                mix_loss = loss_function(vit_xU, label_feature_xU)
                loss = loss1 + loss_adv1
                loss_vit = loss2 + mix_loss + loss_adv2


            optimizer.zero_grad()
            optimizer1.zero_grad()
            loss.backward()
            loss_vit.backward()
            optimizer.step()
            optimizer1.step()

            """
            Train D
            """
            model.eval()
            model1.eval()
            D_2d.train()
            with torch.no_grad():
                outputs_cnn_xU = model(xU)
                outputs_vit_xU = model1(glb_loc_xU, 1)

            _, label_xU1 = generate_mip_projectional(xU, outputs_cnn_xU, 2)
            _, label_xU2 = generate_mip_projectional(xU, outputs_vit_xU, 2)

            D_output_1 = D_2d(label_xU1, image_xU)
            D_output_2 = D_2d(label_xU2, image_xU)
            D_output_3 = D_2d(label_project_xL_gt, image_xL)


            loss_fake1 = criterion(D_output_1, fake_labels)
            loss_fake2 = criterion(D_output_2, fake_labels)
            loss_real = criterion(D_output_3, real_labels)
            total_2d_loss = loss_fake1 + loss_fake2 + loss_real

            # 计算损失
            D_loss = total_2d_loss

            # 优化步骤
            Dopt.zero_grad()
            D_loss.backward()  # 不需要 retain_graph=True，除非你确实需要多次反向传播
            Dopt.step()

            iter_num = iter_num + 1

            if iter_num % 500 == 0:
                logging.info('Fold {}, iteration {}: loss: {:.3f}, loss_vit:{:.3f}, '
                             .format(fold_id, iter_num,
                                     loss, loss_vit
                                     ))

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            if iter_num % 2000 == 0:
                model.eval()
                dice_all, std_all, metric_all_cases = test_util.validation_all_case(model,
                                                                                    num_classes=num_classes,
                                                                                    dataset=args.datasets,
                                                                                    image_list=test_list,
                                                                                    patch_size=patch_size,
                                                                                    stride_xy=16,
                                                                                  stride_z=16)

                print(f'metric_all_cases:{metric_all_cases}')
                dice_avg = dice_all.mean()



                logging.info('iteration {}, '
                             'average DSC: {:.3f}, '
                             'vessel: {:.3f}, '
                             .format(iter_num,
                                     dice_avg,
                                     dice_all[0],
                                     ))

                if dice_avg > best_dice_avg:

                    best_dice_avg = dice_avg
                    save_mode_path = os.path.join(snapshot_path,
                                                  'vnet_iter_{}_dice_{}.pth'.format(iter_num, best_dice_avg))
                    save_mode_path2 = os.path.join(snapshot_path,
                                                   'unetr_iter_{}_dice_{}.pth'.format(iter_num, best_dice_avg))
                    save_best_path1 = os.path.join(snapshot_path, '{}_best_model.pth'.format('model'))
                    save_best_path2 = os.path.join(snapshot_path, '{}_best_model.pth'.format('model1'))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model1.state_dict(), save_mode_path2)
                    torch.save(model.state_dict(), save_best_path1)
                    torch.save(model1.state_dict(), save_best_path2)
                    logging.info("save best model to {}".format(save_mode_path))

                model.train()

        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    logging.getLogger().removeHandler(handler)
    logging.getLogger().removeHandler(sh)

    return metric_all_cases


if __name__ == "__main__":

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code', shutil.ignore_patterns(['.git', '__pycache__']))

    with open(txt_path, 'r') as f:
        total_list = f.readlines()
    total_list = ["{}".format(item.strip()) for item in total_list]

    train_list, test_list = total_list[:args.max_train_samples], total_list[args.max_train_samples:]
    np.random.shuffle(train_list)
    metric_final = train(train_list, test_list)

    # 12x4x13
    # 4x13, 4x13
    metric_mean, metric_std = np.mean(metric_final, axis=0), np.std(metric_final, axis=0)

    metric_save_path = os.path.join(snapshot_path, 'metric_final_{}_{}.npy'.format(args.dataset_name, args.exp))
    np.save(metric_save_path, metric_final)

    handler, sh = config_log(snapshot_path, 'total_metric')
    logging.info('Final Average DSC:{:.4f}, jc: {:.4f}, ASD: {:.4f}, NSD:{:.4f}'
                 .format(metric_mean[0].mean(), metric_mean[1].mean(), metric_mean[2].mean(),metric_mean[3].mean()
                         ))

    logging.getLogger().removeHandler(handler)
    logging.getLogger().removeHandler(sh)
