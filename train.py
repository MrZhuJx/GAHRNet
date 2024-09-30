import json
import os
import datetime
import torch
from torch.utils import data
import numpy as np

from my_dataset import DataKeyPoint2,DataKeyPoint55,DataKeyPoint22
import distributed_utils as utils
from loss import KpLoss,KpLoss3,KpLoss1
import math
import gc

from model import GAHRNet



result = []
LOS = []
def train_one_epoch(model, optimizer, data_loader, device, epoch,
                    print_freq=100, warmup=False, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0 and warmup is True:  # 当训练第一轮（epoch=0）时，启用warmup训练方式，可理解为热身训练
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    mse = KpLoss()
    
    mloss = torch.zeros(1).to(device)  # mean losses
    for i, [images, targets, points, targets1, points1] in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # with torch.no_grad():
        #     images = torch.stack([image.to(device) for image in images])
        # print(targets.size())
        images = torch.stack([image.to(device) for image in images])
        # 混合精度训练上下文管理器，如果在CPU环境中不起任何作用
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            results = model(images)
            # results,results2 = model(images)
            # print(results.size())
            # losses = mse(results, targets, results2, target2)
            losses = mse(results, targets)

        # reduce losses over all GPUs for logging purpose
        loss_dict_reduced = utils.reduce_dict({"losses": losses})
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()
        # 记录训练损失
        mloss = (mloss * i + loss_value) / (i + 1)  # update mean losses

        if not math.isfinite(loss_value):  # 当计算的损失为无穷大时停止训练
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:  # 第一轮使用warmup训练方式
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced)
        now_lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=now_lr)
        # del results, losses
        # gc.collect()
    LOS.append(mloss)

    
    return mloss, now_lr

def euclidean_distance(tensor1, tensor2, dim, device):
   
    # 计算差值的平方
    squared_difference = (tensor1.to(device) - tensor2.to(device))**2
    
    # 沿着指定的维度求和
    sum_along_dim = squared_difference.sum(dim)
    
    # 取平方根
    distance = torch.sqrt(sum_along_dim)
    
    return distance


def get_max_preds(batch_heatmaps):

    batch_size, num_joints, h, w = batch_heatmaps.shape
    heatmaps_reshaped = batch_heatmaps.reshape(batch_size, num_joints, -1)
    maxvals, idx = torch.max(heatmaps_reshaped, dim=2)

    maxvals = maxvals.unsqueeze(dim=-1)
    idx = idx.float()

    preds = torch.zeros((batch_size, num_joints, 2)).to(batch_heatmaps)

    preds[:, :, 0] = idx % w  # column 对应最大值的x坐标
    preds[:, :, 1] = torch.floor(idx / w)  # row 对应最大值的y坐标

    pred_mask = torch.gt(maxvals, 0.0).repeat(1, 1, 2).float().to(batch_heatmaps.device)

    preds *= pred_mask
    return preds 

def main(args):
    
    # device = torch.device("cpu")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Using {} device training.".format(device.type))

    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

   

    fixed_size = args.fixed_size
    # heatmap_hw = (args.fixed_size[0] // 4, args.fixed_size[1] // 4)
    
    data_root = args.data_path

 
    train_dataset = DataKeyPoint55(data_root, "train",fixed_size=args.fixed_size)

    
    batch_size = args.batch_size
    
    nw = 0  # number of workers
    # print('Using %g dataloader workers' % nw)

    train_data_loader = data.DataLoader(train_dataset,
                                        batch_size=batch_size,
                                        shuffle=True,
                                        pin_memory=True)

    val_dataset = DataKeyPoint55(data_root, "test", fixed_size=args.fixed_size)
    val_data_loader = data.DataLoader(val_dataset,
                                      batch_size=1,
                                      shuffle=False,
                                      pin_memory=True)


    model = GAHRNet(num_joints=29,base_channel=32)


    model.to(device)

    # define optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params,
                                  lr=args.lr,
                                  weight_decay=args.weight_decay)

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    # learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)

    # 如果指定了上次训练保存的权重文件地址，则接着上次结果接着训练
    if args.resume != "":
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp and "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        print("the training process from epoch{}...".format(args.start_epoch))

    train_loss = []
    learning_rate = []
    val_map = []

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch, printing every 50 iterations
        mean_loss, lr = train_one_epoch(model, optimizer, train_data_loader,
                                              device=device, epoch=epoch,
                                              print_freq=50, warmup=True,
                                              scaler=scaler)
        train_loss.append(mean_loss.item())
        learning_rate.append(lr)

        # update the learning rate
        lr_scheduler.step()
     
        model.eval()
        L = []
        
        metric_logger = utils.MetricLogger(delimiter="  ")
    


        for i, [images, targets, points, targets1, points1] in enumerate(metric_logger.log_every(val_data_loader,50)):
            images = torch.stack([image.to(device) for image in images])
            results = model(images)
            # results, results2 = model(images)

            point1 = get_max_preds(results)[0].to(device)
            # point11 = get_max_preds(results1)[0].to(device)
            y = torch.tensor([1935 / 3360,2400 / 3360]).to(device)
            point1 = point1 * y
            point2 = points[0].to(device)
            point2 = point2 * y
            y = euclidean_distance(point1,point2,1,device) 
            y2 = y.sum(dim=0) / 29.0
            L.append(y2)

            
            

        l = 0
        for i in L:
            l += i
        print('SDR=',l.item() / 171.0)
        result.append(l.item() / 171.0)

        save_files = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch}
        if args.amp:
            save_files["scaler"] = scaler.state_dict()
        torch.save(save_files, "/home/jxzhu/GAHRNet/weight/model-{}.pth".format(l.item() / 171.0))

    with open('/home/jxzhu/GAHRNet/sdr.txt', 'w') as f:
        for res in result:
            f.write(str(res) + '\n')
    with open('/home/jxzhu/GAHRNet/losses.txt', 'w') as f:
        for loss in LOS:
            f.write(str(loss) + '\n')
        



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__)

    # 训练设备类型
    parser.add_argument('--device', default='cuda:1', help='device')
    parser.add_argument('--data-path', default='/home/jxzhu/MBSI/data', help='dataset')
    parser.add_argument('--fixed-size', default=[672, 672], nargs='+', type=int, help='input size')
    # keypoints点数
    parser.add_argument('--num-joints', default=29, type=int, help='num_joints')
    # 文件保存地址
    parser.add_argument('--output-dir', default='/home/jxzhu/NewNet/weight1', help='path where to save')
    # 若需要接着上次训练，则指定上次训练保存权重文件地址
    parser.add_argument('--resume', default='', type=str, help='resume from checkpoint')
    # 指定接着从哪个epoch数开始训练
    parser.add_argument('--start-epoch', default=0, type=int, help='start epoch')
    # 训练的总epoch数
    parser.add_argument('--epochs', default=300, type=int, metavar='N',
                        help='number of total epochs to run')
    # 针对torch.optim.lr_scheduler.MultiStepLR的参数
    parser.add_argument('--lr-steps', default=[170, 200], nargs='+', type=int, help='decrease lr every step-size epochs')
    # 针对torch.optim.lr_scheduler.MultiStepLR的参数
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    # 学习率
    parser.add_argument('--lr', default=0.001, type=float,
                        help='initial learning rate, 0.02 is the default value for training '
                             'on 8 gpus and 2 images_per_gpu')
    # AdamW的weight_decay参数
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    # 训练的batch size
    parser.add_argument('--batch-size', default=8, type=int, metavar='N',
                        help='batch size when training.')
    # 是否使用混合精度训练(需要GPU支持混合精度)
    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training")

    args = parser.parse_args()
    print(args)

    # 检查保存权重文件夹是否存在，不存在则创建
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    main(args)
