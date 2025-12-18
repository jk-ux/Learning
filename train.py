# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.cuda.amp import autocast,GradScaler
from datasets.make_dataloader import make_dataset
import torch.backends.cudnn as cudnn
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import time
import os
import yaml
from shutil import copyfile
from utils import get_model_list, load_network, save_network, make_weights_for_balanced_classes
from optimizers.make_optimizer import make_optimizer
from losses.triplet_loss import Tripletloss,TripletLoss
from losses.cal_loss import cal_kl_loss,cal_loss,cal_triplet_loss
from models.model import make_model
# 添加 TensorBoard 相关导入
from torch.utils.tensorboard import SummaryWriter

version =  torch.__version__
#fp16
try:
    from apex.fp16_utils import *
    from apex import amp
except ImportError:
    print('This is not an error. If you want to use low precision, i.e., fp16, please install the apex with cuda support (https://github.com/NVIDIA/apex) and update pytorch to 1.0')
######################################################################
# Options
# --------
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--name',default='convnext_tri', type=str, help='output model name')
parser.add_argument('--data_dir',default='../data/train',type=str, help='training dir path')
parser.add_argument('--train_all', action='store_false', help='use all training data' )
parser.add_argument('--color_jitter', action='store_false', help='use color jitter in training' )
parser.add_argument('--batchsize', default=8, type=int, help='batchsize')
parser.add_argument('--pad', default=10, type=int, help='padding')
parser.add_argument('--h', default=252, type=int, help='height (14的整数倍，适配DINOv2)')
parser.add_argument('--w', default=252, type=int, help='width (14的整数倍，适配DINOv2)')
parser.add_argument('--views', default=2, type=int, help='the number of views')
parser.add_argument('--erasing_p', default=0.5, type=float, help='Random Erasing probability, in [0,1]')
parser.add_argument('--warm_epoch', default=0, type=int, help='the first K epoch that needs warm up')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--droprate', default=0.5, type=float, help='drop rate')
parser.add_argument('--DA', action='store_false', help='use Color Data Augmentation' )
parser.add_argument('--resnet', action='store_true', default=False, help='use resnet' )
parser.add_argument('--share', action='store_false',default=True, help='share weight between different view' )
parser.add_argument('--resume', action='store_true', help='use resume trainning' )
parser.add_argument('--autocast', action='store_true',default=True, help='use mix precision' )
parser.add_argument('--fp16', action='store_true', help='use float16 instead of float32, which will save about 50% memory' )
parser.add_argument('--block', default=2, type=int, help='')
parser.add_argument('--kl_loss', action='store_true',default=False, help='kl_loss' )
parser.add_argument('--triplet_loss', default=0.3, type=float, help='')
parser.add_argument('--sample_num', default=1, type=float, help='')
parser.add_argument('--model', default='convnext_small_22k_224', type=str, metavar='MODEL', help='Name of model to train')
parser.add_argument('--epochs', default=200, type=int, help='' )
parser.add_argument('--fname', default='train.txt', type=str, help='Name of log txt')
parser.add_argument('--steps', default=[80,120], type=int, help='' )
# === 新添加的 DINOv2 参数 ===
parser.add_argument('--dinov2', action='store_true', help='use DINOv2 backbone instead of ResNet/ConvNeXt')
parser.add_argument('--dinov2_size', default='vitb14', type=str,choices=['vits14', 'vitb14', 'vitl14', 'vitg14'],help='DINOv2 model size: vits14(384d), vitb14(768d), vitl14(1024d), vitg14(1536d)')
parser.add_argument('--use_cls_token', action='store_true',help='use CLS token as global feature (faster) instead of spatial pooling')
parser.add_argument('--freeze_backbone', action='store_true',help='freeze DINOv2 backbone parameters (only train classifiers)')
parser.add_argument('--dinov2_dropout', default=0.5, type=float,help='dropout rate for DINOv2 classifier heads')
# === TensorBoard 参数 ===
parser.add_argument('--tensorboard', action='store_true', default=True, help='enable TensorBoard logging')
parser.add_argument('--tb_log_dir', default='./tb_logs', type=str, help='TensorBoard log directory')
opt = parser.parse_args()

dir_name = os.path.join('./model',opt.name)

if not opt.resume:
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    copyfile('./train.py', dir_name+'/train.py')
    if opt.dinov2:
        copyfile('models/dinov2_backbone.py', dir_name + '/dinov2_backbone.py')
        copyfile('models/model.py', dir_name + '/model.py')
    else:
        copyfile('models/ConvNext/backbones/model_convnext.py', dir_name + '/model.py')

# === 初始化 TensorBoard SummaryWriter ===
if opt.tensorboard:
    # 创建 TensorBoard 日志目录（按模型名称和时间戳区分）
    import datetime
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    tb_log_path = os.path.join(opt.tb_log_dir, f'{opt.name}_{timestamp}')
    os.makedirs(tb_log_path, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_log_path)
    print(f"TensorBoard logging enabled. Logs saved to: {tb_log_path}")
else:
    writer = None

if opt.resume:
    model, opt, start_epoch = load_network(opt.name, opt)
else:
    start_epoch = 0

str_ids = opt.gpu_ids.split(',')
gpu_ids = []
for str_id in str_ids:
    gid = int(str_id)
    if gid >=0:
        gpu_ids.append(gid)

# set gpu ids
if len(gpu_ids)>0:
    torch.cuda.set_device(gpu_ids[0])
    cudnn.benchmark = True
######################################################################
# Load Data
# ---------
dataloaders,class_names,dataset_sizes = make_dataset(opt)
opt.nclasses = len(class_names)
print(dataset_sizes)
if not opt.resume:
    with open(os.path.join('model',opt.name,opt.fname),'a',encoding='utf-8') as f:
        text = str(dataset_sizes)+'\n'
        f.write(text)
use_gpu = torch.cuda.is_available()

######################################################################
# Training the model
# ------------------
y_loss = {} # loss history
y_loss['train'] = []
y_err = {}
y_err['train'] = []

def train_model(model, opt, model_test, optimizer, scheduler, num_epochs=25):
    since = time.time()

    scaler = GradScaler()
    criterion = nn.CrossEntropyLoss()
    loss_kl = nn.KLDivLoss(reduction='batchmean')
    triplet_loss = Tripletloss(margin=opt.triplet_loss)

    min_loss = 1.0

    warm_up = 0.1 # We start from the 0.1*lrRate
    warm_iteration = round(dataset_sizes['satellite']/opt.batchsize)*opt.warm_epoch # first 5 epoch

    # 记录迭代步数（用于 TensorBoard 按步数监控）
    global_step = 0

    for epoch in range(num_epochs-start_epoch):
        epoch = epoch + start_epoch
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        with open(os.path.join('model',opt.name,opt.fname),'a',encoding='utf-8') as f:
            text = str('Epoch {}/{}'.format(epoch, num_epochs - 1))+'\n'+('-' * 10)+'\n'
            f.write(text)
        
        # 注意：这里 phase 只有 'train'（根据你的代码逻辑），且 dataloaders 是单一对象
        phase = 'train'
        model.train(True)  # 只训练阶段（你的代码中没有验证阶段逻辑）

        running_cls_loss = 0.0
        running_triplet = 0.0
        running_kl_loss = 0.0
        running_loss = 0.0
        running_corrects = 0.0
        running_corrects2 = 0.0
        running_corrects3 = 0.0
      
        for data,data3 in dataloaders:
            # satallite (data) # street (data2) # drone (data3)
            loss = 0.0
            # 正确解包：每个视图是 (图像 tensor, 标签 tensor)
            inputs, labels = data          # satellite 视图：图像+标签
            # inputs2, labels2 = data2       # street 视图：图像+标签
            inputs3, labels3 = data3       # drone 视图：图像+标签
            
            now_batch_size,c,h,w = inputs.shape
            if now_batch_size<opt.batchsize: # skip the last batch
                continue
            
            # 数据移到 GPU
            if use_gpu:
                inputs = inputs.cuda(non_blocking=True)
                # inputs2 = inputs2.cuda(non_blocking=True)
                inputs3 = inputs3.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
                # labels2 = labels2.cuda(non_blocking=True)
                labels3 = labels3.cuda(non_blocking=True)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            with autocast():
                if opt.views == 2:
                    outputs, outputs2 = model(inputs, inputs3)  # satellite 和 drone
                elif opt.views == 3:
                    outputs, outputs3, outputs2 = model(inputs, inputs2, inputs3)# satellite, street, drone
            
            f_triplet_loss = torch.tensor(0.0).cuda()
            if opt.triplet_loss>0:
                # 适配 DINOv2 输出：outputs = (预测列表, 特征)
                if opt.dinov2:
                    features = outputs[1]    # DINOv2 输出的全局特征
                    features2 = outputs2[1]
                    outputs = outputs[0]    # DINOv2 输出的分类预测
                    outputs2 = outputs2[0]
                else:
                    features = outputs[1]
                    features2 = outputs2[1]
                    outputs = outputs[0]
                    outputs2 = outputs2[0]
                
                split_num = opt.batchsize//opt.sample_num
                f_triplet_loss = cal_triplet_loss(features,features2,labels,triplet_loss,split_num)

            # 处理预测结果（多分类器列表）
            if isinstance(outputs,list):
                preds = []
                preds2 = []
                batch_acc1 = 0.0  # 批次级 satellite 准确率
                batch_acc2 = 0.0  # 批次级 drone 准确率
                for out,out2 in zip(outputs,outputs2):
                    pred1 = torch.max(out.data,1)[1]
                    pred2 = torch.max(out2.data,1)[1]
                    preds.append(pred1)
                    preds2.append(pred2)
                    # 计算单个分类器的批次准确率
                    batch_acc1 += float(torch.sum(pred1 == labels.data)) / now_batch_size
                    batch_acc2 += float(torch.sum(pred2 == labels3.data)) / now_batch_size
                # 平均所有分类器的批次准确率
                batch_acc1 /= len(preds)
                batch_acc2 /= len(preds2)
            else:
                _, pred1 = torch.max(outputs.data, 1)
                _, pred2 = torch.max(outputs2.data, 1)
                preds = pred1
                preds2 = pred2
                # 计算批次准确率
                batch_acc1 = float(torch.sum(pred1 == labels.data)) / now_batch_size
                batch_acc2 = float(torch.sum(pred2 == labels3.data)) / now_batch_size

            kl_loss = torch.tensor(0.0).cuda()
            if opt.views == 2:
                # 分类损失：satellite 对应 labels，drone 对应 labels3
                cls_loss = cal_loss(outputs, labels, criterion) + cal_loss(outputs2, labels3, criterion)
                if opt.kl_loss:
                    kl_loss = cal_kl_loss(outputs, outputs2, loss_kl)
            elif opt.views == 3:
                # 适配 DINOv2 输出
                if opt.dinov2:
                    outputs3 = outputs3[0]
                else:
                    outputs3 = outputs3[0]
                
                if isinstance(outputs, list):
                    preds3 = []
                    for out3 in outputs3:
                        preds3.append(torch.max(out3.data, 1)[1])
                    cls_loss = cal_loss(outputs, labels, criterion) + cal_loss(outputs2, labels3, criterion) + cal_loss(outputs3,labels2,criterion)
                    loss += cls_loss
                else:
                    _, preds3 = torch.max(outputs3.data, 1)
                    cls_loss = cal_loss(outputs, labels, criterion) + cal_loss(outputs2, labels3, criterion) + cal_loss(outputs3,labels2,criterion)
                    loss += cls_loss

            # 总损失
            loss = kl_loss + cls_loss + f_triplet_loss
            
            # 热身阶段
            if epoch<opt.warm_epoch and phase == 'train': 
                warm_up = min(1.0, warm_up + 0.9 / warm_iteration)
                loss *= warm_up

            # 反向传播
            if phase == 'train':
                if opt.autocast:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            # 统计损失
            if torch.__version__ >= '1.0':
                batch_total_loss = loss.item()
                batch_cls_loss = cls_loss.item()
                batch_triplet_loss = f_triplet_loss.item()
                batch_kl_loss = kl_loss.item()
                
                running_loss += batch_total_loss * now_batch_size
                running_cls_loss += batch_cls_loss * now_batch_size
                running_triplet += batch_triplet_loss * now_batch_size
                running_kl_loss += batch_kl_loss * now_batch_size
            else:
                batch_total_loss = loss.data[0]
                batch_cls_loss = cls_loss.data[0]
                batch_triplet_loss = f_triplet_loss.data[0]
                batch_kl_loss = kl_loss.data[0]
                
                running_loss += batch_total_loss * now_batch_size
                running_cls_loss += batch_cls_loss * now_batch_size
                running_triplet += batch_triplet_loss * now_batch_size
                running_kl_loss += batch_kl_loss * now_batch_size

            # 统计准确率
            if isinstance(preds,list) and isinstance(preds2,list):
                running_corrects += sum([float(torch.sum(pred == labels.data)) for pred in preds])/len(preds)
                if opt.views==2:
                    running_corrects2 += sum([float(torch.sum(pred == labels3.data)) for pred in preds2]) / len(preds2)
                else:
                    running_corrects2 += sum([float(torch.sum(pred == labels3.data)) for pred in preds2])/len(preds2)
            else:
                running_corrects += float(torch.sum(preds == labels.data))
                if opt.views == 2:
                    running_corrects2 += float(torch.sum(preds2 == labels3.data))
                else:
                    running_corrects2 += float(torch.sum(preds2 == labels3.data))
            if opt.views == 3:
                if isinstance(preds,list) and isinstance(preds2,list):
                    running_corrects3 += sum([float(torch.sum(pred == labels2.data)) for pred in preds3])/len(preds3)
                else:
                    running_corrects3 += float(torch.sum(preds3 == labels2.data))

            # === TensorBoard：按迭代步数记录批次级指标 ===
            if opt.tensorboard and writer is not None:
                writer.add_scalar('Train/Batch_Total_Loss', batch_total_loss, global_step)
                writer.add_scalar('Train/Batch_Cls_Loss', batch_cls_loss, global_step)
                writer.add_scalar('Train/Batch_Triplet_Loss', batch_triplet_loss, global_step)
                writer.add_scalar('Train/Batch_KL_Loss', batch_kl_loss, global_step)
                writer.add_scalar('Train/Batch_Satellite_Acc', batch_acc1, global_step)
                writer.add_scalar('Train/Batch_Drone_Acc', batch_acc2, global_step)
                
                # 记录学习率
                if opt.dinov2 and not opt.freeze_backbone:
                    writer.add_scalar('Train/LR_Backbone', optimizer.state_dict()['param_groups'][0]['lr'], global_step)
                    writer.add_scalar('Train/LR_Other', optimizer.state_dict()['param_groups'][1]['lr'], global_step)
                else:
                    writer.add_scalar('Train/LR', optimizer.state_dict()['param_groups'][0]['lr'], global_step)
            
            global_step += 1  # 步数递增

        # 计算 epoch 平均指标
        epoch_cls_loss = running_cls_loss / dataset_sizes['satellite']
        epoch_kl_loss = running_kl_loss / dataset_sizes['satellite']
        epoch_triplet_loss = running_triplet / dataset_sizes['satellite']
        epoch_loss = running_loss / dataset_sizes['satellite']
        epoch_acc = running_corrects / dataset_sizes['satellite']
        epoch_acc2 = running_corrects2 / dataset_sizes['satellite']

        # 学习率日志
        if opt.dinov2 and not opt.freeze_backbone:
            lr_backbone = optimizer.state_dict()['param_groups'][0]['lr']
            lr_other = optimizer.state_dict()['param_groups'][1]['lr']
        elif opt.dinov2 and opt.freeze_backbone:
            lr_backbone = 0.0
            lr_other = optimizer.state_dict()['param_groups'][0]['lr']
        else:
            lr_backbone = optimizer.state_dict()['param_groups'][0]['lr']
            lr_other = optimizer.state_dict()['param_groups'][1]['lr']
        
        # 打印日志
        if opt.views == 2:
            print('{} Loss: {:.4f} Cls_Loss:{:.4f} KL_Loss:{:.4f} Triplet_Loss {:.4f} Satellite_Acc: {:.4f}  Drone_Acc: {:.4f} lr_backbone:{:.6f} lr_other {:.6f}'
                  .format(phase, epoch_loss,epoch_cls_loss,epoch_kl_loss,epoch_triplet_loss, epoch_acc, epoch_acc2,lr_backbone,lr_other))
            with open(os.path.join('model', opt.name, opt.fname), 'a', encoding='utf-8') as f:
                text = str('{} Loss: {:.4f} Cls_Loss:{:.4f} KL_Loss:{:.4f} Triplet_Loss {:.4f} Satellite_Acc: {:.4f}  Drone_Acc: {:.4f} lr_backbone:{:.6f} lr_other {:.6f}'
                           .format(phase, epoch_loss,epoch_cls_loss,epoch_kl_loss,epoch_triplet_loss, epoch_acc, epoch_acc2,lr_backbone,lr_other)) + '\n'
                f.write(text)
        elif opt.views == 3:
            epoch_acc3 = running_corrects3 / dataset_sizes['satellite']
            print('{} Loss: {:.4f} Cls_Loss:{:.4f} KL_Loss:{:.4f} Triplet_Loss {:.4f} Satellite_Acc: {:.4f}  Street_Acc: {:.4f} Drone_Acc: {:.4f} lr_backbone:{:.6f} lr_other {:.6f}'
                  .format(phase,epoch_loss,epoch_cls_loss,epoch_kl_loss,epoch_triplet_loss, epoch_acc, epoch_acc3, epoch_acc2, lr_backbone,lr_other))
            with open(os.path.join('model', opt.name, opt.fname), 'a', encoding='utf-8') as f:
                text = str('{} Loss: {:.4f} Cls_Loss:{:.4f} KL_Loss:{:.4f} Triplet_Loss {:.4f} Satellite_Acc: {:.4f}  Street_Acc: {:.4f} Drone_Acc: {:.4f} lr_backbone:{:.6f} lr_other {:.6f}'
                           .format(phase,epoch_loss,epoch_cls_loss,epoch_kl_loss,epoch_triplet_loss, epoch_acc, epoch_acc3, epoch_acc2, lr_backbone,lr_other)) + '\n'
                f.write(text)

        # === TensorBoard：按 epoch 记录全局指标 ===
        if opt.tensorboard and writer is not None:
            writer.add_scalar('Train/Epoch_Total_Loss', epoch_loss, epoch)
            writer.add_scalar('Train/Epoch_Cls_Loss', epoch_cls_loss, epoch)
            writer.add_scalar('Train/Epoch_Triplet_Loss', epoch_triplet_loss, epoch)
            writer.add_scalar('Train/Epoch_KL_Loss', epoch_kl_loss, epoch)
            writer.add_scalar('Train/Epoch_Satellite_Acc', epoch_acc, epoch)
            writer.add_scalar('Train/Epoch_Drone_Acc', epoch_acc2, epoch)
            writer.add_scalar('Train/Epoch_Avg_Acc', (epoch_acc + epoch_acc2) / 2, epoch)  # 平均准确率
            
            # 记录学习率（按 epoch）
            writer.add_scalar('Train/Epoch_LR_Backbone', lr_backbone, epoch)
            writer.add_scalar('Train/Epoch_LR_Other', lr_other, epoch)
            
            # （可选）记录模型参数分布（每5个epoch记录一次，避免日志过大）
            if epoch % 5 == 0:
                for name, param in model.named_parameters():
                    if 'backbone' in name and param.requires_grad:
                        writer.add_histogram(f'Params/{name}', param.data.cpu().numpy(), epoch)
                        if param.grad is not None:
                            writer.add_histogram(f'Grads/{name}', param.grad.data.cpu().numpy(), epoch)

        # 记录损失曲线
        y_loss[phase].append(epoch_loss)
        y_err[phase].append(1.0-epoch_acc)        
        
        # 学习率调度
        scheduler.step()
        
        # 保存最佳模型
        if epoch >= 90 and epoch_loss < min_loss:
            save_network(model, opt.name, epoch)
            min_loss = epoch_loss

        # 打印耗时
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print()
        with open(os.path.join('model',opt.name,opt.fname), 'a', encoding='utf-8') as f:
            text = str('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60)) + '\n'
            f.write(text)

    # 训练结束后关闭 TensorBoard Writer
    if opt.tensorboard and writer is not None:
        writer.close()
        print(f"TensorBoard logging finished. Logs saved to: {tb_log_path}")

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    with open(os.path.join('model',opt.name,opt.fname), 'a', encoding='utf-8') as f:
        text = str('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60)) + '\n'
        f.write(text)

    return model
  
######################################################################
# Draw Curve
#---------------------------
x_epoch = []
fig = plt.figure()
ax0 = fig.add_subplot(121, title="loss")
ax1 = fig.add_subplot(122, title="top1err")
def draw_curve(current_epoch):
    x_epoch.append(current_epoch)
    ax0.plot(x_epoch, y_loss['train'], 'bo-', label='train')
    ax0.plot(x_epoch, y_loss['val'], 'ro-', label='val')
    ax1.plot(x_epoch, y_err['train'], 'bo-', label='train')
    ax1.plot(x_epoch, y_err['val'], 'ro-', label='val')
    if current_epoch == 0:
        ax0.legend()
        ax1.legend()
    fig.savefig( os.path.join('./model',opt.name,'train.jpg'))

######################################################################
# Finetuning the convnet
# ----------------------
if not opt.resume:
    model = make_model(opt)
    # save opts
    with open('%s/opts.yaml'%dir_name,'a') as fp:
        yaml.dump(vars(opt), fp, default_flow_style=False)

# For resume:
if start_epoch>=40:
    opt.lr = opt.lr*0.01

# 优化器配置
if opt.dinov2:
    if opt.freeze_backbone:
        print("Freezing DINOv2 backbone, only training classifiers")
        params_to_optimize = []
        for name, param in model.named_parameters():
            if 'backbone' not in name and param.requires_grad:
                params_to_optimize.append(param)
        
        optimizer_ft = torch.optim.SGD(
            params_to_optimize,
            lr=opt.lr,
            weight_decay=5e-4,
            momentum=0.9,
            nesterov=True
        )
        
        exp_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer_ft,
            milestones=opt.steps,
            gamma=0.1
        )
    else:
        print("Training DINOv2 backbone with lower learning rate")
        backbone_params = []
        other_params = []
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                if 'backbone' in name:
                    backbone_params.append(param)
                else:
                    other_params.append(param)
        
        optimizer_ft = torch.optim.SGD([
            {'params': backbone_params, 'lr': opt.lr * 0.1},
            {'params': other_params, 'lr': opt.lr}
        ], weight_decay=5e-4, momentum=0.9, nesterov=True)
        
        exp_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer_ft,
            milestones=opt.steps,
            gamma=0.1
        )
else:
    optimizer_ft, exp_lr_scheduler = make_optimizer(model, opt)

######################################################################
# Train and evaluate
# ^^^^^^^^^^^^^^^^^^
model = model.cuda()
if opt.fp16:
    model, optimizer_ft = amp.initialize(model, optimizer_ft, opt_level = "O1")

model_test = None
num_epochs = opt.epochs

model = train_model(model, opt, model_test, optimizer_ft, exp_lr_scheduler,
                       num_epochs=num_epochs)
