# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from tqdm import tqdm
import time
import os
import scipy.io
import yaml
import math
from torchvision.transforms import InterpolationMode
from utils import load_network
from datasets.queryDataset import Dataset_query,Query_transforms

#fp16
try:
    from apex.fp16_utils import *
except ImportError:
    print('This is not an error. If you want to use low precision, i.e., fp16, please install the apex with cuda support (https://github.com/NVIDIA/apex) and update pytorch to 1.0')

######################################################################
# Options
# --------

parser = argparse.ArgumentParser(description='Testing')
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--which_epoch',default='last', type=str, help='0,1,2,3...or last')
parser.add_argument('--test_dir',default='../data/test',type=str, help='./test_data')
parser.add_argument('--name', default='convnext_tri', type=str, help='save model path')
parser.add_argument('--part', default='', type=str, help='test drone distance')
parser.add_argument('--mode', default=2, type=int, help='2:drone->satellite   1:satellite->drone')
parser.add_argument('--padmode',default='', type=str,help='bp or fp')
parser.add_argument('--pad', default=0, type=int, help='')
parser.add_argument('--pool', default='avg', type=str, help='avg|max')
parser.add_argument('--batchsize', default=8, type=int, help='batchsize')
parser.add_argument('--h', default=256, type=int, help='height')
parser.add_argument('--w', default=256, type=int, help='width')
parser.add_argument('--multi', action='store_true', help='use multiple query' )
parser.add_argument('--fp16', action='store_true', help='use fp16.' )
parser.add_argument('--ms',default='1', type=str,help='multiple_scale: e.g. 1 1,1.1  1,1.1,1.2')
parser.add_argument('--model', default='convnext_tiny', type=str, metavar='MODEL',
                    help='Name of model to train')
# ⭐ 新增：SUES-200 多高度测试参数
parser.add_argument('--dataset', default='university', type=str, 
                    choices=['university', 'sues200'],
                    help='dataset type: university or sues200')
parser.add_argument('--heights', default='150,200,250,300', type=str,
                    help='test heights for SUES-200 (comma-separated)')
opt = parser.parse_args()

###############################################################################
# 加载训练配置（支持 DINOv2）
###############################################################################
print("="*70)
print("🔍 Loading training configuration...")
print("="*70)

yaml.warnings({'YAMLLoadWarning': False})
config_path = os.path.join('./model',opt.name,'opts.yaml')

if not os.path.exists(config_path):
    print(f"❌ Error: Config file not found at {config_path}")
    print("   Please make sure the model was trained and opts.yaml exists.")
    exit(1)

with open(config_path, 'r') as stream:
    config = yaml.load(stream, Loader=yaml.FullLoader)

# === 加载基础配置 ===
opt.fp16 = config.get('fp16', False)
opt.fname = 'test.txt'
opt.views = config.get('views', 2)
opt.block = config.get('block', 2)
opt.share = config.get('share', True)

if 'resnet' in config:
    opt.resnet = config['resnet']
if 'h' in config:
    opt.h = config['h']
    opt.w = config['w']
if 'nclasses' in config:
    opt.nclasses = config['nclasses']
else:
    opt.nclasses = 729

# === 加载 DINOv2 配置（关键部分）===
opt.dinov2 = config.get('dinov2', False)
if opt.dinov2:
    opt.dinov2_size = config.get('dinov2_size', 'vitb14')
    opt.use_cls_token = config.get('use_cls_token', False)
    opt.freeze_backbone = config.get('freeze_backbone', False)
    opt.dinov2_dropout = config.get('dinov2_dropout', 0.5)
    opt.droprate = opt.dinov2_dropout
    
    print("✅ DINOv2 model configuration loaded:")
    print(f"   - Model size: {opt.dinov2_size}")
    print(f"   - Use CLS token: {opt.use_cls_token}")
    print(f"   - Backbone frozen during training: {opt.freeze_backbone}")
    print(f"   - Dropout: {opt.dinov2_dropout}")
else:
    opt.dinov2_size = None
    opt.use_cls_token = False
    opt.freeze_backbone = False
    opt.dinov2_dropout = 0.5
    opt.droprate = config.get('droprate', 0.5)
    
    print("✅ ConvNeXt model configuration loaded:")
    print(f"   - Model: {config.get('model', 'convnext_small_22k_224')}")
    print(f"   - Droprate: {opt.droprate}")
    print(f"   - Share weights: {opt.share}")

if 'model' in config:
    opt.model = config['model']

print(f"   - Dataset: {opt.dataset}")
print(f"   - Views: {opt.views}")
print(f"   - Image size: {opt.h}x{opt.w}")
print(f"   - Classes: {opt.nclasses}")
print("="*70)

###############################################################################
# GPU 设置
###############################################################################

str_ids = opt.gpu_ids.split(',')
gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >=0:
        gpu_ids.append(id)

print('We use the scale: %s'%opt.ms)
str_ms = opt.ms.split(',')
ms = []
for s in str_ms:
    s_f = float(s)
    ms.append(math.sqrt(s_f))

if len(gpu_ids)>0:
    torch.cuda.set_device(gpu_ids[0])
    cudnn.benchmark = True

######################################################################
# 数据变换
######################################################################

data_transforms = transforms.Compose([
        transforms.Resize((opt.h, opt.w), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

data_query_transforms = transforms.Compose([
        transforms.Resize((opt.h, opt.w), interpolation=3),
        Query_transforms(pad=opt.pad, size=opt.w),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

use_gpu = torch.cuda.is_available()

######################################################################
# 特征提取函数
######################################################################

def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1, -1, -1).long()
    # ⭐ 添加这行：将索引移到与图像相同的设备
    inv_idx = inv_idx.to(img.device)
    img_flip = img.index_select(3, inv_idx)
    return img_flip

def which_view(name):
    if 'satellite' in name:
        return 1
    elif 'street' in name:
        return 2
    elif 'drone' in name:
        return 3
    else:
        print('unknown view')
    return -1

def extract_feature(model, dataloader, view_index=None):
    features = []
    model.eval()

    for i, (img, path) in enumerate(tqdm(dataloader, ncols=80)):
        if i == 0:
            print(f"[DEBUG] First image path: {path[0]}")
        img = img.cuda()
        ff = None

        for i in range(2):
            img_i = img
            if i == 1:
                img_i = fliplr(img_i)

            for scale in ms:
                img_scaled = img_i
                if scale != 1:
                    img_scaled = nn.functional.interpolate(
                        img_i,
                        scale_factor=scale,
                        mode='bilinear',
                        align_corners=False
                    )

                with torch.no_grad():
                    raw = model(img_scaled, None)

                # ===== 统一解包 embedding =====
                if isinstance(raw, tuple):
                    raw = raw[1]

                if raw.dim() == 3:
                    raw = raw.mean(dim=2)

                if isinstance(raw, list):
                    raw = torch.stack(raw, dim=0).mean(dim=0)

                if ff is None:
                    ff = raw
                else:
                    ff += raw

        ff = ff / torch.norm(ff, p=2, dim=1, keepdim=True)
        features.append(ff.cpu())

    return torch.cat(features, dim=0)


def get_id(img_path):
    labels = []
    paths = []
    for path, v in img_path:
        folder_name = os.path.basename(os.path.dirname(path))
        labels.append(int(folder_name))
        paths.append(path)
    return labels, paths

######################################################################
# ⭐ 单次测试函数（用于多高度循环）
######################################################################

def test_single_height(model, test_dir_path, height_name=''):
    """
    对单个高度/数据集进行测试
    height_name: 用于区分不同高度，如 '150m' 或 ''（university 数据集）
    """
    print(f"\n{'='*70}")
    if height_name:
        print(f"📍 Testing {height_name}")
    else:
        print(f"📍 Testing on {test_dir_path}")
    print(f"{'='*70}")
    
    # 加载数据
    if opt.multi:
        image_datasets = {
            x: datasets.ImageFolder(os.path.join(test_dir_path, x), data_transforms) 
            for x in ['gallery','query','multi-query']
        }
        dataloaders = {
            x: torch.utils.data.DataLoader(
                image_datasets[x], 
                batch_size=opt.batchsize, 
                shuffle=False,
                num_workers=8,      # ⭐ 添加多进程
                pin_memory=True     # ⭐ 加速传输
            ) for x in ['gallery','query','multi-query']
        }
    elif opt.part != '':
        image_datasets_query = {
            x: datasets.ImageFolder(os.path.join(test_dir_path, x), data_query_transforms) 
            for x in ['query_drone']
        }
        image_datasets_gallery = {
            x: datasets.ImageFolder(os.path.join(test_dir_path, x), data_transforms) 
            for x in ['gallery_satellite']
        }
        image_datasets = {**image_datasets_query, **image_datasets_gallery}
        dataloaders = {
            x: torch.utils.data.DataLoader(
                image_datasets[x], 
                batch_size=opt.batchsize, 
                shuffle=False,
                num_workers=8,
                pin_memory=True
            ) for x in ['gallery_satellite', 'query_drone']
        }
    else:
        image_datasets_query = {
            x: datasets.ImageFolder(os.path.join(test_dir_path, x), data_query_transforms) 
            for x in ['query_satellite', 'query_drone']
        }
        image_datasets_gallery = {
            x: datasets.ImageFolder(os.path.join(test_dir_path, x), data_transforms) 
            for x in ['gallery_satellite', 'gallery_drone']
        }
        image_datasets = {**image_datasets_query, **image_datasets_gallery}
        dataloaders = {
            x: torch.utils.data.DataLoader(
                image_datasets[x], 
                batch_size=opt.batchsize, 
                shuffle=False,
                num_workers=8,
                pin_memory=True
            ) for x in ['gallery_satellite', 'gallery_drone', 'query_satellite', 'query_drone']
        }
    
    # 确定查询和检索库
    if opt.mode == 1:
        query_name = 'query_satellite'
        gallery_name = 'gallery_drone'
    elif opt.mode == 2:
        query_name = 'query_drone'
        gallery_name = 'gallery_satellite'
    else:
        raise Exception("opt.mode is not required")
    
    print(f"[DEBUG] query_name   = {query_name}")
    print(f"[DEBUG] gallery_name = {gallery_name}")
    
    which_gallery = which_view(gallery_name)
    which_query = which_view(query_name)
    
    print(f'{which_query} -> {which_gallery}')
    print(f"[DEBUG] Dataset root = {test_dir_path}")
    print(f"[DEBUG] Query dataset size = {len(image_datasets[query_name].imgs)}")
    print(f"[DEBUG] First 3 query images:")
    for p, _ in image_datasets[query_name].imgs[:3]:
        print("   ", p)
    
    # 提取特征
    since = time.time()
    
    with torch.no_grad():
        print(f"\n📸 Extracting query features from {query_name}...")
        query_feature = extract_feature(model, dataloaders[query_name], which_query)
        print(f"   Query feature shape: {query_feature.shape}")
        
        print(f"\n📸 Extracting gallery features from {gallery_name}...")
        gallery_feature = extract_feature(model, dataloaders[gallery_name], which_gallery)
        print(f"   Gallery feature shape: {gallery_feature.shape}")
    
        print(
            f"[Height={height_name}] "
            f"Query mean={query_feature.mean().item():.6f}, "
            f"std={query_feature.std().item():.6f}"
        )
        
    time_elapsed = time.time() - since
    print(f'\n✅ Feature extraction complete in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s')
    
    # 获取标签
    gallery_label, gallery_path = get_id(image_datasets[gallery_name].imgs)
    query_label, query_path = get_id(image_datasets[query_name].imgs)
    
    # 保存结果
    result = {
        'gallery_f': gallery_feature.numpy(),
        'gallery_label': gallery_label,
        'gallery_path': gallery_path,
        'query_f': query_feature.numpy(),
        'query_label': query_label,
        'query_path': query_path
    }
    
    # 根据高度名称保存不同的文件
    if height_name:
        result_mat = f'model/{opt.name}/pytorch_result_{height_name}.mat'
        result_txt = f'model/{opt.name}/result_{height_name}.txt'
        gallery_txt = f'model/{opt.name}/gallery_name_{height_name}.txt'
        query_txt = f'model/{opt.name}/query_name_{height_name}.txt'
    else:
        result_mat = f'model/{opt.name}/pytorch_result.mat'
        result_txt = f'model/{opt.name}/result.txt'
        gallery_txt = f'model/{opt.name}/gallery_name.txt'
        query_txt = f'model/{opt.name}/query_name.txt'
    
    scipy.io.savemat(result_mat, result)
    print(f"💾 Saved to: {result_mat}")
    
    # 保存路径信息
    with open(gallery_txt, 'w') as f:
        for p in gallery_path:
            f.write(p + '\n')
    with open(query_txt, 'w') as f:
        for p in query_path:
            f.write(p + '\n')
    
    # 评估
    print(f"\n📊 Running evaluation...")
    print("="*70)
    os.system(f'python evaluate.py --result_mat {result_mat} | tee -a {result_txt}')
    
    return result

######################################################################
# 加载模型
######################################################################

print('\n-------test-----------')

print("\n" + "="*70)
print("📦 Loading trained model...")
print("="*70)

model, _, epoch = load_network(opt.name, opt)

print(f"✅ Model loaded successfully from epoch: {epoch}")
print("="*70)

if hasattr(model, 'head'):
    model.head = nn.Sequential()

model = model.eval()
if use_gpu:
    model = model.cuda()

######################################################################
# ⭐ 主测试流程（支持多高度）
######################################################################

if __name__ == "__main__":
    
    if opt.dataset == 'sues200':
        # ⭐ SUES-200 多高度测试
        heights = opt.heights.split(',')
        
        print("\n" + "="*70)
        print(f"🚀 SUES-200 Multi-Height Testing")
        print(f"   Heights: {heights}")
        print("="*70)
        
        all_results = {}
        
        for height in heights:
            height_test_dir = os.path.join(opt.test_dir, height)
            
            # 检查目录是否存在
            if not os.path.exists(height_test_dir):
                print(f"\n⚠️  Skipping {height}m: Directory not found")
                print(f"    Expected path: {height_test_dir}")
                continue
            
            # 测试该高度
            result = test_single_height(model, height_test_dir, height_name=f'{height}m')
            all_results[height] = result
        
        # 汇总结果
        print("\n" + "="*70)
        print("📊 SUES-200 Multi-Height Test Summary")
        print("="*70)
        for height in heights:
            if height in all_results:
                result_file = f'./model/{opt.name}/result_{height}m.txt'
                if os.path.exists(result_file):
                    print(f"\n{height}m Results:")
                    os.system(f"grep 'top1\\|Recall@' {result_file} | head -3")
        
        print("\n✅ All heights tested!")
        print(f"   Results saved in: ./model/{opt.name}/result_*m.txt")
        
    else:
        # ⭐ University-1652 标准测试
        print("\n" + "="*70)
        print(f"🚀 University-1652 Testing")
        print("="*70)
        
        test_single_height(model, opt.test_dir)
        
        print("\n✅ Test completed!")
        print(f"   Results saved in: ./model/{opt.name}/result.txt")
