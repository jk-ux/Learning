"""
修改后的 make_model.py - 集成 DINOv2 backbone
保留原始 MCCG 架构，仅替换 backbone 部分
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from timm.models import create_model
from .backbones.model_convnext import convnext_tiny
from .backbones.resnet import Resnet
import numpy as np
from torch.nn import init
from torch.nn.parameter import Parameter
import warnings

warnings.filterwarnings('ignore')


# ============ 保留原始的辅助类（不变） ============  
class Gem_heat(nn.Module):
    def __init__(self, dim=768, p=3, eps=1e-6):
        super(Gem_heat, self).__init__()
        self.p = nn.Parameter(torch.ones(dim) * p) 
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3):
        p = F.softmax(p).unsqueeze(-1)
        x = torch.matmul(x, p)
        x = x.view(x.size(0), x.size(1))
        return x


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ZPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class AttentionGate(nn.Module):
    def __init__(self):
        super(AttentionGate, self).__init__()
        kernel_size = 7
        self.compress = ZPool()
        self.conv = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.conv(x_compress)
        scale = torch.sigmoid_(x_out)
        return x * scale


class TripletAttention(nn.Module):
    """
    ⭐ 原版 TripletAttention
    返回两个特征：(x_out11, x_out21)
    """
    def __init__(self):
        super(TripletAttention, self).__init__()
        self.cw = AttentionGate()
        self.hc = AttentionGate()
    
    def forward(self, x):
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        x_out1 = self.cw(x_perm1)
        x_out11 = x_out1.permute(0, 2, 1, 3).contiguous()
        
        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        x_out2 = self.hc(x_perm2)
        x_out21 = x_out2.permute(0, 3, 2, 1).contiguous()
        
        return x_out11, x_out21


class ClassBlock(nn.Module):
    """
    ⭐ 原版 ClassBlock
    包含 bottleneck、BatchNorm、Dropout、分类器
    """
    def __init__(self, input_dim, class_num, droprate, relu=False, bnorm=True, num_bottleneck=512, linear=True, return_f=False):
        super(ClassBlock, self).__init__()
        self.return_f = return_f
        
        add_block = []
        if linear:
            add_block += [nn.Linear(input_dim, num_bottleneck)]
        else:
            num_bottleneck = input_dim
        
        if bnorm:
            add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate > 0:
            add_block += [nn.Dropout(p=droprate)]
        
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier
    
    def forward(self, x):
        x = self.add_block(x)
        if self.training:
            if self.return_f:
                f = x
                x = self.classifier(x)
                return x, f
            else:
                x = self.classifier(x)
                return x
        else:
            return x


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, std=0.001)
        nn.init.constant_(m.bias.data, 0.0)


# ============ 新增：DINOv2 Backbone ============
class DINOv2Backbone(nn.Module):
    """
    DINOv2 backbone - 输出格式与 ConvNeXt 一致
    """
    def __init__(self, model_size='vitb14', freeze_backbone=False):
        super().__init__()
        self.freeze_backbone = freeze_backbone
        
        # 维度映射
        self.dim_mapping = {
            'vits14': 384,
            'vitb14': 768,
            'vitl14': 1024,
            'vitg14': 1536,
        }
        self.feature_dim = self.dim_mapping.get(model_size, 768)
        
        # 加载预训练模型
        print(f"[INFO] Loading DINOv2: dinov2_{model_size}")
        self.backbone = torch.hub.load(
            'facebookresearch/dinov2', 
            f'dinov2_{model_size}',
            trust_repo=True
        )
        
        # 冻结权重
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print(f"[INFO] DINOv2 backbone is FROZEN")
        else:
            print(f"[INFO] DINOv2 backbone is TRAINABLE")
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    
    def forward(self, x):
        """
        返回: (gap_feature, part_features) - 与 ConvNeXt 格式一致
        """
        B = x.shape[0]
        
        # 特征提取
        if self.freeze_backbone:
            with torch.no_grad():
                features_dict = self.backbone.forward_features(x)
        else:
            features_dict = self.backbone.forward_features(x)
        
        # Patch tokens -> 空间特征图
        patch_tokens = features_dict['x_norm_patchtokens']
        num_patches = patch_tokens.shape[1]
        h = w = int(num_patches ** 0.5)
        part_features = patch_tokens.reshape(B, h, w, -1).permute(0, 3, 1, 2)
        
        # 全局特征
        gap_feature = self.avgpool(part_features).view(B, -1)
        
        return gap_feature, part_features


# ============ 修改后的 build_convnext（支持 DINOv2） ============
class build_convnext(nn.Module):
    """
    MCCG 模型构建类（支持多种 backbone + 结构感知注意力）
    
    ⭐ 新增：集成结构感知注意力机制
    """
    def __init__(
        self, 
        num_classes, 
        block=4, 
        return_f=False, 
        backbone_type='convnext',      # 'convnext', 'resnet', 'dinov2', 'hybrid'
        dinov2_model='vitb14',         # DINOv2 模型大小
        freeze_dinov2=False,           # 是否冻结 DINOv2
        use_structure_aware=False,     # ⭐ 新增：是否启用结构感知模块
        use_hybrid=False,              # ⭐ 新增：是否启用混合特征提取
        dropout=0.5,                   # Dropout 比例
    ):
        super(build_convnext, self).__init__()
        
        self.return_f = return_f
        self.block = block
        self.num_classes = num_classes
        self.use_structure_aware = use_structure_aware
        self.use_hybrid = use_hybrid
        self.backbone_type = backbone_type
        
        # ========== Backbone 选择 ==========
        if backbone_type == 'resnet':
            print('[INFO] Using ResNet101 as backbone')
            self.in_planes = 2048
            
            try:
                from .resnet_backbone import Resnet
                self.convnext = Resnet(pretrained=True)
            except ImportError:
                print('[ERROR] ResNet backbone not found!')
                raise
            
            # ResNet 不支持结构感知和混合架构
            if use_structure_aware or use_hybrid:
                print('[WARNING] Structure-aware and hybrid modes are only supported for DINOv2 backbone')
                print('[WARNING] Disabling these features for ResNet')
                self.use_structure_aware = False
                self.use_hybrid = False
        
        elif backbone_type == 'dinov2' or use_hybrid:
            # ⭐ DINOv2 或混合架构
            if use_hybrid:
                print(f'[INFO] Using Hybrid Architecture (DINOv2 + CNN)')
                
                from .hybrid_backbone import HybridBackbone
                self.convnext = HybridBackbone(
                    dinov2_size=dinov2_model,
                    freeze_dinov2=freeze_dinov2,
                    fusion_dim=512,
                    use_structure_aware=False  # 混合架构的结构感知在外部处理
                )
                self.in_planes = 512  # 混合后的特征维度
            
            else:
                print(f'[INFO] Using DINOv2-{dinov2_model} as backbone')
                
                from .dinov2_backbone import DINOv2Backbone
                self.convnext = DINOv2Backbone(
                    model_size=dinov2_model,
                    freeze_backbone=freeze_dinov2,
                    use_structure_aware=False  # 结构感知在外部处理
                )
                
                dim_mapping = {
                    'vits14': 384,
                    'vitb14': 768,
                    'vitl14': 1024,
                    'vitg14': 1536,
                }
                self.in_planes = dim_mapping.get(dinov2_model, 768)
        
        else:  # 默认使用 ConvNeXt
            convnext_name = "convnext_tiny"
            print(f'[INFO] Using {convnext_name} as backbone')
            
            if 'base' in convnext_name:
                self.in_planes = 1024
            elif 'large' in convnext_name:
                self.in_planes = 1536
            elif 'xlarge' in convnext_name:
                self.in_planes = 2048
            else:
                self.in_planes = 768
            
            from timm import create_model
            self.convnext = create_model(convnext_name, pretrained=True)
            
            # ConvNeXt 不支持结构感知和混合架构
            if use_structure_aware or use_hybrid:
                print('[WARNING] Structure-aware and hybrid modes are only supported for DINOv2 backbone')
                print('[WARNING] Disabling these features for ConvNeXt')
                self.use_structure_aware = False
                self.use_hybrid = False
        
        # ========== ⭐ 结构感知注意力模块（插入点）==========
        if use_structure_aware:
            print('[INFO] Initializing Structure-Aware Attention Module...')
            from .structure_attention import StructureAwareAttentionNetwork
            self.structure_attention = StructureAwareAttentionNetwork(
                feat_dim=self.in_planes,
                num_levels=3,
                reduction=16
            )
            print('[INFO]   ✅ Structure-Aware Attention initialized')
        else:
            self.structure_attention = None
        
        # ========== 分类器（所有 backbone 共用）==========
        self.classifier1 = ClassBlock(
            self.in_planes, 
            num_classes, 
            dropout, 
            return_f=return_f
        )
        self.tri_layer = TripletAttention()
        
        # 多分类器
        for i in range(self.block):
            name = 'classifier_mcb' + str(i + 1)
            setattr(self, name, ClassBlock(
                self.in_planes, 
                num_classes, 
                dropout, 
                return_f=self.return_f
            ))
        
        print(f'[INFO] MCCG model initialized:')
        print(f'       - Backbone: {backbone_type}')
        print(f'       - Feature dim: {self.in_planes}')
        print(f'       - Num classifiers: {self.block + 1}')
        print(f'       - Structure-aware: {self.use_structure_aware}')
        print(f'       - Hybrid: {self.use_hybrid}')
    
    def forward(self, x, x2=None, return_structure=False):
        """
        前向传播（完全向后兼容）
        
        Args:
            x: 第一个视图 [B, 3, H, W]
            x2: 第二个视图 [B, 3, H, W]（可选）
            return_structure: 是否返回结构信息（默认False）
        
        Returns:
            原版模式：(cls, features) 或 y
            增强模式：(feat_info, attn_info, embed_info)
        """
        
        # 保存原始图像（用于结构感知）
        raw_image = x
        raw_image2 = x2
        
        # ========== 1. Backbone 特征提取 ==========
        gap_feature, part_features = self.convnext(x)
        
        if x2 is not None:
            gap_feature2, part_features2 = self.convnext(x2)
        
        # ========== 2. ⭐ 结构感知注意力增强（插入点）==========
        if self.use_structure_aware and self.structure_attention is not None:
            # 单视图处理
            enhanced_feat, global_feat, align_embed, attn_maps = self.structure_attention(
                feat_map=part_features,
                raw_image=raw_image,
                view2_feat=part_features2 if x2 is not None else None,
                view2_image=raw_image2 if x2 is not None else None
            )
            
            # ⭐ 使用增强后的特征
            part_features = enhanced_feat
            gap_feature = global_feat
            
            if x2 is not None:
                # 对第二视图也进行处理
                enhanced_feat2, global_feat2, align_embed2, attn_maps2 = self.structure_attention(
                    feat_map=part_features2,
                    raw_image=raw_image2,
                    view2_feat=part_features,
                    view2_image=raw_image
                )
                part_features2 = enhanced_feat2
                gap_feature2 = global_feat2
        else:
            attn_maps = None
            align_embed = None
            if x2 is not None:
                attn_maps2 = None
                align_embed2 = None
        
        # ========== 3. TripletAttention 处理（保持不变）==========
        tri_features = self.tri_layer(part_features)
        convnext_feature = self.classifier1(gap_feature)
        
        if x2 is not None:
            tri_features2 = self.tri_layer(part_features2)
            convnext_feature2 = self.classifier1(gap_feature2)
        
        # ========== 4. 多分类器（保持不变）==========
        tri_list = []
        for i in range(len(tri_features)):
            tri_list.append(tri_features[i].mean([-2, -1]))
        
        while len(tri_list) < self.block:
            tri_list.append(tri_list[0])
        
        triatten_features = torch.stack(tri_list[:self.block], dim=2)
        
        if x2 is not None:
            tri_list2 = []
            for i in range(len(tri_features2)):
                tri_list2.append(tri_features2[i].mean([-2, -1]))
            
            while len(tri_list2) < self.block:
                tri_list2.append(tri_list2[0])
            
            triatten_features2 = torch.stack(tri_list2[:self.block], dim=2)
        
        # 部分分类器
        if self.block == 0:
            y = []
            if x2 is not None:
                y2 = []
        else:
            y = self.part_classifier(self.block, triatten_features, cls_name='classifier_mcb')
            if x2 is not None:
                y2 = self.part_classifier(self.block, triatten_features2, cls_name='classifier_mcb')
        
        # ========== 5. 返回结果 ==========
        if self.training:
            # 训练模式
            y = y + [convnext_feature]
            if x2 is not None:
                y2 = y2 + [convnext_feature2]
            
            if self.return_f:
                # 需要返回特征（用于 triplet loss）
                cls, features = [], []
                for i in y:
                    cls.append(i[0])
                    features.append(i[1])
                
                if x2 is not None:
                    cls2, features2 = [], []
                    for i in y2:
                        cls2.append(i[0])
                        features2.append(i[1])
                    
                    # ⭐ 根据是否需要结构信息返回
                    if return_structure and self.use_structure_aware:
                        return (
                            ((cls, features), (cls2, features2)),   # 分类+特征
                            (attn_maps, attn_maps2),                 # 注意力图
                            (align_embed, align_embed2)              # 对齐嵌入
                        )
                    else:
                        # ✅ 原版返回格式
                        return (cls, features), (cls2, features2)
                else:
                    # 单视图
                    if return_structure and self.use_structure_aware:
                        return (cls, features), attn_maps, align_embed
                    else:
                        return (cls, features)
            else:
                # 不返回特征
                if x2 is not None:
                    return y, y2
                else:
                    return y
        
        else:
            # ✅ 测试模式（保持原样）
            ffeature = convnext_feature.view(convnext_feature.size(0), -1, 1)
            
            if self.block == 0:
                y = ffeature
            else:
                y = torch.cat([y, ffeature], dim=2)
            
            if x2 is not None:
                ffeature2 = convnext_feature2.view(convnext_feature2.size(0), -1, 1)
                
                if self.block == 0:
                    y2 = ffeature2
                else:
                    y2 = torch.cat([y2, ffeature2], dim=2)
                
                if return_structure and self.use_structure_aware:
                    return (y, y2), (attn_maps, attn_maps2), (align_embed, align_embed2)
                else:
                    return y, y2
            else:
                if return_structure and self.use_structure_aware:
                    return y, attn_maps, align_embed
                else:
                    return y
    
    
    def part_classifier(self, block, x, cls_name='classifier_mcb'):
        """
        部分分类器（保持不变）
        """
        part = {}
        predict = {}
        
        for i in range(block):
            part[i] = x[:, :, i].view(x.size(0), -1)
            name = cls_name + str(i + 1)
            c = getattr(self, name)
            predict[i] = c(part[i])
        y = []
        for i in range(block):
            y.append(predict[i])
        if not self.training:
            return torch.stack(y, dim=2)
        return y


def part_classifier(self, block, x, cls_name='classifier_mcb'):
        """
        部分分类器（保持不变）
        """
        part = {}
        predict = {}
        for i in range(block):
            part[i] = x[:, :, i].view(x.size(0), -1)
            name = cls_name + str(i + 1)
            c = getattr(self, name)
            predict[i] = c(part[i])
        
        y = []
        for i in range(block):
            y.append(predict[i])
        
        if not self.training:
            return torch.stack(y, dim=2)
        
        return y

# ============ 工厂函数 ============
def make_model(
    num_class,
    block=4,
    return_f=False,
    backbone='convnext',            # 'convnext', 'resnet', 'dinov2'
    dinov2_model='vitb14',          # DINOv2 模型: vits14/vitb14/vitl14/vitg14
    freeze_dinov2=False,            # 是否冻结 DINOv2
    use_structure_aware=False,      # 是否启用结构感知模块
    use_hybrid=False,               # ⭐ 是否启用混合特征提取
    dropout=0.5,                    # Dropout 比例
):
    """
    统一的模型创建接口
    
    Args:
        num_class: 类别数
        block: MCCG 多分类器数量
        return_f: 是否返回特征（用于 triplet loss）
        backbone: backbone 类型 ['convnext', 'resnet', 'dinov2']
        dinov2_model: DINOv2 模型大小
        freeze_dinov2: 是否冻结 DINOv2 backbone
        use_structure_aware: 是否启用结构感知模块
        use_hybrid: ⭐ 是否启用混合特征提取（DINOv2 + CNN）
        dropout: Dropout 比例
    
    Returns:
        model: 创建的模型实例
    
    使用示例:
        # 标准 DINOv2
        model = make_model(701, backbone='dinov2')
        
        # ⭐ 混合架构（DINOv2 + CNN）
        model = make_model(701, backbone='dinov2', use_hybrid=True)
        
        # 混合架构 + 结构感知
        model = make_model(701, backbone='dinov2', use_hybrid=True, use_structure_aware=True)
    """
    
    # ========== 打印模型配置 ==========
    print('='*70)
    print(f'Building MCCG model with {backbone.upper()} backbone')
    
    if backbone == 'dinov2':
        print(f'  - Model size: {dinov2_model}')
        print(f'  - Freeze backbone: {freeze_dinov2}')
        print(f'  - Dropout: {dropout}')
        
        # ⭐ 打印优化模块状态
        if use_hybrid:
            print(f'  - 🔥 Hybrid Feature Extraction: ENABLED')
            print(f'       (DINOv2 + Lightweight CNN + Cross-Attention)')
        else:
            print(f'  - ⭕ Hybrid Feature Extraction: DISABLED')
        
        if use_structure_aware:
            print(f'  - 🔥 Structure-Aware Module: ENABLED')
        else:
            print(f'  - ⭕ Structure-Aware Module: DISABLED')
    
    print('='*70)
    
    # ========== 创建模型 ==========
    if backbone == 'dinov2':
        # DINOv2 + MCCG
        model = build_convnext(
            num_classes=num_class,
            block=block,
            return_f=return_f,
            backbone_type='dinov2',
            dinov2_model=dinov2_model,
            freeze_dinov2=freeze_dinov2,
            use_structure_aware=use_structure_aware,
            use_hybrid=use_hybrid,  # ⭐ 传递混合架构参数
            dropout=dropout
        )
    
    elif backbone == 'resnet':
        # ResNet + MCCG
        if use_structure_aware or use_hybrid:
            print("[WARNING] Structure-aware and hybrid modes are only supported for DINOv2 backbone.")
            print("[WARNING] Falling back to standard ResNet model.")
        
        model = build_convnext(
            num_classes=num_class,
            block=block,
            return_f=return_f,
            backbone_type='resnet',
            dropout=dropout
        )
    
    else:  # 默认 convnext
        # ConvNeXt + MCCG
        if use_structure_aware or use_hybrid:
            print("[WARNING] Structure-aware and hybrid modes are only supported for DINOv2 backbone.")
            print("[WARNING] Falling back to standard ConvNeXt model.")
        
        model = build_convnext(
            num_classes=num_class,
            block=block,
            return_f=return_f,
            backbone_type='convnext',
            dropout=dropout
        )
    
    return model

def build_mccg_model(
    num_classes,
    block=4,
    return_f=False,
    backbone='dinov2',
    dinov2_model='vitb14',
    freeze_dinov2=False,
    use_structure_aware=False,
    use_hybrid=False,
    dropout=0.5
):
    """
    工厂函数：创建 MCCG 模型
    这是推荐的创建模型的方式
    """
    return build_convnext(
        num_classes=num_classes,
        block=block,
        return_f=return_f,
        backbone_type=backbone,
        dinov2_model=dinov2_model,
        freeze_dinov2=freeze_dinov2,
        use_structure_aware=use_structure_aware,
        use_hybrid=use_hybrid,
        dropout=dropout
    )
    
# 向后兼容：保留原始接口
def make_convnext_model(num_class, block=4, return_f=False, resnet=False):
    """原始接口（向后兼容）"""
    backbone = 'resnet' if resnet else 'convnext'
    return make_model(num_class, block, return_f, backbone=backbone)
  
if __name__ == "__main__":
    print("\n" + "="*70)
    print("测试 build_convnext 类")
    print("="*70)
    
    # 测试 1: 原版 DINOv2（单视图）
    print("\n[Test 1] Original DINOv2 (single view)")
    model = build_convnext(
        num_classes=701,
        block=4,
        return_f=True,
        backbone_type='dinov2',
        dinov2_model='vitb14',
        freeze_dinov2=False,
        use_structure_aware=False  # ✅ 原版
    )
    
    x = torch.randn(2, 3, 224, 224)
    model.eval()
    
    with torch.no_grad():
        y = model(x)
    
    print(f"✅ Output shape: {y.shape}")
    
    # 测试 2: 增强版 DINOv2（单视图）
    print("\n[Test 2] Enhanced DINOv2 with Structure-Aware (single view)")
    model = build_convnext(
        num_classes=701,
        block=4,
        return_f=True,
        backbone_type='dinov2',
        dinov2_model='vitb14',
        use_structure_aware=True  # ⭐ 启用结构感知
    )
    
    model.eval()
    
    with torch.no_grad():
        # 原版调用（向后兼容）
        y = model(x)
        print(f"✅ Original call - Output shape: {y.shape}")
        
        # 增强调用
        y, attn, embed = model(x, return_structure=True)
        print(f"✅ Enhanced call - Output: {y.shape}, Attn: {attn.shape}, Embed: {embed.shape}")
    
    # 测试 3: 双视图训练
    print("\n[Test 3] Dual-view training mode")
    model = build_convnext(
        num_classes=701,
        block=4,
        return_f=True,
        backbone_type='dinov2',
        use_structure_aware=True
    )
    
    model.train()
    x1 = torch.randn(2, 3, 224, 224)
    x2 = torch.randn(2, 3, 224, 224)
    
    # 原版调用
    result = model(x1, x2, return_structure=False)
    (cls1, feat1), (cls2, feat2) = result
    print(f"✅ Original dual-view - cls1: {len(cls1)}, feat1: {len(feat1)}")
    
    # 增强调用
    result = model(x1, x2, return_structure=True)
    feat_info, attn_info, embed_info = result
    print(f"✅ Enhanced dual-view - Features, Attentions, Embeddings returned")
    
    print("\n" + "="*70)
    print("✅ All tests passed! Model is backward compatible.")
    print("="*70)
