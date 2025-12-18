"""
DINOv2 Backbone for Cross-view Geo-localization
模仿 ConvNext.py 的结构，实现 DINOv2 backbone
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

warnings.filterwarnings('ignore')  # 忽略 torch.hub 下载警告

class DINOv2WithClassifier(nn.Module):
    """
    DINOv2 backbone + 多分类器头（使用 patch tokens 空间池化）
    模仿原代码的 MCCG 结构
    """
    def __init__(
        self, 
        num_class=701,
        model_size='vitb14',  # vits14, vitb14, vitl14, vitg14
        block=4,  # 分类器数量
        return_f=False,  # 是否返回特征用于 triplet loss
        freeze_backbone=False,
        dropout=0.5,
    ):
        super().__init__()
        
        self.return_f = return_f
        self.block = block
        
        # DINOv2 模型维度映射
        self.dim_mapping = {
            'vits14': 384,
            'vitb14': 768,
            'vitl14': 1024,
            'vitg14': 1536,
        }
        self.feature_dim = self.dim_mapping.get(model_size, 768)
        
        # 加载 DINOv2 预训练模型
        print(f"\n[INFO] Loading DINOv2 model: dinov2_{model_size} (patch token mode)")
        try:
            self.backbone = torch.hub.load(
                'facebookresearch/dinov2', 
                f'dinov2_{model_size}',
                trust_repo=True  # 信任远程仓库（避免下载警告）
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load DINOv2: {str(e)}") from e
        
        # 冻结 backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print(f"[INFO] DINOv2 backbone is frozen")
        
        # 全局平均池化 + Dropout
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)
        
        # 构建多分类器头（MCCG）
        self.classifiers = nn.ModuleList()
        for i in range(block):
            classifier = nn.Sequential(
                nn.Linear(self.feature_dim, self.feature_dim // 2),
                nn.BatchNorm1d(self.feature_dim // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(self.feature_dim // 2, num_class)
            )
            self.classifiers.append(classifier)
        
        # 初始化分类器
        self._init_classifiers()
    
    def _init_classifiers(self):
        """Kaiming 初始化分类器权重"""
        for classifier in self.classifiers:
            for m in classifier.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm1d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        前向传播
        Args:
            x: [B, 3, H, W] - 输入图像（需满足 DINOv2 输入尺寸，默认 224x224）
        Returns:
            predictions: 列表，每个元素为 [B, num_class]（多分类器输出）
            features_for_triplet: [B, feature_dim]（用于 triplet loss 的特征）
        """
        B = x.shape[0]
        
        # 特征提取（禁用梯度计算当 backbone 冻结时，加速训练）
        with torch.set_grad_enabled(self.training and not next(self.backbone.parameters()).requires_grad):
            features_dict = self.backbone.forward_features(x)
        
        # Patch tokens -> 空间特征图 -> 全局池化
        patch_tokens = features_dict['x_norm_patchtokens']  # [B, num_patches, dim]
        num_patches = patch_tokens.shape[1]
        h = w = int(num_patches ** 0.5)
        features_spatial = patch_tokens.reshape(B, h, w, -1).permute(0, 3, 1, 2)  # [B, dim, h, w]
        features_pooled = self.avgpool(features_spatial).view(B, -1)  # [B, dim]
        
        # 保存原始特征（用于 triplet loss）
        features_for_triplet = features_pooled.detach().clone()  # 避免梯度传递干扰
        
        # 多分类器预测
        predictions = []
        x_cls = self.dropout(features_pooled)
        for classifier in self.classifiers:
            predictions.append(classifier(x_cls))
        
        return (predictions, features_for_triplet) if self.return_f else predictions


class DINOv2WithCLSToken(nn.Module):
    """
    DINOv2 backbone + 多分类器头（使用 CLS token 作为全局特征）
    更简单高效的实现方式
    """
    def __init__(
        self, 
        num_class=701,
        model_size='vitb14',
        block=4,
        return_f=False,
        freeze_backbone=False,
        dropout=0.5,
    ):
        super().__init__()
        
        self.return_f = return_f
        self.block = block
        
        self.dim_mapping = {
            'vits14': 384,
            'vitb14': 768,
            'vitl14': 1024,
            'vitg14': 1536,
        }
        self.feature_dim = self.dim_mapping.get(model_size, 768)
        
        # 加载 DINOv2 预训练模型
        print(f"\n[INFO] Loading DINOv2 model: dinov2_{model_size} (CLS token mode)")
        try:
            self.backbone = torch.hub.load(
                'facebookresearch/dinov2', 
                f'dinov2_{model_size}',
                trust_repo=True
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load DINOv2: {str(e)}") from e
        
        # 冻结 backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print(f"[INFO] DINOv2 backbone is frozen")
        
        # Dropout + 多分类器
        self.dropout = nn.Dropout(p=dropout)
        self.classifiers = nn.ModuleList()
        for i in range(block):
            classifier = nn.Sequential(
                nn.Linear(self.feature_dim, self.feature_dim // 2),
                nn.BatchNorm1d(self.feature_dim // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(self.feature_dim // 2, num_class)
            )
            self.classifiers.append(classifier)
        
        self._init_classifiers()
    
    def _init_classifiers(self):
        """Kaiming 初始化分类器权重"""
        for classifier in self.classifiers:
            for m in classifier.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm1d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        前向传播（直接使用 CLS token）
        Args:
            x: [B, 3, H, W]
        Returns:
            同 DINOv2WithClassifier
        """
        with torch.set_grad_enabled(self.training and not next(self.backbone.parameters()).requires_grad):
            features_dict = self.backbone.forward_features(x)
        
        cls_token = features_dict['x_norm_clstoken']  # [B, dim]
        features_for_triplet = cls_token.detach().clone()
        
        # 多分类器预测
        predictions = []
        x_cls = self.dropout(cls_token)
        for classifier in self.classifiers:
            predictions.append(classifier(x_cls))
        
        return (predictions, features_for_triplet) if self.return_f else predictions


def make_dinov2_model(
    num_class=701, 
    block=4, 
    return_f=False, 
    model_size='vitb14', 
    use_cls_token=False, 
    freeze_backbone=False, 
    dropout=0.5
):
    """
    创建 DINOv2 模型的工厂函数
    Args:
        use_cls_token: True=使用 CLS token，False=使用 patch token 空间池化
    Returns:
        初始化后的 DINOv2 模型
    """
    if use_cls_token:
        model = DINOv2WithCLSToken(
            num_class=num_class,
            model_size=model_size,
            block=block,
            return_f=return_f,
            freeze_backbone=freeze_backbone,
            dropout=dropout
        )
    else:
        model = DINOv2WithClassifier(
            num_class=num_class,
            model_size=model_size,
            block=block,
            return_f=return_f,
            freeze_backbone=freeze_backbone,
            dropout=dropout
        )
    return model


# 测试代码（完整验证模型功能）
if __name__ == "__main__":
    print("="*60)
    print("Testing DINOv2 Backbone for Cross-view Geo-localization")
    print("="*60)
    
    # 测试配置
    NUM_CLASS = 701
    BATCH_SIZE = 2
    IMG_SIZE = 224  # DINOv2 默认输入尺寸
    MODEL_SIZE = 'vitb14'  # 轻量版可选 'vits14'（下载更快）
    
    # 1. 测试 patch token 模式（空间池化）
    print("\n" + "-"*50)
    print(f"Test 1: DINOv2WithClassifier (patch token mode)")
    print("-"*50)
    try:
        model1 = make_dinov2_model(
            num_class=NUM_CLASS,
            block=4,
            return_f=True,
            model_size=MODEL_SIZE,
            use_cls_token=False,
            freeze_backbone=False
        )
        model1.eval()  # 推理模式
        
        # 生成随机输入
        x = torch.randn(BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE)
        
        # 前向传播
        with torch.no_grad():  # 禁用梯度计算，加速测试
            predictions1, features1 = model1(x)
        
        # 验证输出
        print(f"✓ Model created successfully")
        print(f"  - Number of classifiers: {len(predictions1)}")
        print(f"  - Each prediction shape: {predictions1[0].shape} (expected: [{BATCH_SIZE}, {NUM_CLASS}])")
        print(f"  - Feature shape: {features1.shape} (expected: [{BATCH_SIZE}, {model1.feature_dim}])")
    except Exception as e:
        print(f"✗ Test 1 failed: {str(e)}")
    
    # 2. 测试 CLS token 模式（更高效）
    print("\n" + "-"*50)
    print(f"Test 2: DINOv2WithCLSToken (CLS token mode)")
    print("-"*50)
    try:
        model2 = make_dinov2_model(
            num_class=NUM_CLASS,
            block=4,
            return_f=True,
            model_size=MODEL_SIZE,
            use_cls_token=True,
            freeze_backbone=True  # 测试冻结 backbone
        )
        model2.eval()
        
        # 生成随机输入
        x = torch.randn(BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE)
        
        # 前向传播
        with torch.no_grad():
            predictions2, features2 = model2(x)
        
        # 验证输出
        print(f"✓ Model created successfully")
        print(f"  - Number of classifiers: {len(predictions2)}")
        print(f"  - Each prediction shape: {predictions2[0].shape} (expected: [{BATCH_SIZE}, {NUM_CLASS}])")
        print(f"  - Feature shape: {features2.shape} (expected: [{BATCH_SIZE}, {model2.feature_dim}])")
        print(f"  - Backbone frozen: {not next(model2.backbone.parameters()).requires_grad}")
    except Exception as e:
        print(f"✗ Test 2 failed: {str(e)}")
    
    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)
