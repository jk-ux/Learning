"""
DINOv2 Backbone for Cross-view Geo-localization
完整实现，包含两种模式：
1. DINOv2Backbone - 输出格式与ConvNeXt兼容（用于MCCG架构）
2. DINOv2WithCLSToken - 集成分类器的完整模型（简化版）
"""

import torch
import torch.nn as nn
import warnings

warnings.filterwarnings('ignore')


class DINOv2Backbone(nn.Module):
    """
    DINOv2 backbone - 输出格式与 ConvNeXt 一致
    
    ⭐ 新增功能：
    - 支持结构感知模块（可选）
    - 向后兼容，默认行为不变
    
    返回格式：
    - 原版模式: (gap_feature, part_features)
    - 增强模式: (gap_feature, part_features, struct_feat, attn_maps, align_embed)
    """
    def __init__(self, model_size='vitb14', freeze_backbone=False, use_structure_aware=False):
        super().__init__()
        self.freeze_backbone = freeze_backbone
        self.use_structure_aware = use_structure_aware  # ⭐ 新增：结构感知开关
        
        # DINOv2 模型维度映射
        self.dim_mapping = {
            'vits14': 384,
            'vitb14': 768,
            'vitl14': 1024,
            'vitg14': 1536,
        }
        self.feature_dim = self.dim_mapping.get(model_size, 768)
        
        # ⭐ 结构感知模块（默认不使用）
        self.structure_module = None
        
        # 加载 DINOv2 预训练模型
        print(f"[INFO] Loading DINOv2: dinov2_{model_size}")
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
            print(f"[INFO] DINOv2 backbone is FROZEN")
        else:
            print(f"[INFO] DINOv2 backbone is TRAINABLE")
        
        # 全局平均池化
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # ⭐ 打印结构感知状态
        if use_structure_aware:
            print(f"[INFO] Structure-Aware mode: ENABLED (will be initialized by model)")
        else:
            print(f"[INFO] Structure-Aware mode: DISABLED (original behavior)")
    
    def forward(self, x, return_structure=False):
        """
        前向传播（向后兼容）
        
        Args:
            x: [B, 3, H, W] 输入图像
            return_structure: 是否返回结构信息（默认False，保持兼容）
        
        Returns:
            原版模式（return_structure=False 或 structure_module=None）：
                gap_feature: [B, feature_dim] - 全局特征
                part_features: [B, feature_dim, h, w] - 空间特征图
            
            增强模式（return_structure=True 且 structure_module 已初始化）：
                gap_feature: [B, feature_dim] - 全局特征
                part_features: [B, feature_dim, h, w] - 空间特征图
                struct_feat: [B, feature_dim] - 结构感知特征
                attn_maps: [B, num_parts, h, w] - 结构注意力图
                align_embed: [B, 256] - 对齐嵌入（用于对比学习）
        """
        B = x.shape[0]
        
        # ========== 1. DINOv2 特征提取 ==========
        if self.freeze_backbone:
            with torch.no_grad():
                features_dict = self.backbone.forward_features(x)
        else:
            features_dict = self.backbone.forward_features(x)
        
        # ========== 2. 处理 Patch Tokens ==========
        patch_tokens = features_dict['x_norm_patchtokens']  # [B, num_patches, dim]
        num_patches = patch_tokens.shape[1]
        h = w = int(num_patches ** 0.5)
        
        assert num_patches == h * w, \
            f"Patch reshape error: {num_patches} patches cannot form {h}x{w} grid"
        
        # 重塑为空间特征图
        part_features = patch_tokens.reshape(B, h, w, -1).permute(0, 3, 1, 2)  # [B, C, h, w]
        
        # ========== 3. 全局特征（原有逻辑）==========
        gap_feature = self.avgpool(part_features).view(B, -1)  # [B, feature_dim]
        
        # ========== 4. 结构感知处理（可选）==========
        if return_structure and self.structure_module is not None:
            # ✅ 增强模式：使用结构感知模块
            struct_feat, attn_maps, align_embed = self.structure_module(
                feat_map=part_features,  # [B, C, h, w] 空间特征图
                image=x                   # [B, 3, H, W] 原始图像
            )
            
            # 返回完整信息
            return gap_feature, part_features, struct_feat, attn_maps, align_embed
        
        else:
            # ✅ 原版模式：保持原有返回格式
            return gap_feature, part_features


class DINOv2WithCLSToken(nn.Module):
    """
    DINOv2 + 分类器（使用CLS token）
    这是一个独立的模型实现，不依赖原始MCCG架构
    输出格式：(predictions_list, features_list) 或 predictions_list
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
        self.freeze_backbone = freeze_backbone
        
        # DINOv2 模型维度映射
        self.dim_mapping = {
            'vits14': 384,
            'vitb14': 768,
            'vitl14': 1024,
            'vitg14': 1536,
        }
        self.feature_dim = self.dim_mapping.get(model_size, 768)
        
        # 加载 DINOv2 预训练模型
        print(f"[INFO] Loading DINOv2: dinov2_{model_size} (CLS token mode)")
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
            print(f"[INFO] DINOv2 backbone is FROZEN")
        else:
            print(f"[INFO] DINOv2 backbone is TRAINABLE")
        
        # Dropout
        self.dropout = nn.Dropout(p=dropout)
        
        # 多分类器（类似原始MCCG）
        self.classifiers = nn.ModuleList()
        for i in range(block):
            classifier = nn.Sequential(
                nn.Linear(self.feature_dim, self.feature_dim),
                nn.BatchNorm1d(self.feature_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(self.feature_dim, self.feature_dim // 2),
                nn.BatchNorm1d(self.feature_dim // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(self.feature_dim // 2, num_class)
            )
            self.classifiers.append(classifier)
        
        self._init_classifiers()
    
    def _init_classifiers(self):
        """Kaiming 初始化"""
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
            x: [B, 3, H, W]
        Returns:
            训练模式 + return_f=True: (predictions_list, features_list)
            训练模式 + return_f=False: predictions_list
            测试模式: stacked_features [B, feature_dim, block]
        """
        # 特征提取
        if self.freeze_backbone:
            with torch.no_grad():
                features_dict = self.backbone.forward_features(x)
        else:
            features_dict = self.backbone.forward_features(x)
        
        # 使用 CLS token
        cls_token = features_dict['x_norm_clstoken']  # [B, dim]
        
        # 保存特征（用于triplet loss，不detach以保持梯度）
        features_for_triplet = cls_token
        
        # 多分类器预测
        predictions = []
        features = []
        
        x_cls = self.dropout(cls_token)
        for classifier in self.classifiers:
            pred = classifier(x_cls)
            predictions.append(pred)
            features.append(features_for_triplet)  # 每个分类器共享相同特征
        
        # 返回格式
        if self.training:
            if self.return_f:
                return predictions, features
            else:
                return predictions
        else:
            # 测试模式：返回特征堆叠 [B, dim, block]
            return torch.stack([cls_token] * self.block, dim=2)


def weights_init_kaiming(m):
    """Kaiming初始化（兼容原始代码）"""
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        if m.bias is not None:
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
    """分类器初始化（兼容原始代码）"""
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)


def make_dinov2_model(
    num_class,
    model_size='vitb14',
    block=4,
    return_f=False,
    freeze_backbone=False,
    use_cls_token=True,
    dropout=0.5
):
    """
    创建 DINOv2 模型的工厂函数（向后兼容）
    
    Args:
        num_class: 类别数量
        model_size: DINOv2 模型大小 (vits14/vitb14/vitl14/vitg14)
        block: 多分类器数量
        return_f: 是否返回特征（用于triplet loss）
        freeze_backbone: 是否冻结backbone
        use_cls_token: 使用CLS token (True) 或 patch token pooling (False)
        dropout: dropout率
    
    Returns:
        DINOv2 模型
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
        # 如果需要使用patch token pooling模式
        # 需要结合MCCG架构，这里暂时不支持独立使用
        raise NotImplementedError(
            "Patch token pooling mode should be used with MCCG architecture. "
            "Please use DINOv2Backbone directly in build_dinov2_mccg."
        )
    
    return model


# ============ 测试代码 ============
if __name__ == "__main__":
    import torch
    
    print("="*70)
    print("测试 DINOv2Backbone")
    print("="*70)
    
    # 创建模型
    model = DINOv2Backbone(
        model_size='vitb14',
        freeze_backbone=False,
        use_structure_aware=True  # 启用结构感知支持
    )
    
    # 测试输入
    batch_size = 2
    x = torch.randn(batch_size, 3, 224, 224)
    
    print("\n" + "="*70)
    print("测试 1: 原版模式（return_structure=False）")
    print("="*70)
    
    model.eval()
    with torch.no_grad():
        gap_feat, part_feat = model(x, return_structure=False)
    
    print(f"✅ 原版输出:")
    print(f"   gap_feature: {gap_feat.shape}")
    print(f"   part_features: {part_feat.shape}")
    
    print("\n" + "="*70)
    print("测试 2: 增强模式（return_structure=True，但未初始化结构模块）")
    print("="*70)
    
    with torch.no_grad():
        result = model(x, return_structure=True)
    
    # 应该仍然返回原版格式（因为 structure_module=None）
    if len(result) == 2:
        print(f"✅ 回退到原版输出（结构模块未初始化）")
        print(f"   gap_feature: {result[0].shape}")
        print(f"   part_features: {result[1].shape}")
    
    print("\n" + "="*70)
    print("测试 3: 增强模式（初始化结构模块后）")
    print("="*70)
    
    # 手动初始化结构感知模块（通常由 BuildDINOv2MCCG 完成）
    from structure_alignment import StructureAwareModule
    model.structure_module = StructureAwareModule(
        feat_dim=768,
        num_parts=4
    )
    
    with torch.no_grad():
        gap_feat, part_feat, struct_feat, attn_maps, align_embed = model(
            x, return_structure=True
        )
    
    print(f"✅ 增强模式输出:")
    print(f"   gap_feature: {gap_feat.shape}")
    print(f"   part_features: {part_feat.shape}")
    print(f"   struct_feat: {struct_feat.shape}")
    print(f"   attn_maps: {attn_maps.shape}")
    print(f"   align_embed: {align_embed.shape}")
    
    print("\n" + "="*70)
    print("✅ 所有测试通过！向后兼容性验证成功")
    print("="*70)
