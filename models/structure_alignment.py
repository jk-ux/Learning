# -*- coding: utf-8 -*-
"""
结构感知对齐模块
功能：
1. 边缘检测（可学习 Sobel 算子）
2. 多部分结构注意力
3. 跨视角结构对齐
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class StructureAwareModule(nn.Module):
    """
    结构感知模块
    捕捉建筑轮廓、道路等稳定结构特征
    """
    def __init__(self, feat_dim=512, num_parts=4):
        super().__init__()
        
        self.num_parts = num_parts
        self.feat_dim = feat_dim
        
        # 1. 可学习的边缘检测器
        self.edge_detector = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Sigmoid()
        )
        
        # 初始化为 Sobel-like 边缘检测
        # （可选：可以用预训练的边缘检测器）
        
        # 2. 结构注意力生成器（多部分）
        self.structure_attn_gen = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim // 4, 1),
            nn.BatchNorm2d(feat_dim // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_dim // 4, num_parts, 1),
            nn.Sigmoid()
        )
        
        # 3. 每个部分的特征聚合器
        self.part_aggregators = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(feat_dim, feat_dim // num_parts),
                nn.ReLU()
            ) for _ in range(num_parts)
        ])
        
        # 4. 跨视角对齐投影（用于对比学习）
        self.alignment_proj = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, 256)  # 对比学习嵌入空间
        )
        
    def forward(self, feat_map, image=None):
        """
        Args:
            feat_map: [B, C, H, W] 特征图
            image: [B, 3, H', W'] 原始图像（用于边缘检测）
        
        Returns:
            struct_feat: [B, C] 结构感知特征
            attn_maps: [B, num_parts, H, W] 多部分注意力图
            align_embed: [B, 256] 对齐嵌入（用于对比学习）
        """
        B, C, H, W = feat_map.shape
        
        # 1. 提取边缘先验（如果提供了原始图像）
        edge_map = None
        if image is not None:
            edge_map = self.edge_detector(image)  # [B, 1, H', W']
            # 调整到特征图尺寸
            edge_map = F.interpolate(edge_map, size=(H, W), mode='bilinear', align_corners=False)
        
        # 2. 生成多部分结构注意力
        attn_maps = self.structure_attn_gen(feat_map)  # [B, num_parts, H, W]
        
        # 3. 如果有边缘先验，融合增强
        if edge_map is not None:
            # 边缘区域的注意力加权增强
            enhanced_attn = attn_maps * (1.0 + edge_map)  # [B, num_parts, H, W]
        else:
            enhanced_attn = attn_maps
        
        # 4. 基于注意力提取各部分特征
        part_features = []
        for i in range(self.num_parts):
            # 应用第 i 个注意力图
            attn_i = enhanced_attn[:, i:i+1, :, :]  # [B, 1, H, W]
            weighted_feat = feat_map * attn_i  # [B, C, H, W]
            
            # 聚合该部分特征
            part_feat_i = self.part_aggregators[i](weighted_feat)  # [B, C//num_parts]
            part_features.append(part_feat_i)
        
        # 5. 拼接所有部分特征
        struct_feat = torch.cat(part_features, dim=1)  # [B, C]
        
        # 6. 对齐投影（用于对比学习）
        align_embed = self.alignment_proj(struct_feat)  # [B, 256]
        
        return struct_feat, enhanced_attn, align_embed


# ============================================================================
# 测试代码（可选）
# ============================================================================
if __name__ == "__main__":
    # 测试结构感知模块
    batch_size = 4
    feat_dim = 512
    feat_map = torch.randn(batch_size, feat_dim, 7, 7)
    image = torch.randn(batch_size, 3, 224, 224)
    
    module = StructureAwareModule(feat_dim=feat_dim, num_parts=4)
    
    struct_feat, attn_maps, align_embed = module(feat_map, image)
    
    print(f"✅ Structure-Aware Module Test:")
    print(f"   Input feat_map: {feat_map.shape}")
    print(f"   Input image: {image.shape}")
    print(f"   Output struct_feat: {struct_feat.shape}")
    print(f"   Output attn_maps: {attn_maps.shape}")
    print(f"   Output align_embed: {align_embed.shape}")
