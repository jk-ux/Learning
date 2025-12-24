# -*- coding: utf-8 -*-
"""
测试结构感知注意力网络
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# 3.3.1 关键结构信息提取 (Key Structure Information Extraction)
# ============================================================================

class StructureExtractor(nn.Module):
    """
    关键结构信息提取器
    
    功能：
    1. 边缘检测（Sobel）
    2. 角点检测（Harris-like）
    3. 方向梯度（HOG-like）
    """
    def __init__(self, in_channels=3):
        super().__init__()
        
        # ========== 1. 可学习的 Sobel 边缘检测器 ==========
        self.edge_detector = nn.Conv2d(
            in_channels, 2, kernel_size=3, padding=1, bias=False
        )
        
        # 初始化为 Sobel 算子
        sobel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32) / 4.0
        
        sobel_y = torch.tensor([
            [-1, -2, -1],
            [ 0,  0,  0],
            [ 1,  2,  1]
        ], dtype=torch.float32) / 4.0
        
        # ⭐ 修正初始化逻辑
        with torch.no_grad():
            # Sobel X: [out_channels=1, in_channels=3, 3, 3]
            for c in range(in_channels):
                self.edge_detector.weight.data[0, c, :, :] = sobel_x
                self.edge_detector.weight.data[1, c, :, :] = sobel_y
        
        # ========== 2. Harris 角点检测器 ==========
        self.corner_detector = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
        # ========== 3. 方向梯度特征 ==========
        self.gradient_encoder = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 8, kernel_size=1),  # 8 个方向 bin
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] 输入图像或特征图
        
        Returns:
            edge_map: [B, 1, H, W] 边缘图
            corner_map: [B, 1, H, W] 角点图
            gradient_map: [B, 8, H, W] 方向梯度图
        """
        # 1. 边缘检测
        edges = self.edge_detector(x)  # [B, 2, H, W]
        edge_magnitude = torch.sqrt(edges[:, 0:1]**2 + edges[:, 1:2]**2 + 1e-8)
        edge_map = edge_magnitude  # [B, 1, H, W]
        
        # 2. 角点检测
        corner_map = self.corner_detector(edges)  # [B, 1, H, W]
        
        # 3. 方向梯度
        gradient_map = self.gradient_encoder(edges)  # [B, 8, H, W]
        
        return edge_map, corner_map, gradient_map


# ============================================================================
# 3.3.2 跨层级结构感知注意力模块设计
# ============================================================================

class CrossLevelStructureAttention(nn.Module):
    """
    跨层级结构感知注意力模块
    
    功能：
    1. 多尺度结构信息融合
    2. 跨视图结构对齐
    3. 自适应权重学习
    """
    def __init__(self, feat_dim=768, num_levels=3, reduction=16):
        """
        Args:
            feat_dim: 特征维度
            num_levels: 金字塔层级数
            reduction: 通道压缩比例
        """
        super().__init__()
        
        self.feat_dim = feat_dim
        self.num_levels = num_levels
        
        # ========== 1. 多尺度特征金字塔 ==========
        self.pyramid_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(feat_dim, feat_dim, kernel_size=3, 
                         stride=2**i if i > 0 else 1, 
                         padding=1),
                nn.BatchNorm2d(feat_dim),
                nn.ReLU(inplace=True)
            ) for i in range(num_levels)
        ])
        
        # ========== 2. 结构信息聚合 ==========
        # 整合边缘、角点、梯度信息
        self.structure_aggregator = nn.Sequential(
            nn.Conv2d(1 + 1 + 8, 32, kernel_size=3, padding=1),  # edge + corner + gradient
            nn.ReLU(inplace=True),
            nn.Conv2d(32, feat_dim, kernel_size=1),
            nn.Sigmoid()
        )
        
        # ========== 3. 跨视图注意力 ==========
        self.cross_view_attn = nn.MultiheadAttention(
            embed_dim=feat_dim,
            num_heads=8,
            batch_first=True
        )
        
        # ========== 4. 通道注意力（SE-like）==========
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(feat_dim, feat_dim // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_dim // reduction, feat_dim, 1),
            nn.Sigmoid()
        )
        
        # ========== 5. 空间注意力 ==========
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        
        # ========== 6. 层级融合权重 ==========
        self.level_weights = nn.Parameter(torch.ones(num_levels) / num_levels)
        
        # ========== 7. 最终投影 ==========
        self.output_proj = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim, kernel_size=1),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, feat_map, edge_map, corner_map, gradient_map, view2_feat=None):
        """
        Args:
            feat_map: [B, C, H, W] 主特征图
            edge_map: [B, 1, H, W] 边缘图
            corner_map: [B, 1, H, W] 角点图
            gradient_map: [B, 8, H, W] 方向梯度图
            view2_feat: [B, C, H, W] 第二视图特征（可选）
        
        Returns:
            enhanced_feat: [B, C, H, W] 增强后的特征
            attention_maps: dict 包含各种注意力图
        """
        B, C, H, W = feat_map.shape
        
        # ========== 1. 多尺度特征提取 ==========
        multi_scale_feats = []
        for i, conv in enumerate(self.pyramid_convs):
            scale_feat = conv(feat_map)  # 不同尺度
            # 上采样回原尺寸
            scale_feat = F.interpolate(scale_feat, size=(H, W), mode='bilinear', align_corners=False)
            multi_scale_feats.append(scale_feat)
        
        # 加权融合多尺度特征
        weights = F.softmax(self.level_weights, dim=0)
        fused_feat = sum(w * f for w, f in zip(weights, multi_scale_feats))
        
        # ========== 2. 结构信息聚合 ==========
        # ⭐ 首先将结构信息上采样到特征图尺寸
        edge_map_resized = F.interpolate(edge_map, size=(H, W), mode='bilinear', align_corners=False)
        corner_map_resized = F.interpolate(corner_map, size=(H, W), mode='bilinear', align_corners=False)
        gradient_map_resized = F.interpolate(gradient_map, size=(H, W), mode='bilinear', align_corners=False)
        
        # 拼接结构信息
        structure_info = torch.cat([edge_map_resized, corner_map_resized, gradient_map_resized], dim=1)  # [B, 10, H, W]
        
        # 生成结构注意力图
        structure_attn = self.structure_aggregator(structure_info)  # [B, C, H, W]
        
        # 应用结构注意力
        struct_enhanced_feat = fused_feat * structure_attn
        
        # ========== 3. 跨视图注意力（如果提供第二视图）==========
        if view2_feat is not None:
            # Reshape for attention: [B, C, H, W] -> [B, H*W, C]
            feat1_flat = struct_enhanced_feat.flatten(2).transpose(1, 2)  # [B, HW, C]
            feat2_flat = view2_feat.flatten(2).transpose(1, 2)            # [B, HW, C]
            
            # Cross-attention
            cross_attn_out, cross_attn_weights = self.cross_view_attn(
                query=feat1_flat,
                key=feat2_flat,
                value=feat2_flat
            )
            
            # Reshape back: [B, HW, C] -> [B, C, H, W]
            cross_attn_out = cross_attn_out.transpose(1, 2).reshape(B, C, H, W)
            
            # 残差连接
            struct_enhanced_feat = struct_enhanced_feat + cross_attn_out
        else:
            cross_attn_weights = None
        
        # ========== 4. 通道注意力 ==========
        channel_attn = self.channel_attention(struct_enhanced_feat)  # [B, C, 1, 1]
        channel_refined_feat = struct_enhanced_feat * channel_attn
        
        # ========== 5. 空间注意力 ==========
        # 计算空间统计
        avg_pool = torch.mean(channel_refined_feat, dim=1, keepdim=True)  # [B, 1, H, W]
        max_pool, _ = torch.max(channel_refined_feat, dim=1, keepdim=True)  # [B, 1, H, W]
        spatial_input = torch.cat([avg_pool, max_pool], dim=1)  # [B, 2, H, W]
        
        spatial_attn = self.spatial_attention(spatial_input)  # [B, 1, H, W]
        spatial_refined_feat = channel_refined_feat * spatial_attn
        
        # ========== 6. 最终投影 ==========
        output_feat = self.output_proj(spatial_refined_feat)
        
        # 残差连接
        enhanced_feat = feat_map + output_feat
        
        # ========== 7. 收集注意力图（用于可视化）==========
        attention_maps = {
            'structure_attn': structure_attn,
            'channel_attn': channel_attn,
            'spatial_attn': spatial_attn,
            'cross_view_attn': cross_attn_weights,
            'edge_map': edge_map,
            'corner_map': corner_map
        }
        
        return enhanced_feat, attention_maps


# ============================================================================
# 完整的结构感知注意力网络
# ============================================================================

class StructureAwareAttentionNetwork(nn.Module):
    """
    完整的结构感知注意力网络
    
    整合 3.3.1 和 3.3.2 的所有功能
    """
    def __init__(self, feat_dim=768, num_levels=3, reduction=16):
        super().__init__()
        
        # 3.3.1 关键结构信息提取
        self.structure_extractor = StructureExtractor(in_channels=3)
        
        # 3.3.2 跨层级结构感知注意力
        self.cross_level_attn = CrossLevelStructureAttention(
            feat_dim=feat_dim,
            num_levels=num_levels,
            reduction=reduction
        )
        
        # 全局平均池化（用于生成全局特征）
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # 特征投影（用于对比学习）
        self.feature_proj = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256)
        )
    
    def forward(self, feat_map, raw_image, view2_feat=None, view2_image=None):
        """
        Args:
            feat_map: [B, C, H, W] 主特征图（来自 backbone）
            raw_image: [B, 3, H, W] 原始图像（用于提取结构信息）
            view2_feat: [B, C, H, W] 第二视图特征（可选）
            view2_image: [B, 3, H, W] 第二视图原始图像（可选）
        
        Returns:
            enhanced_feat: [B, C, H, W] 增强后的特征图
            global_feat: [B, C] 全局特征
            align_embed: [B, 256] 对齐嵌入（用于对比学习）
            attention_maps: dict 注意力图字典
        """
        B, C, H, W = feat_map.shape
        
        # ========== 1. 提取关键结构信息 ==========
        edge_map, corner_map, gradient_map = self.structure_extractor(raw_image)
        
        # 如果有第二视图，也提取其结构信息
        if view2_image is not None:
            edge_map2, corner_map2, gradient_map2 = self.structure_extractor(view2_image)
        else:
            edge_map2 = corner_map2 = gradient_map2 = None
        
        # ========== 2. 应用跨层级结构感知注意力 ==========
        enhanced_feat, attention_maps = self.cross_level_attn(
            feat_map=feat_map,
            edge_map=edge_map,
            corner_map=corner_map,
            gradient_map=gradient_map,
            view2_feat=view2_feat
        )
        
        # ========== 3. 生成全局特征 ==========
        global_feat = self.global_pool(enhanced_feat).view(B, -1)  # [B, C]
        
        # ========== 4. 生成对齐嵌入 ==========
        align_embed = self.feature_proj(global_feat)  # [B, 256]
        
        # ========== 5. 添加第二视图的结构信息到注意力图 ==========
        if view2_image is not None:
            attention_maps.update({
                'edge_map2': edge_map2,
                'corner_map2': corner_map2
            })
        
        return enhanced_feat, global_feat, align_embed, attention_maps


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("测试结构感知注意力网络")
    print("="*70)
    
    # 创建模型
    model = StructureAwareAttentionNetwork(
        feat_dim=768,
        num_levels=3,
        reduction=16
    )
    
    print(f"\n✅ 模型创建成功")
    
    # 模拟输入
    batch_size = 2
    feat_map = torch.randn(batch_size, 768, 16, 16)  # 来自 backbone 的特征图
    raw_image = torch.randn(batch_size, 3, 256, 256)  # 原始图像
    
    # 单视图测试
    print("\n[Test 1] 单视图处理")
    enhanced_feat, global_feat, align_embed, attn_maps = model(feat_map, raw_image)
    
    print(f"✅ Enhanced feature: {enhanced_feat.shape}")
    print(f"✅ Global feature: {global_feat.shape}")
    print(f"✅ Alignment embedding: {align_embed.shape}")
    print(f"✅ Attention maps: {len(attn_maps)} 个")
    
    # 双视图测试
    print("\n[Test 2] 双视图跨视图对齐")
    feat_map2 = torch.randn(batch_size, 768, 16, 16)
    raw_image2 = torch.randn(batch_size, 3, 256, 256)
    
    enhanced_feat, global_feat, align_embed, attn_maps = model(
        feat_map, raw_image,
        view2_feat=feat_map2,
        view2_image=raw_image2
    )
    
    print(f"✅ Enhanced feature: {enhanced_feat.shape}")
    print(f"✅ Cross-view attention: {'available' if attn_maps['cross_view_attn'] is not None else 'N/A'}")
    
    # 可视化注意力图
    print("\n[Test 3] 注意力图可视化")
    print("可用的注意力图：")
    for key, value in attn_maps.items():
        if value is not None and isinstance(value, torch.Tensor):
            print(f"  - {key}: {value.shape}")
    
    # 参数量统计
    print("\n[Test 4] 模型参数统计")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✅ 总参数量: {total_params:,}")
    print(f"✅ 可训练参数: {trainable_params:,}")
    
    print("\n" + "="*70)
    print("✅ 所有测试通过！")
    print("="*70)
