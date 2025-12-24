import torch.nn as nn
from .ConvNext import make_convnext_model
from .dinov2_backbone import make_dinov2_model
import torch
import torch.nn as nn

# 导入不同的模型实现
from .ConvNext.make_model import build_convnext, make_convnext_model
from .dinov2_backbone import DINOv2Backbone, make_dinov2_model

class two_view_net(nn.Module):
    def __init__(self, class_num, block=4, return_f=False, resnet=False, dinov2=False, dinov2_size='vitb14'):
        super(two_view_net, self).__init__()
        
        # 根据 backbone 类型选择模型
        if dinov2:
            self.model_1 = make_dinov2_model(
                num_class=class_num, 
                block=block, 
                return_f=return_f, 
                model_size=dinov2_size
            )
        else:
            self.model_1 = make_convnext_model(
                num_class=class_num, 
                block=block, 
                return_f=return_f, 
                resnet=resnet
            )

    def forward(self, x1, x2, return_f=False):
        if self.training:
            # 训练：必须双视图
            f1, p1 = self.model_1(x1)
            f2, p2 = self.model_2(x2)
            if return_f:
                return (p1, p2), (f1, f2)
            return p1, p2
        
        else:
            # ⭐ 测试：单视图
            if x1 is not None:
                f1, p1 = self.model_1(x1)
                return p1, f1 if return_f else p1
            else:
                f2, p2 = self.model_2(x2)
                return p2, f2 if return_f else p2


class three_view_net(nn.Module):
    def __init__(self, class_num, share_weight=False, block=4, return_f=False, resnet=False, dinov2=False, dinov2_size='vitb14'):
        super(three_view_net, self).__init__()
        self.share_weight = share_weight
        
        # 根据 backbone 类型选择模型
        if dinov2:
            self.model_1 = make_dinov2_model(
                num_class=class_num, 
                block=block, 
                return_f=return_f, 
                model_size=dinov2_size
            )
        else:
            self.model_1 = make_convnext_model(
                num_class=class_num, 
                block=block, 
                return_f=return_f, 
                resnet=resnet
            )

        if self.share_weight:
            self.model_2 = self.model_1
        else:
            if dinov2:
                self.model_2 = make_dinov2_model(
                    num_class=class_num, 
                    block=block, 
                    return_f=return_f, 
                    model_size=dinov2_size
                )
            else:
                self.model_2 = make_convnext_model(
                    num_class=class_num, 
                    block=block, 
                    return_f=return_f, 
                    resnet=resnet
                )

    def forward(self, x1, x2, x3, x4=None):  # x4 is extra data
        if x1 is None:
            y1 = None
        else:
            y1 = self.model_1(x1)

        if x2 is None:
            y2 = None
        else:
            y2 = self.model_2(x2)

        if x3 is None:
            y3 = None
        else:
            y3 = self.model_1(x3)

        if x4 is None:
            return y1, y2, y3
        else:
            y4 = self.model_2(x4)
        return y1, y2, y3, y4


def make_model(opt):
    """
    根据 opt 参数创建模型（兼容训练脚本）
    
    Args:
        opt: argparse.Namespace 对象，包含所有训练参数
    
    Returns:
        model: 初始化后的模型
    """
    num_classes = opt.nclasses
    block = opt.block
    return_f = (opt.triplet_loss > 0)  # 如果使用triplet loss，需要返回特征
    
    # ========== 根据参数选择 backbone ==========
    opt.dinov2 = getattr(opt, 'dinov2', False)
    if opt.dinov2:
        print("="*70)
        print(f"Building MCCG with DINOv2-{opt.dinov2_size} backbone")
        print(f"  - Model size: {opt.dinov2_size}")
        print(f"  - Use CLS token: {opt.use_cls_token}")
        print(f"  - Freeze backbone: {opt.freeze_backbone}")
        print(f"  - Dropout: {opt.dinov2_dropout}")
        print("="*70)
        
        # 创建 DINOv2 + MCCG 模型
        model = build_dinov2_mccg(
            num_classes=num_classes,
            model_size=opt.dinov2_size,
            block=block,
            return_f=return_f,
            freeze_backbone=opt.freeze_backbone,
            use_cls_token=opt.use_cls_token,
            dropout=opt.dinov2_dropout
        )
    
    elif opt.resnet:
        print("="*70)
        print("Building MCCG with ResNet101 backbone")
        print("="*70)
        
        model = make_convnext_model(
            num_class=num_classes,
            block=block,
            return_f=return_f,
            resnet=True
        )
    
    else:  # 默认使用 ConvNeXt
        print("="*70)
        print(f"Building MCCG with {opt.model} backbone")
        print("="*70)
        
        model = make_convnext_model(
            num_class=num_classes,
            block=block,
            return_f=return_f,
            resnet=False
        )
    
    return model


def build_dinov2_mccg(
    num_classes,
    model_size='vitb14',
    block=4,
    return_f=False,
    freeze_backbone=False,
    use_cls_token=False,
    dropout=0.5
):
    """
    创建 DINOv2 + MCCG 完整模型
    
    这个函数需要根据你的具体实现调整：
    1. 如果你使用的是我之前提供的 make_model_with_dinov2.py 中的实现，
       那么需要导入 BuildDINOv2 类
    2. 如果你使用的是原始的 dinov2_backbone.py，则需要修改
    
    这里假设你已经按照我之前的建议修改了 make_model.py
    """
    from .ConvNext.make_model import (
        ClassBlock, 
        TripletAttention,
        build_convnext  # 复用原始架构
    )
    
    # 创建 DINOv2Backbone
    class BuildDINOv2MCCG(nn.Module):
        """
        DINOv2 + MCCG 完整模型
        保留两种模式：
        1. use_cls_token=True: 使用集成的 DINOv2WithCLSToken
        2. use_cls_token=False: 使用 DINOv2Backbone + TripletAttention + ClassBlock
        """
        def __init__(
            self,
            num_classes,
            model_size='vitb14',
            block=4,
            return_f=False,
            freeze_backbone=False,
            use_cls_token=False,
            dropout=0.5
        ):
            super().__init__()
            
            self.return_f = return_f
            self.block = block
            self.num_classes = num_classes
            
            # ========== 模式1：使用集成模型 ==========
            if use_cls_token:
                from .dinov2_backbone import DINOv2WithCLSToken
                
                # ✅ 关键修复：backbone内部必须return_f=True
                # 这样才能获取特征用于triplet loss
                self.backbone = DINOv2WithCLSToken(
                    num_class=num_classes,
                    model_size=model_size,
                    block=block,
                    return_f=True,  # ✅ 强制返回特征
                    freeze_backbone=freeze_backbone,
                    dropout=dropout
                )
                self.use_integrated_model = True
                print(f"[INFO] Using DINOv2WithCLSToken (integrated model)")
            
            # ========== 模式2：使用分离架构 ==========
            else:
                from .dinov2_backbone import DINOv2Backbone
                
                self.backbone = DINOv2Backbone(
                    model_size=model_size,
                    freeze_backbone=freeze_backbone
                )
                self.use_integrated_model = False
                
                # 特征维度
                dim_mapping = {
                    'vits14': 384,
                    'vitb14': 768,
                    'vitl14': 1024,
                    'vitg14': 1536,
                }
                self.in_planes = dim_mapping.get(model_size, 768)
                
                # 主分类器
                self.classifier1 = ClassBlock(
                    self.in_planes,
                    num_classes,
                    dropout,
                    return_f=return_f
                )
                
                # TripletAttention
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
                
                print(f"[INFO] Using DINOv2Backbone + MCCG architecture")
        
        def forward(self, x, x2=None):
            """
            前向传播 - 支持2视图
            
            Args:
                x: satellite view [B, 3, H, W]
                x2: drone view [B, 3, H, W] (可选)
            
            Returns:
                训练模式:
                    if return_f: ((preds1, feats1), (preds2, feats2))
                    else: (preds1, preds2)
                测试模式:
                    (features1, features2)
            """
            # ========== 模式1：集成模型 ==========
            if self.use_integrated_model:
                # DINOv2WithCLSToken返回: (predictions_list, features_list)
                preds1, feats1 = self.backbone(x)
                
                if x2 is not None:
                    preds2, feats2 = self.backbone(x2)
                    
                    # ✅ 关键修复：根据 self.return_f 决定输出格式
                    if self.return_f:
                        # 训练时需要特征用于 triplet loss
                        return (preds1, feats1), (preds2, feats2)
                    else:
                        # 不需要 triplet loss，只返回预测
                        return preds1, preds2
                else:
                    if self.return_f:
                        return (preds1, feats1)
                    else:
                        return preds1
            
            # ========== 模式2：分离架构（完整MCCG） ==========
            else:
                # 提取backbone特征
                gap_feature, part_features = self.backbone(x)
                
                if x2 is not None:
                    gap_feature2, part_features2 = self.backbone(x2)
                
                # TripletAttention处理
                tri_features = self.tri_layer(part_features)
                convnext_feature = self.classifier1(gap_feature)
                
                if x2 is not None:
                    tri_features2 = self.tri_layer(part_features2)
                    convnext_feature2 = self.classifier1(gap_feature2)
                
                # 多分类器特征聚合
                tri_list = []
                for i in range(self.block):
                    tri_list.append(tri_features[i].mean([-2, -1]))
                triatten_features = torch.stack(tri_list, dim=2)
                
                if x2 is not None:
                    tri_list2 = []
                    for i in range(self.block):
                        tri_list2.append(tri_features2[i].mean([-2, -1]))
                    triatten_features2 = torch.stack(tri_list2, dim=2)
                
                # 部分分类器
                if self.block == 0:
                    y = []
                    if x2 is not None:
                        y2 = []
                else:
                    y = self.part_classifier(self.block, triatten_features, cls_name='classifier_mcb')
                    if x2 is not None:
                        y2 = self.part_classifier(self.block, triatten_features2, cls_name='classifier_mcb')
                
                # 返回结果
                if self.training:
                    y = y + [convnext_feature]
                    if x2 is not None:
                        y2 = y2 + [convnext_feature2]
                        
                    if self.return_f:
                        cls, features = [], []
                        cls2, features2 = [], []
                        for i in y:
                            cls.append(i[0])
                            features.append(i[1])
                        if x2 is not None:
                            for i in y2:
                                cls2.append(i[0])
                                features2.append(i[1])
                            return (cls, features), (cls2, features2)
                        else:
                            return (cls, features)
                    else:
                        if x2 is not None:
                            return y, y2
                        else:
                            return y
                else:
                    # 测试模式
                    ffeature = convnext_feature.view(convnext_feature.size(0), -1, 1)
                    y = torch.cat([y, ffeature], dim=2)
                    
                    if x2 is not None:
                        ffeature2 = convnext_feature2.view(convnext_feature2.size(0), -1, 1)
                        y2 = torch.cat([y2, ffeature2], dim=2)
                        return y, y2
                    else:
                        return y
        
        def part_classifier(self, block, x, cls_name='classifier_mcb'):
            """部分分类器 - 与原始代码完全一致"""
            part = {}
            predict = {}
            for i in range(block):
                part[i] = x[:, :, i].view(x.size(0), -1)
                name = cls_name + str(i+1)
                c = getattr(self, name)
                predict[i] = c(part[i])
            y = []
            for i in range(block):
                y.append(predict[i])
            if not self.training:
                return torch.stack(y, dim=2)
            return y
        
        # 创建并返回模型
    model = BuildDINOv2MCCG(
        num_classes=num_classes,
        model_size=model_size,
        block=block,
        return_f=return_f,
        freeze_backbone=freeze_backbone,
        use_cls_token=use_cls_token,
        dropout=dropout
        )
    
    return model
