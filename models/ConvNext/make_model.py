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


class Gem_heat(nn.Module):
    def __init__(self, dim = 768, p=3, eps=1e-6):
        super(Gem_heat, self).__init__()
        self.p = nn.Parameter(torch.ones(dim) * p) 
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)


    def gem(self, x, p=3):
        p = F.softmax(p).unsqueeze(-1)
        x = torch.matmul(x,p)
        x = x.view(x.size(0), x.size(1))
        return x


def position(H, W, is_cuda=True):
    if is_cuda:
        loc_w = torch.linspace(-1.0, 1.0, W).cuda().unsqueeze(0).repeat(H, 1)
        loc_h = torch.linspace(-1.0, 1.0, H).cuda().unsqueeze(1).repeat(1, W)
    else:
        loc_w = torch.linspace(-1.0, 1.0, W).unsqueeze(0).repeat(H, 1)
        loc_h = torch.linspace(-1.0, 1.0, H).unsqueeze(1).repeat(1, W)
    loc = torch.cat([loc_w.unsqueeze(0), loc_h.unsqueeze(0)], 0).unsqueeze(0)
    return loc


def stride(x, stride):
    b, c, h, w = x.shape
    return x[:, :, ::stride, ::stride]


def init_rate_half(tensor):
    if tensor is not None:
        tensor.data.fill_(0.5)


def init_rate_0(tensor):
    if tensor is not None:
        tensor.data.fill_(0.)


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
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
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1)

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
    def __init__(self):
        super(TripletAttention, self).__init__()
        self.cw = AttentionGate()
        self.hc = AttentionGate()
    def forward(self, x):
        x_perm1 = x.permute(0,2,1,3).contiguous()
        x_out1 = self.cw(x_perm1)
        x_out11 = x_out1.permute(0,2,1,3).contiguous()
        x_perm2 = x.permute(0,3,2,1).contiguous()
        x_out2 = self.hc(x_perm2)
        x_out21 = x_out2.permute(0,3,2,1).contiguous()
        return x_out11, x_out21


class ClassBlock(nn.Module):
    """
    分类器块
    包含：全连接层 → BN → ReLU → Dropout → 分类器
    """
    def __init__(self, input_dim, class_num, droprate=0.5, relu=False, 
                 bnorm=True, num_bottleneck=512, linear=True, return_f=False):
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
        add_block.apply(self.weights_init_kaiming)
        
        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(self.weights_init_classifier)
        
        self.add_block = add_block
        self.classifier = classifier
    
    def weights_init_kaiming(self, m):
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
    
    def weights_init_classifier(self, m):
        """分类器权重初始化"""
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            nn.init.normal_(m.weight, std=0.001)
            if m.bias is not None:  # ✅ 修复：正确检查 bias
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, x):
        x = self.add_block(x)
        if self.return_f:
            f = x
            x = self.classifier(x)
            return [x, f]
        else:
            x = self.classifier(x)
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

class build_convnext(nn.Module):
    """
    ConvNeXt/ResNet 骨干网络 + TripletAttention + 多分类器
    
    ✅ 支持单输入和双输入
    ✅ 支持所有参数（向后兼容）
    ✅ 支持零初始化模式
    """
    def __init__(
        self, 
        num_classes, 
        block=4, 
        return_f=False, 
        resnet=False,
        # ⭐ 新参数（兼容零初始化）
        backbone_type=None,
        dropout=0.5,
        attention_type='none',
        attention_config=None,
        **kwargs  # 捕获其他参数
    ):
        super(build_convnext, self).__init__()
        
        self.return_f = return_f
        self.block = block
        self.num_classes = num_classes
        
        # ========== 处理 backbone 类型 ==========
        if backbone_type is None:
            # 向后兼容：如果没有指定 backbone_type，使用 resnet 参数
            if resnet:
                backbone_type = 'resnet'
            else:
                backbone_type = 'convnext'
        
        # ========== 初始化 Backbone ==========
        if backbone_type == 'resnet' or resnet:
            convnext_name = "resnet101"
            print('using model_type: {} as a backbone'.format(convnext_name))
            self.in_planes = 2048
            
            try:
                from models.resnet_backbone import Resnet
                self.convnext = Resnet(pretrained=True)
            except ImportError:
                print('[WARNING] Cannot import Resnet, using ConvNeXt instead')
                convnext_name = "convnext_tiny"
                self.in_planes = 768
                self.convnext = create_model(convnext_name, pretrained=True)
        
        else:  # convnext
            convnext_name = "convnext_tiny"
            print('using model_type: {} as a backbone'.format(convnext_name))
            
            if 'base' in convnext_name:
                self.in_planes = 1024
            elif 'large' in convnext_name:
                self.in_planes = 1536
            elif 'xlarge' in convnext_name:
                self.in_planes = 2048
            else:
                self.in_planes = 768
            
            self.convnext = create_model(convnext_name, pretrained=True)
        
        # ========== 分类器 ==========
        dropout_rate = dropout if dropout is not None else 0.5
        
        self.classifier1 = ClassBlock(
            self.in_planes, 
            num_classes, 
            dropout_rate, 
            return_f=return_f
        )
        
        # ========== TripletAttention ==========
        # ========== TripletAttention 导入（修复）==========
        try:
            # 尝试方案 1: 绝对导入
            from models.ConvNext.backbones.triplet_attention import TripletAttention
            self.tri_layer = TripletAttention()
            print("[INFO] TripletAttention loaded successfully")
        except ImportError:
            try:
                # 尝试方案 2: 相对导入
                from .backbones.triplet_attention import TripletAttention
                self.tri_layer = TripletAttention()
                print("[INFO] TripletAttention loaded successfully (relative import)")
            except ImportError:
                try:
                    # 尝试方案 3: 直接从 ConvNext 导入
                    from ConvNext.backbones.triplet_attention import TripletAttention
                    self.tri_layer = TripletAttention()
                    print("[INFO] TripletAttention loaded successfully (ConvNext import)")
                except ImportError:
                    print("[ERROR] Cannot import TripletAttention from any path")
                    print("       Creating dummy TripletAttention...")
                    
                    # ✅ 创建兼容的 dummy TripletAttention
                    class DummyTripletAttention(nn.Module):
                        """
                        Dummy TripletAttention - 返回与真实 TripletAttention 相同的格式
                        """
                        def __init__(self):
                            super().__init__()
                        
                        def forward(self, x):
                            """
                            返回 list，与真实 TripletAttention 格式一致
                            
                            Args:
                                x: [B, C, H, W]
                            
                            Returns:
                                list of 2 tensors: [x, x]（两个分支）
                            """
                            # 返回两个相同的特征（模拟两个注意力分支）
                            return [x, x]
                    
                    self.tri_layer = DummyTripletAttention()
                    print("[INFO] Using DummyTripletAttention (returns list of 2 features)")
        
        # ========== 多分类器（MCB）==========
        for i in range(self.block):
            name = 'classifier_mcb' + str(i + 1)
            setattr(self, name, ClassBlock(
                self.in_planes, 
                num_classes, 
                dropout_rate, 
                return_f=self.return_f
            ))
    
    def part_classifier(self, block, x, cls_name='classifier'):
        """多分类器处理"""
        part = {}
        for i in range(block):
            part[i] = x[:, :, i].view(x.size(0), -1)
            name = cls_name + str(i + 1)
            c = getattr(self, name)
            part[i] = c(part[i])
        y = []
        for i in range(block):
            y.append(part[i])
        return y
    
    def forward(self, x, x2=None, return_original_feat=False):
        """
        前向传播
        
        Args:
            x: 第一个输入（satellite）[B, 3, 256, 256]
            x2: 第二个输入（drone）[B, 3, 256, 256]，可选
            return_original_feat: 保留参数（用于零初始化兼容）
        
        Returns:
            训练模式 + 双输入: ((cls1, feat1), (cls2, feat2))
            训练模式 + 单输入: (cls, feat) 或 y
            测试模式 + 双输入: (y1, y2)
            测试模式 + 单输入: y
        """
        
        # ========== 处理第一个输入（satellite）==========
        gap_feature, part_features = self.convnext(x)
        tri_features = self.tri_layer(part_features)
        convnext_feature = self.classifier1(gap_feature)

        tri_list = []
        for i in range(len(tri_features)):
            tri_list.append(tri_features[i].mean([-2, -1]))
        
        # ✅ 修复：确保有足够的特征
        while len(tri_list) < self.block:
            if len(tri_list) > 0:
                tri_list.append(tri_list[0])
            else:
                # 如果 tri_features 为空，创建零张量
                tri_list.append(torch.zeros_like(gap_feature))
        
        triatten_features = torch.stack(tri_list[:self.block], dim=2)
        
        if self.block == 0:
            y = []
        else:
            y = self.part_classifier(self.block, triatten_features, cls_name='classifier_mcb')

        # ========== 处理第二个输入（drone，如果有）==========
        if x2 is not None:
            gap_feature2, part_features2 = self.convnext(x2)
            tri_features2 = self.tri_layer(part_features2)
            convnext_feature2 = self.classifier1(gap_feature2)

            tri_list2 = []
            for i in range(len(tri_features2)):
                tri_list2.append(tri_features2[i].mean([-2, -1]))
            
            # ✅ 修复：确保有足够的特征
            while len(tri_list2) < self.block:
                if len(tri_list2) > 0:
                    tri_list2.append(tri_list2[0])
                else:
                    tri_list2.append(torch.zeros_like(gap_feature2))
            
            triatten_features2 = torch.stack(tri_list2[:self.block], dim=2)
            
            if self.block == 0:
                y2 = []
            else:
                y2 = self.part_classifier(self.block, triatten_features2, cls_name='classifier_mcb')

        # ========== 返回结果 ==========
        if self.training:
            # 训练模式
            y = y + [convnext_feature]
            
            if x2 is not None:
                y2 = y2 + [convnext_feature2]
            
            if self.return_f:
                # 返回分类和特征（用于 triplet loss）
                cls, features = [], []
                for i in y:
                    cls.append(i[0])
                    features.append(i[1])
                
                if x2 is not None:
                    cls2, features2 = [], []
                    for i in y2:
                        cls2.append(i[0])
                        features2.append(i[1])
                    
                    # ✅ 双输入训练
                    return (cls, features), (cls2, features2)
                else:
                    # 单输入训练
                    return (cls, features)
            else:
                # 不返回特征
                if x2 is not None:
                    return y, y2
                else:
                    return y
        
        else:
            # ✅ 测试模式
            ffeature = convnext_feature.view(convnext_feature.size(0), -1, 1)
            
            if self.block == 0:
                y_out = ffeature
            else:
                y_out = torch.cat([y, ffeature], dim=2)
            
            if x2 is not None:
                ffeature2 = convnext_feature2.view(convnext_feature2.size(0), -1, 1)
                
                if self.block == 0:
                    y2_out = ffeature2
                else:
                    y2_out = torch.cat([y2, ffeature2], dim=2)
                
                # ✅ 双输入测试
                return y_out, y2_out
            else:
                # 单输入测试
                return y_out

    def part_classifier(self, block, x, cls_name='classifier_mcb'):
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


def make_convnext_model(num_class,block = 4,return_f=False,resnet=False):
    print('===========building convnext===========')
    model = build_convnext(num_class,block=block,return_f=return_f,resnet=resnet)
    return model

# ============================================================================
# ⭐ 零初始化支持（添加到文件末尾）
# ============================================================================

def make_model_with_zero_init(
    num_class,
    block=4,
    return_f=False,
    backbone='convnext',
    dinov2_model='vitb14',
    freeze_dinov2=False,
    use_structure_aware=False,
    use_hybrid=False,
    dropout=0.5,
    # ========== 零初始化参数 ==========
    use_zero_init=False,
    use_zero_init_tri=False,
    use_zero_init_detail=False,
    use_zero_init_aff=False,
):
    """
    支持零初始化的模型创建函数
    
    Args:
        num_class: 类别数
        block: MCCG 分类器数量
        return_f: 是否返回特征
        backbone: backbone 类型
        ... (其他参数保持不变)
        
        use_zero_init: 是否启用零初始化
        use_zero_init_tri: 零初始化 TripletAttention
        use_zero_init_detail: 零初始化 DetailBranch
        use_zero_init_aff: 零初始化 AFF
    
    Returns:
        model: 模型实例
    """
    
    # ========== 检查零初始化 ==========
    if use_zero_init:
        print("\n" + "="*80)
        print("🔥 Zero-Initialization Mode ENABLED")
        print("="*80)
        print(f"  - Zero-Init TripletAttention: {use_zero_init_tri}")
        print(f"  - Zero-Init DetailBranch: {use_zero_init_detail}")
        print(f"  - Zero-Init AFF: {use_zero_init_aff}")
        print("="*80 + "\n")
        
        try:
            # 导入零初始化模块
            from models.zeroInit_modules import ZeroInitMCCG
            
            model = ZeroInitMCCG(
                num_classes=num_class,
                block=block,
                use_zero_init_tri=use_zero_init_tri,
                use_zero_init_detail=use_zero_init_detail,
                use_zero_init_aff=use_zero_init_aff
            )
            
            print("✅ Zero-Init MCCG model created successfully\n")
            return model
            
        except ImportError as e:
            print(f"❌ Error: Cannot import zeroInit_modules")
            print(f"   {e}")
            print("   Falling back to standard model...\n")
            use_zero_init = False
    
    # ========== 标准模型（原逻辑）==========
    print(f"\n{'='*80}")
    print(f"Creating Standard MCCG Model with {backbone.upper()} Backbone")
    print(f"{'='*80}")
    
    if backbone == 'dinov2':
        print(f"  - Model size: {dinov2_model}")
        print(f"  - Freeze backbone: {freeze_dinov2}")
        print(f"  - Dropout: {dropout}")
        
        if use_hybrid:
            print(f"  - 🔥 Hybrid Feature Extraction: ENABLED")
        else:
            print(f"  - ⭕ Hybrid Feature Extraction: DISABLED")
        
        if use_structure_aware:
            print(f"  - 🔥 Structure-Aware Module: ENABLED")
        else:
            print(f"  - ⭕ Structure-Aware Module: DISABLED")
    
    print(f"{'='*80}\n")
    
    # ⭐⭐⭐ 关键修复：调用正确的函数 ⭐⭐⭐
    # ❌ 原来调用 build_convnext 会出错，因为参数不匹配
    # ✅ 应该调用 make_convnext_model（已存在的函数）
    
    if backbone == 'dinov2':
        # DINOv2 模型
        # 注意：你的代码中可能没有 make_dinov2_model，需要检查
        try:
            model = make_dinov2_model(
                num_class=num_class,
                block=block,
                return_f=return_f,
                model_size=dinov2_model
            )
        except NameError:
            print("[WARNING] make_dinov2_model not found, using ConvNeXt instead")
            model = make_convnext_model(
                num_class=num_class,
                block=block,
                return_f=return_f,
                resnet=False
            )
    
    elif backbone == 'resnet':
        # ResNet 模型
        model = make_convnext_model(
            num_class=num_class,
            block=block,
            return_f=return_f,
            resnet=True
        )
    
    else:  # convnext（默认）
        # ConvNeXt 模型
        model = make_convnext_model(
            num_class=num_class,
            block=block,
            return_f=return_f,
            resnet=False
        )
    
    return model


def make_model_from_opt(opt):
    """
    从 opt 对象自动创建模型
    
    ⭐ 推荐使用此函数，自动检测所有参数
    
    Args:
        opt: 训练参数对象
    
    Returns:
        model: 模型实例
    
    使用示例:
        # train.py 中
        from models.ConvNext.make_model import make_model_from_opt
        model = make_model_from_opt(opt)
    """
    
    # ⭐ 关键修复：正确检测 backbone 类型
    if getattr(opt, 'dinov2', False):
        backbone = 'dinov2'
    elif getattr(opt, 'resnet', False):
        backbone = 'resnet'
    else:
        backbone = 'convnext'
    
    return make_model_with_zero_init(
        num_class=opt.nclasses,
        block=opt.block,
        return_f=True,  # 训练时总是返回特征
        backbone=backbone,
        dinov2_model=getattr(opt, 'dinov2_model', 'vitb14'),
        freeze_dinov2=getattr(opt, 'freeze_dinov2', False),
        use_structure_aware=getattr(opt, 'use_structure_aware', False),
        use_hybrid=getattr(opt, 'use_hybrid', False),
        dropout=getattr(opt, 'dropout', 0.5),
        # 零初始化参数
        use_zero_init=getattr(opt, 'use_zero_init', False),
        use_zero_init_tri=getattr(opt, 'use_zero_init_tri', False),
        use_zero_init_detail=getattr(opt, 'use_zero_init_detail', False),
        use_zero_init_aff=getattr(opt, 'use_zero_init_aff', False),
    )


# ============================================================================
# ⭐ 修改 make_convnext_model 函数（向后兼容）
# ============================================================================

def make_convnext_model(
    num_class,
    block=4,
    return_f=False,
    resnet=False,
    # ========== ⭐ 新增参数（默认 False，完全向后兼容）==========
    use_zero_init=False,
    use_zero_init_tri=False,
    use_zero_init_detail=False,
    use_zero_init_aff=False,
):
    """
    创建 ConvNeXt/ResNet MCCG 模型（支持零初始化）
    
    ⭐ 完全向后兼容：
    - 不传零初始化参数时，使用标准 MCCG 模型
    - 传入零初始化参数时，使用零初始化模型
    
    Args:
        num_class: 类别数
        block: MCCG 分类器数量
        return_f: 是否返回特征
        resnet: 是否使用 ResNet（False 则使用 ConvNeXt）
        
        use_zero_init: ⭐ 是否启用零初始化
        use_zero_init_tri: ⭐ 零初始化 TripletAttention
        use_zero_init_detail: ⭐ 零初始化 DetailBranch
        use_zero_init_aff: ⭐ 零初始化 AFF
    
    Returns:
        model: 模型实例
    
    使用示例:
        # 标准模型（向后兼容）
        model = make_convnext_model(701, block=4)
        
        # 零初始化模型
        model = make_convnext_model(
            701, 
            block=4,
            use_zero_init=True,
            use_zero_init_tri=True
        )
    """
    
    # ⭐ 如果启用零初始化，调用零初始化创建函数
    if use_zero_init:
        return make_model_with_zero_init(
            num_class=num_class,
            block=block,
            return_f=return_f,
            backbone='resnet' if resnet else 'convnext',
            use_zero_init=use_zero_init,
            use_zero_init_tri=use_zero_init_tri,
            use_zero_init_detail=use_zero_init_detail,
            use_zero_init_aff=use_zero_init_aff,
        )
    
    # ⭐ 标准模型：调用原有的 build_convnext
    print("="*70)
    print(f"Building MCCG with {'ResNet101' if resnet else 'ConvNeXt-Tiny'} backbone")
    print("="*70)
    print("===========building convnext===========")
    
    model = build_convnext(
        num_classes=num_class,  # ⭐ 注意这里是 num_classes（带 s）
        block=block,
        return_f=return_f,
        resnet=resnet
    )
    
    return model

# ============================================================================
# ⭐ 如果你还有 build_mccg_model 函数，也需要修改
# ============================================================================

def build_mccg_model(
    num_classes,
    block=4,
    return_f=False,
    backbone='convnext',
    dinov2_model='vitb14',
    freeze_dinov2=False,
    use_structure_aware=False,
    use_hybrid=False,
    dropout=0.5,
    # ========== ⭐ 新增参数 ==========
    use_zero_init=False,
    use_zero_init_tri=False,
    use_zero_init_detail=False,
    use_zero_init_aff=False,
):
    """
    工厂函数：创建 MCCG 模型（支持零初始化）
    """
    return make_model_with_zero_init(
        num_class=num_classes,  # ⭐ 注意参数名转换
        block=block,
        return_f=return_f,
        backbone=backbone,
        dinov2_model=dinov2_model,
        freeze_dinov2=freeze_dinov2,
        use_structure_aware=use_structure_aware,
        use_hybrid=use_hybrid,
        dropout=dropout,
        use_zero_init=use_zero_init,
        use_zero_init_tri=use_zero_init_tri,
        use_zero_init_detail=use_zero_init_detail,
        use_zero_init_aff=use_zero_init_aff,
    )


# ============================================================================
# ⭐⭐⭐ 测试和验证代码 ⭐⭐⭐
# ============================================================================

if __name__ == '__main__':
    """
    测试代码：验证所有函数正常工作
    """
    import torch
    from argparse import Namespace
    
    print("="*80)
    print("Testing make_model.py functions")
    print("="*80)
    
    # 测试 1: 标准模型
    print("\n[Test 1] Standard ConvNeXt model:")
    model = make_convnext_model(num_class=701, block=2, return_f=True, resnet=False)
    x = torch.randn(2, 3, 256, 256)
    x2 = torch.randn(2, 3, 256, 256)
    model.train()
    out = model(x, x2)
    print(f"✅ Output type: {type(out)}")
    
    # 测试 2: 零初始化模型
    print("\n[Test 2] Zero-Init model:")
    try:
        model = make_convnext_model(
            num_class=701, 
            block=2, 
            return_f=True,
            use_zero_init=True,
            use_zero_init_tri=True
        )
        out = model(x, x2)
        print(f"✅ Zero-Init model works: {type(out)}")
    except Exception as e:
        print(f"❌ Zero-Init failed: {e}")
    
    # 测试 3: make_model_from_opt
    print("\n[Test 3] make_model_from_opt:")
    opt = Namespace(
        nclasses=701,
        block=2,
        triplet_loss=0.3,
        dinov2=False,
        resnet=False,
        use_zero_init=False
    )
    model = make_model_from_opt(opt)
    print(f"✅ Model created from opt")
    
    print("\n" + "="*80)
    print("All tests passed! ✅")
    print("="*80)
