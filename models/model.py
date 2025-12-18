import torch.nn as nn
from .ConvNext import make_convnext_model
from .dinov2_backbone import make_dinov2_model

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

    def forward(self, x1, x2):
        if x1 is None:
            y1 = None
        else:
            y1 = self.model_1(x1)

        if x2 is None:
            y2 = None
        else:
            y2 = self.model_1(x2)
        return y1, y2


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
    # 添加 dinov2 相关参数的默认值
    dinov2 = getattr(opt, 'dinov2', False)
    dinov2_size = getattr(opt, 'dinov2_size', 'vitb14')
    
    if opt.views == 2:
        model = two_view_net(
            opt.nclasses, 
            block=opt.block,
            return_f=opt.triplet_loss,
            resnet=opt.resnet,
            dinov2=dinov2,
            dinov2_size=dinov2_size
        )
    elif opt.views == 3:
        model = three_view_net(
            opt.nclasses, 
            share_weight=opt.share,
            block=opt.block,
            return_f=opt.triplet_loss, 
            resnet=opt.resnet,
            dinov2=dinov2,
            dinov2_size=dinov2_size
        )
    return model
