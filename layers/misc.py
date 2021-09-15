import torch
import torch.nn as nn 
import torch.nn.functional as F


def get_group_gn(dim, dim_per_gp, num_groups):
    """get number of groups used by GroupNorm, based on number of channels."""
    assert dim_per_gp == -1 or num_groups == -1, \
        "GroupNorm: can only specify G or C/G."
    if dim_per_gp > 0:
        assert dim % dim_per_gp == 0, \
            "dim: {}, dim_per_gp: {}".format(dim, dim_per_gp)
        group_gn = dim // dim_per_gp
    else:
        assert dim % num_groups == 0, \
            "dim: {}, num_groups: {}".format(dim, num_groups)
        group_gn = num_groups
    return group_gn


def group_norm(out_channels, affine=True, divisor=1):
    out_channels = out_channels // divisor
    dim_per_gp = -1 // divisor
    num_groups = 32 // divisor
    eps = 1e-5 # default: 1e-5
    return nn.GroupNorm(
        get_group_gn(out_channels, dim_per_gp, num_groups), 
        out_channels, 
        eps, 
        affine)


def make_conv3x3(
    in_channels, 
    out_channels, 
    dilation=1, 
    stride=1, 
    use_gn=False,
    use_relu=False,
    kaiming_init=True
):
    conv = nn.Conv2d(
        in_channels, 
        out_channels, 
        kernel_size=3, 
        stride=stride, 
        padding=dilation, 
        dilation=dilation, 
        bias=False if use_gn else True
    )
    if kaiming_init:
        nn.init.kaiming_normal_(
            conv.weight, mode="fan_out", nonlinearity="relu"
        )
    else:
        torch.nn.init.normal_(conv.weight, std=0.01)
    if not use_gn:
        nn.init.constant_(conv.bias, 0)
    module = [conv,]
    if use_gn:
        module.append(group_norm(out_channels))
    if use_relu:
        module.append(nn.ReLU(inplace=True))
    if len(module) > 1:
        return nn.Sequential(*module)
    return conv


def make_fc(dim_in, hidden_dim, use_gn=False):
    '''
    Caffe2 implementation uses XavierFill, which in fact
    corresponds to kaiming_uniform_ in PyTorch
    '''
    if use_gn:
        fc = nn.Linear(dim_in, hidden_dim, bias=False)
        nn.init.kaiming_uniform_(fc.weight, a=1)
        return nn.Sequential(fc, group_norm(hidden_dim))
    fc = nn.Linear(dim_in, hidden_dim)
    nn.init.kaiming_uniform_(fc.weight, a=1)
    nn.init.constant_(fc.bias, 0)
    return fc


def conv_with_kaiming_uniform(use_gn=False, use_relu=False):
    def make_conv(
        in_channels, out_channels, kernel_size, stride=1, dilation=1
    ):
        conv = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=dilation * (kernel_size - 1) // 2, 
            dilation=dilation, 
            bias=False if use_gn else True
        )
        # Caffe2 implementation uses XavierFill, which in fact
        # corresponds to kaiming_uniform_ in PyTorch
        nn.init.kaiming_uniform_(conv.weight, a=1)
        if not use_gn:
            nn.init.constant_(conv.bias, 0)
        module = [conv,]
        if use_gn:
            module.append(group_norm(out_channels))
        if use_relu:
            module.append(nn.ReLU(inplace=True))
        if len(module) > 1:
            return nn.Sequential(*module)
        return conv
    return make_conv


def box_iou(box1, box2):
    # box1: F(n, 4) # 4: ymin, xmin, ymax, xmax
    # box2: F(m, 4)
    # ->    F(n, m)
    tl = torch.max(box1[:,None,:2], box2[:,:2])  # [n,m,2]
    br = torch.min(box1[:,None,2:], box2[:,2:])  # [n,m,2]
    hw = (br-tl+1).clamp(min=0)  # [n,m,2]
    inter = hw[:,:,0] * hw[:,:,1]  # [n,m]
    area1 = (box1[:,2]-box1[:,0]+1) * (box1[:,3]-box1[:,1]+1)  # [n,]
    area2 = (box2[:,2]-box2[:,0]+1) * (box2[:,3]-box2[:,1]+1)  # [m,]
    iou = (inter+1.0) / (area1[:,None] + area2 - inter+1.0)
    return iou


def to_onehot(target, num_class):
    # target: L(n)
    # num_class: int
    one_hot = torch.zeros(target.shape[0], num_class, 
        device=target.device).scatter_(1, target.view(-1,1), 1)
    return one_hot


def torch_select(tensor_list, m):
    out = []
    for x in tensor_list:
        out.append(x[m])
    return tuple(out)


def torch_cat(tensor_list_list, dim):
    out = []
    for x in tensor_list_list:
        out.append(torch.cat(x, dim=dim))
    return tuple(out)


def transfer_box_(
    pred_reg, 
    valid_ymin, valid_xmin, valid_ymax, valid_xmax, 
    ori_h, ori_w):
    pred_reg[..., 0::2] -= valid_ymin
    pred_reg[..., 1::2] -= valid_xmin
    pred_reg[..., 0::2] *= ori_h / (valid_ymax - valid_ymin + 1)
    pred_reg[..., 1::2] *= ori_w / (valid_xmax - valid_xmin + 1)
    return pred_reg


def bilinear_interpolate(x, size, align=True):
    return F.interpolate(x, size=size, \
            mode='bilinear', align_corners=align)