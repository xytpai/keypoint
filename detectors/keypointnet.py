import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from layers import *
from .backbones import *
from .necks import *
from .heads import *


class Detector(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = ResNet(depth=cfg.model.backbone.depth)
        self.neck = FusionFPN(self.backbone.out_channels, cfg.model.head.channels)
        self.head = ConvHead4(cfg.model.head.channels, cfg.model.num_keypoints)
        if cfg.mode == 'train' and cfg.train.load_pretrained_backbone:
            self.backbone.load_pretrained_params()
        
    def forward(self, imgs, kepoints=None, heatmap=None):
        # imgs: F(b, 3, im_h, im_w)
        # kepoints: F(b, n, nk, 3)
        # heatmap: F(b, nk, im_h, im_w)
        batch_size, _, im_h, im_w = imgs.shape
        out_hm = self.head(self.neck(self.backbone(imgs))).sigmoid()
        _, _, oh, ow = out_hm.shape
        if heatmap is not None:
            heatmap = bilinear_interpolate_as(heatmap, out_hm)
            loss_hm = center_focal_loss(out_hm, heatmap)
            return loss_hm
        else:
            out_hm = peak_nms(out_hm) # F(b, nk, oh, ow)
            hm_score, hm_class = out_hm.max(dim=1) # F(b, oh, ow), L(b, oh, ow)
            batch_indexs = torch.arange(batch_size, 
                device=hm_score.device).view(-1, 1, 1).expand_as(hm_score)
            mesh = aligned_mesh2d(im_h, im_w, oh, ow, batch_size, hm_score.device)
            hm_mask = hm_score > eval('self.cfg.'+self.cfg.mode+'.threshold.heatmap')
            hm_class_selected = hm_class[hm_mask] # L(n)
            hm_score_selected = hm_score[hm_mask] # F(n)
            center_selected = mesh[hm_mask] # F(n, 2)
            batch_indexs_selected = batch_indexs[hm_mask] # L(n)
            return {
                'heatmap': bilinear_interpolate_as(hm_score.unsqueeze(1), imgs)[:, 0],
                'class': hm_class_selected,
                'score': hm_score_selected,
                'center': center_selected,
                'index': batch_indexs_selected}
    
    def load_pretrained_params(self, path):
        self.load_state_dict(torch.load(path, map_location='cpu'))
