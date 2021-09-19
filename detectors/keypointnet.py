import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from layers import *
from detectors.backbones import *
from detectors.necks import *
from detectors.heads import *


class Detector(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.register_buffer('trained_log', torch.zeros(2).long())
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
        _, n, nk, _ = kepoints.shape
        x = self.neck(self.backbone(imgs))
        out_hm = self.head_hm(x) # F(b, nk, oh, ow)
        out_ke = self.head_ke(x) # F(b, f, oh, ow)
        if kepoints is not None and heatmap is not None:
            heatmap = bilinear_interpolate_as(heatmap, out_hm)
            loss_hm = center_focal_loss(out_hm, heatmap)
            kepoints_y = kepoints[:, :, :, 0] / float(im_h) * 2 - 1
            kepoints_x = kepoints[:, :, :, 1] / float(im_w) * 2 - 1
            grid = torch.stack([kepoints_y, kepoints_x], dim=-1)
            out_ke = F.grid_sample(out_ke, grid) # F(b, f, n, nk)
            m_ke = out_ke.mean(dim=3) # F(b, f, n)
            loss_ke_same = ((out_ke - m_ke.unsqueeze(-1))**2).mean()
            corr_m_ke = (m_ke.unsqueeze(-1) - m_ke.unsqueeze(-2))**2 # F(b, f, n, n)
            sigma = 0.1
            ec_m_ke = (-1/2.0/(sigma**2)*corr_m_ke).exp() # F(b, f, n, n)
            loss_ke_diff = ec_m_ke.triu(diagonal=1).mean()
            loss_ke = loss_ke_same + loss_ke_diff
            return 0.001 * loss_ke + 0.999 * loss_hm
        else:
            out_hm = peak_nms(out_hm.sigmoid()) # F(b, nk, oh, ow)
            hm_score, hm_class = out_hm.max(dim=1) # F(b, oh, ow), L(b, oh, ow)
            batch_idxs = torch.arange(batch_size, 
                device=hm_score.device).view(-1, 1, 1).expand_as(hm_score)
            hm_mask = hm_score > self.cfg.eval.threshold.heatmap
            hm_class = hm_class[hm_mask] # L(n)
            hm_score = hm_score[hm_mask] # F(n)
            batch_idxs = batch_idxs[hm_mask] # L(n)
            _, oh, ow = hm_score.shape
            out_ke = out_ke.permute(0, 2, 3, 1).contiguous()
            ft = out_ke[hm_mask] # F(n, f)
            similarity = (ft.unsqueeze(1) - ft.unsqueeze(0))**2
            similarity = similarity.mean(dim=2) # F(n,n)
            return {
                'class': hm_class,
                'score': hm_score,
                'similarity': similarity}
    
    def load_pretrained_params(self, path):
        self.load_state_dict(torch.load(path, map_location='cpu'))
