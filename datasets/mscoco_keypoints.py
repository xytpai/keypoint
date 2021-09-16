import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image, ImageDraw
if __name__ != '__main__':
    from datasets.utils import *
else:
    from utils import *


class Dataset(torchvision.datasets.coco.CocoDetection):
    def __init__(self, cfg):
        self.cfg = cfg
        self.data_num_keypoints = cfg.data.num_keypoints
        self.normalizer = transforms.Normalize(*cfg.data.norm)
        super(Dataset, self).__init__(cfg.dataset.train_root, cfg.dataset.train_json)
        # filter self.ids
        ids = []
        for img_id in self.ids:
            img_info = self.coco.loadImgs(img_id)[0]
            height, width = img_info['height'], img_info['width']
            if min(height, width) < 32: continue
            ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
            anno = self.coco.loadAnns(ann_ids)
            if len(filter_annotation(anno, height, width))>0: ids.append(img_id)
        self.ids = ids

    def __getitem__(self, idx):
        # return
        # img: F(3, h, w)
        # bbox: F(n, 4) ymin,xmin,ymax,xmax
        # keypoint: F(n, num_keypoints, 3) y,x,flag; flag:0-no-ann/1-not-visiable/2-ok
        img, anno = super().__getitem__(idx)
        anno = filter_annotation(anno, img.size[1], img.size[0])
        bbox = [obj['bbox'] for obj in anno]
        bbox = torch.as_tensor(bbox).float().reshape(-1, 4)  # guard against no boxes
        xmin_ymin, w_h = bbox.split([2, 2], dim=1)
        xmax_ymax = xmin_ymin + w_h - 1
        xmin, ymin = xmin_ymin.split([1, 1], dim=1)
        xmax, ymax = xmax_ymax.split([1, 1], dim=1)
        bbox = torch.cat([ymin, xmin, ymax, xmax], dim=1)
        keypoints = [obj['keypoints'] for obj in anno]
        keypoints = torch.as_tensor(keypoints).float().reshape(-1, self.data_num_keypoints, 3)
        keypoints = torch.stack([keypoints[:, :, 1], keypoints[:, :, 0], keypoints[:, :, 2]], dim=-1)
        # transform
        if random.random() < 0.5: img, bbox, keypoints = x_flip(img, bbox, keypoints)
        img, bbox, keypoints = resize_img(img, self.cfg.data.size, bbox, keypoints)
        img = transforms.ToTensor()(img)
        if self.cfg.data.norm_en: img = self.normalizer(img)
        # heatmap
        _, h, w = img.shape
        heatmap = np.zeros((self.data_num_keypoints, h, w))
        for i in range(keypoints.shape[0]):
            ymin, xmin, ymax, xmax = bbox[i].tolist()
            r = gaussian_radius((ymax-ymin, xmax-xmin))
            for j in range(self.data_num_keypoints):
                y, x, flag = keypoints[i, j].tolist()
                if int(flag) == 2: heatmap[j] = \
                    draw_umich_gaussian(heatmap[j], (round(y), round(x)), round(r))
        return img, bbox, keypoints, torch.from_numpy(heatmap)


    def collate_fn(self, data):
        img, bbox, keypoints, heatmap = zip(*data)
        batch_num = len(img)
        img = torch.stack(img)
        heatmap = torch.stack(heatmap)
        max_n = 0
        for b in range(batch_num):
            if bbox[b].shape[0] > max_n: max_n = bbox[b].shape[0]
        bbox_t = torch.zeros(batch_num, max_n, 4).float()
        keypoints_t = torch.zeros(batch_num, max_n, self.data_num_keypoints, 3).float()
        for b in range(batch_num):
            bbox_t[b, :bbox[b].shape[0]] = bbox[b]
            keypoints_t[b, :keypoints[b].shape[0]] = keypoints[b]
        return {'img':img, 'bbox':bbox_t, 'keypoints':keypoints_t, 'heatmap':heatmap}

    
    def transform_inference_img(self, img_pil):
        if img_pil.mode != 'RGB': img_pil = img_pil.convert('RGB')
        img_pil, _, _ = resize_img(img_pil, self.cfg.data.size)
        img = transforms.ToTensor()(img_pil)
        if self.cfg.data.norm_en: img = self.normalizer(img)
        img = img.unsqueeze(0)
        return img
    
    def make_loader(self):
        batch_size = self.cfg.train.batch_size
        return data.DataLoader(self, batch_size=batch_size, shuffle=True, 
            num_workers=self.cfg.train.num_workers, collate_fn=self.collate_fn)
    
    def show(self, img, pred, file_name=None):
        if not isinstance(img, Image.Image):
            img = transforms.ToPILImage()(img)
        bbox, keypoints = pred.get('bbox', None), pred.get('keypoints', None)
         # sort
        hw = bbox[:, 2:] - bbox[:, :2]
        area = hw[:, 0] * hw[:, 1] # N
        select = area.sort(descending=True)[1] # L(n)
        # draw
        drawObj = ImageDraw.Draw(img)
        for i in range(select.shape[0]):
            i = int(select[i])
            box = bbox[i]
            keypoint = keypoints[i] if keypoints is not None else None
            draw_bbox_keypoint(drawObj, box[0], box[1], box[2], box[3], keypoint, color=COLOR_TABLE[i])
        if file_name is not None: img.save(file_name)
        else: img.show()


if __name__ == '__main__':
    from addict import Dict

    cfg = Dict()
    cfg.task = "person_keypoints"
    cfg.loss_def = "self.detector(data['img'], data['bbox'], data['labels'], data['keypoints']).mean()"
    cfg.data.num_keypoints = 17
    cfg.data.norm = [(0.485,0.456,0.406), (0.229,0.224,0.225)]
    cfg.data.norm_en = False
    cfg.data.size = 513
    cfg.dataset.train_root = '/home/xytpai/dataset/mscoco/val2017'
    cfg.dataset.train_json = '/home/xytpai/dataset/mscoco/person_keypoints_val2017.json'
    cfg.train.batch_size=2
    cfg.train.num_workers=0

    dataset = Dataset(cfg)
    loader = dataset.make_loader()
    for data in loader:
        img, bbox, keypoints, heatmap = data['img'], data['bbox'], data['keypoints'], data['heatmap']
        print('img:', img.shape)
        print('bbox:', bbox.shape)
        print('keypoints:', keypoints.shape)
        print('heatmap:', heatmap.shape)
        b = random.randint(0, cfg.train.batch_size-1)
        hm, _ = torch.max(heatmap[b], dim=0)
        # hm = heatmap[b][0]
        hm[hm!=1] = 0
        hm = hm.view(1,513,513).expand(3,513,513)
        dataset.show(img[b], {'bbox':bbox[b], 'keypoints':keypoints[b]})
        dataset.show(hm, {'bbox':bbox[b]})
        break
