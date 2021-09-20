import numpy as np
from PIL import Image, ImageDraw
import torch
import warnings
import torchvision.transforms as transforms
import random
warnings.filterwarnings("ignore")


def filter_annotation(anno, height, width, hw_th=1, area_th=1):
    anno = [obj for obj in anno if not obj.get('ignore', False)]
    anno = [obj for obj in anno if obj['iscrowd'] == 0] # filter crowd annotations
    anno = [obj for obj in anno if obj['area'] >= area_th]
    anno = [obj for obj in anno if all(o >= hw_th for o in obj['bbox'][2:])]
    _anno = []
    for obj in anno:
        xmin, ymin, w, h = obj['bbox']
        inter_w = max(0, min(xmin + w, width) - max(xmin, 0))
        inter_h = max(0, min(ymin + h, height) - max(ymin, 0))
        if inter_w * inter_h > 0: _anno.append(obj)
    return _anno


def x_flip(img, bbox=None, keypoints=None):
    img = img.transpose(Image.FLIP_LEFT_RIGHT) 
    w = img.width
    if bbox is not None and bbox.shape[0] != 0:
        xmin = w - bbox[:, 3] - 1
        xmax = w - bbox[:, 1] - 1
        bbox[:, 1] = xmin
        bbox[:, 3] = xmax
    if keypoints is not None and keypoints.shape[0] != 0:
        keypoints[:, :, 1] = w - keypoints[:, :, 1] - 1
    return img, bbox, keypoints


def resize_img(img, size, bbox=None, keypoints=None):
    w, h = img.size
    img = img.resize((size, size), Image.BILINEAR)
    scale_h = float(size) / h
    scale_w = float(size) / w
    if bbox is not None and bbox.shape[0] != 0:
        bbox[:, 0] *= scale_h
        bbox[:, 2] *= scale_h
        bbox[:, 1] *= scale_w
        bbox[:, 3] *= scale_w
        bbox[:, :2].clamp_(min=0)
        bbox[:, 2:].clamp_(max=size-2)
        ymin_xmin, ymax_xmax = bbox.split([2, 2], dim=1)
        h_w = ymax_xmax - ymin_xmin + 1
        m = h_w.min(dim=1)[0] <= 1
        ymax_xmax[m] = ymin_xmin[m] + 1
        bbox = torch.cat([ymin_xmin, ymax_xmax], dim=1)
    if keypoints is not None and keypoints.shape[0] != 0:
        keypoints[:, :, 0] *= scale_h
        keypoints[:, :, 1] *= scale_w
        keypoints[:, :, :2].clamp_(min=0)
        keypoints[:, :, :2].clamp_(max=size-1)
    return img, bbox, keypoints


COLOR_TABLE = [
    (256,0,0), (0,256,0), (0,0,256), 
    (255,0,255), (255,106,106),(139,58,58),(205,51,51),
    (139,0,139),(139,0,0),(144,238,144),(0,139,139)
] * 100


def draw_bbox_keypoint(drawObj, bbox, keypoint, color, bd=1):
    if bbox is not None:
        ymin, xmin, ymax, xmax = bbox
        drawObj.rectangle((xmin, ymin, xmax, ymin+bd), fill=color)
        drawObj.rectangle((xmin, ymax-bd, xmax, ymax), fill=color)
        drawObj.rectangle((xmin, ymin, xmin+bd, ymax), fill=color)
        drawObj.rectangle((xmax-bd, ymin, xmax, ymax), fill=color)
    if keypoint is not None:
        for i in range(len(keypoint)):
            if int(keypoint[i][2]) == 2:
                y, x = int(keypoint[i][0]), int(keypoint[i][1])
                drawObj.ellipse((x-bd-1, y-bd-1, x+bd+1, y+bd+1), fill=COLOR_TABLE[i])


def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size
    a1  = 1
    b1  = (height + width)
    c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1  = (b1 + sq1) / 2
    a2  = 4
    b2  = 2 * (height + width)
    c2  = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2  = (b2 + sq2) / 2
    a3  = 4 * min_overlap
    b3  = -2 * min_overlap * (height + width)
    c3  = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3  = (b3 + sq3) / 2
    return min(r1, r2, r3)


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_umich_gaussian(heatmap, center, radius, k=1):
    # ref:https://github.com/xingyizhou/CenterNet/blob/master/src/lib/utils/image.py
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
    y, x = int(center[0]), int(center[1])
    height, width = heatmap.shape[0:2]
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)
    masked_heatmap  = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0: # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap
