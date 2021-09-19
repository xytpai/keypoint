import os, sys
sys.path.append(os.getcwd())
from api import *
from PIL import Image
print('loading cfg ...')
cfg = parse_cfg(sys.argv)
print(cfg)
demo_dir = cfg.demo.root

prepare_device(cfg)
detector = prepare_detector(cfg)
dataset = prepare_dataset(cfg)
inferencer = Inferencer(cfg, detector, dataset)

for filename in os.listdir(demo_dir):
    if filename.endswith('jpg'):
        if filename[:5] == 'pred_': 
            continue
        img = Image.open(os.path.join(demo_dir, filename))
        pred = inferencer.pred(img)
        name = demo_dir + '/pred_' + filename.split('.')[0]+'.jpg'
        dataset.show(img, pred, name)