import os, sys
sys.path.append(os.getcwd())
from .api import *
from PIL import Image
for option in sys.argv:
    if option.startswith('cfg='):
        cfg = load_cfg(option.split('=')[1].strip())
    elif option.startswith('cfg.'): exec(option)
demo_dir = cfg.demo.root

prepare_device(cfg)
detector = prepare_detector(cfg)
dataset = prepare_dataset(cfg, detector)
inferencer = Inferencer(cfg, detector, dataset)

for filename in os.listdir(demo_dir):
    if filename.endswith('jpg'):
        if filename[:5] == 'pred_': 
            continue
        img = Image.open(os.path.join(demo_dir, filename))
        pred = inferencer.pred(img)
        name = demo_dir + '/pred_' + filename.split('.')[0]+'.jpg'
        dataset.show(img, pred, name)