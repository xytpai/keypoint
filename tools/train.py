import os, sys
sys.path.append(os.getcwd())
from api import *
print('loading cfg ...')
cfg = parse_cfg(sys.argv)
print(cfg)

prepare_device(cfg)
detector = prepare_detector(cfg)
dataset = prepare_dataset(cfg)
opt = prepare_optimizer(cfg, detector)
trainer = Trainer(cfg, detector, dataset, opt)

while True:
    if trainer.step_epoch():
        break
print('Schedule finished!')