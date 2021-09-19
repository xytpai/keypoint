import os, sys
sys.path.append(os.getcwd())
from .api import *
for option in sys.argv:
    if option.startswith('cfg='):
        cfg = load_cfg(option.split('=')[1].strip())
    elif option.startswith('cfg.'): exec(option)

prepare_device(cfg)
detector = prepare_detector(cfg)
dataset = prepare_dataset(cfg, detector)
opt = prepare_optimizer(cfg, detector)
trainer = Trainer(cfg, detector, dataset, opt)

while True:
    if trainer.step_epoch():
        break
print('Schedule finished!')