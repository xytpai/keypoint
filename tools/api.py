import os
import torch
import time
import random


def load_cfg(path_to_cfg):
    cfg_name = os.path.split(path_to_cfg)[1]
    cfg_name = cfg_name.split('.')[0]
    cfg = __import__('configs.'+cfg_name, fromlist=(cfg_name,)).cfg
    cfg.name = path_to_cfg
    return cfg


def parse_cfg(argv):
    cfg = None
    for option in argv:
        if option.startswith('cfg='):
            cfg = load_cfg(option.split('=')[1].strip())
        elif option.startswith('cfg.'):
            try: exec(option)
            except:
                options = option.split('=')
                option_ = options[0] + '="' + options[1] + '"'
                exec(option_)
    return cfg


def prepare_device(cfg):
    if cfg.mode == 'train':
        torch.cuda.set_device(cfg.train.devices[0])
        seed = cfg.train.seed
        if seed >= 0:
            random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    else:
        torch.cuda.set_device(eval('cfg.'+cfg.mode+'.device'))


def prepare_detector(cfg):
    dt = __import__('detectors.'+cfg.detector, fromlist=(cfg.detector,))
    detector = dt.Detector(cfg)
    if cfg.mode == 'train':
        if cfg.train.load_ckpt:
            filtered_ckpt = []
            for root, dirs, files in os.walk('./weights'):  
                for file in files:
                    if file.endswith('.ckpt') and file.startswith(cfg.name):
                        filtered_ckpt.append(file)
            if len(filtered_ckpt) == 0: latest_ckpt = None
            else: latest_ckpt = sorted(filtered_ckpt, reverse=True)[0]
            if latest_ckpt is not None:
                detector = torch.load(latest_ckpt)
        detector = torch.nn.DataParallel(detector, device_ids=cfg.train.devices)
        detector = detector.cuda(cfg.train.devices[0])
        detector.train()
    else: 
        detector.load_state_dict(torch.load(cfg.weight_file, map_location='cpu'))
        detector = detector.cuda(eval('cfg.'+cfg.mode+'.device'))
        detector.eval()
    return detector


def prepare_dataset(cfg):
    ds = __import__('datasets.'+cfg.dataset.name, fromlist=(cfg.dataset.name,))
    dataset = ds.Dataset(cfg)
    return dataset


def prepare_optimizer(cfg, detector):
    lr_base = cfg.train.lr_base
    params = []
    for key, value in detector.named_parameters():
        if not value.requires_grad:
            continue
        _lr = lr_base
        _weight_decay = cfg.train.weight_decay
        if "bias" in key:
            _lr = lr_base * 2
            _weight_decay = 0
        params += [{"params": [value], "lr": _lr, "weight_decay": _weight_decay}]
    opt = torch.optim.SGD(params, lr=_lr, momentum=cfg.train.momentum)
    return opt


class Trainer(object):
    def __init__(self, cfg, detector, dataset, opt):
        self.cfg = cfg
        self.detector = detector
        self.detector.train()
        self.dataset = dataset
        self.loader = dataset.make_loader()
        self.opt = opt
        self.step = self.detector.get('step', 0)
        self.epoch = self.detector.get('epoch', 0)
        self.lr_base = cfg.train.lr_base
        self.lr_gamma = cfg.train.lr_gamma
        self.lr_schedule = cfg.train.lr_schedule
        self.warmup_iters = cfg.train.warmup_iters
        self.warmup_factor = cfg.train.warmup_factor
        self.device = cfg.train.devices
        self.save = cfg.train.save
        
    def step_epoch(self, save_last=False):
        if self.epoch >= self.cfg.train.num_epoch: 
            if save_last:
                self.detector.module.step = self.step
                self.detector.module.epoch = self.epoch
                torch.save(self.detector.module.state_dict(), self.cfg.weight_file)
                torch.save(self.detector.module, 'weights/'+self.cfg.name+'_'+str(self.step)+'.ckpt')
            return True
        self.detector.train()
        self.detector.module.backbone.freeze_stages(int(self.cfg.train.freeze_stages))
        if self.cfg.train.backbone.freeze_bn: self.detector.module.backbone.freeze_bn()
        # loop        
        for i, data in enumerate(self.loader):
            # lr function
            lr = self.lr_base
            if self.step < self.warmup_iters:
                alpha = float(self.step) / self.warmup_iters
                warmup_factor = self.warmup_factor * (1.0 - alpha) + alpha
                lr = lr*warmup_factor 
            else:
                for j in range(len(self.lr_schedule)):
                    if self.step < self.lr_schedule[j]:
                        break
                    lr *= self.lr_gamma
            for param_group in self.opt.param_groups:
                param_group['lr'] = lr
            # #########
            if i == 0: batch_size = int(data['img'].shape[0])
            torch.cuda.synchronize()
            start = time.time()
            self.opt.zero_grad()          
            loss = eval(self.cfg.loss_def)
            loss.backward()
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.detector.parameters(), self.grad_clip)
            self.opt.step()
            maxmem = int(torch.cuda.max_memory_allocated(device=\
                self.device[0]) / 1024 / 1024)
            torch.cuda.synchronize()
            totaltime = int((time.time() - start) * 1000)
            print('total_step:%d: epoch:%d, step:%d/%d, loss:%f, maxMem:%dMB, time:%dms, lr:%f' % \
                (self.step, self.epoch, i*batch_size, len(self.dataset), loss, maxmem, totaltime, lr))
            self.step += 1
        self.epoch += 1
        if self.save:
                self.detector.module.step = self.step
                self.detector.module.epoch = self.epoch
                torch.save(self.detector.module.state_dict(), self.cfg.weight_file)
                torch.save(self.detector.module, 'weights/'+self.cfg.name+'_'+str(self.step)+'.ckpt')
        return False


class Inferencer(object):
    def __init__(self, cfg, detector, dataset):
        self.cfg = cfg
        self.detector = detector
        self.detector.eval()
        self.dataset = dataset
        self.normalizer = dataset.normalizer

    def pred(self, img_pil):
        with torch.no_grad():
            img, location = self.dataset.transform_inference_img(img_pil)
            img = img.cuda(eval('self.cfg.'+self.cfg.mode+'.device'))
            return eval(self.cfg.inf_def)
