from addict import Dict
cfg = Dict()

cfg.mode = 'train'
cfg.detector = 'keypointnet'
cfg.weight_file = 'weights/keypointnet.pth'
cfg.loss_def = "self.detector(data['img'], data['keypoints'], data['heatmap']).mean()"
cfg.inf_def = "self.detector(img)"
cfg.demo.root = 'images'

# data
cfg.data.norm = [(0.485,0.456,0.406), (0.229,0.224,0.225)]
cfg.data.num_keypoints = 17
cfg.data.norm_en = True
cfg.data.size = 513

# dataset
cfg.dataset.name = 'mscoco_keypoints'
cfg.dataset.train_root = 'images'
cfg.dataset.train_json = 'images/person_keypoints_mini.json'

# model
cfg.model.backbone.depth = 50
cfg.model.head.channels = 128
cfg.model.num_keypoints = 17 # same as cfg.data.num_keypoints

# train
cfg.train.batch_size = 1
cfg.train.num_workers = 0
cfg.train.backbone.pretrained = True
cfg.train.backbone.freeze_bn = True
cfg.train.backbone.freeze_stages = 1
cfg.train.devices = [0]
cfg.train.seed = 0
cfg.train.load_ckpt = True
cfg.train.lr_base = 0.01
cfg.train.lr_gamma = 0.1
cfg.train.weight_decay = 0.0001
cfg.train.momentum = 0.9
cfg.train.lr_schedule = [60000, 80000]
cfg.train.num_epoch = 12
cfg.train.warmup_iters = 500
cfg.train.warmup_factor = 1.0/3.0
cfg.train.save = True

# eval
cfg.eval.device = 0
cfg.eval.threshold.heatmap = 0.5

# test
cfg.test.device = 0
cfg.test.threshold.heatmap = 0.5