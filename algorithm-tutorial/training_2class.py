import os
import time
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt

from wholeslidedata.interoperability.detectron2.iterator import WholeSlideDetectron2Iterator
from wholeslidedata.interoperability.detectron2.trainer import WholeSlideDectectron2Trainer
from wholeslidedata.interoperability.detectron2.predictor import Detectron2DetectionPredictor
from wholeslidedata.iterators import create_batch_iterator
from utils.vis import plot_boxes, FocalLoss

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.modeling import build_model

from utils.loss import MyFocalROIHeads 
user_config = {
    'wholeslidedata': {
        'default': {
            'yaml_source': "./configs/training_sample.yml",
            "seed": 42,
            "image_backend": "asap",
            'labels': {
                "ROI": 0,
                "lymphocytes": 1,
                "monocytes": 2
            },
        
            
            'batch_shape': {
                'batch_size': 256,
                'spacing': 0.5,
                'shape': [224,224,3],#[128,128,3],
                'y_shape': [1500, 6],
            },
            
            
            
            "annotation_parser": {
                "sample_label_names": ['roi'],
            },
            
            'point_sampler_name': "RandomPointSampler",
            'point_sampler': {
                "buffer": {'spacing': "${batch_shape.spacing}", 'value': -64},
            },
            
            'patch_label_sampler_name': 'DetectionPatchLabelSampler',
            'patch_label_sampler': {
                "max_number_objects": 1500,
                "detection_labels": ['lymphocytes','monocytes'],
                    
            },
            
        }
    }
}

training_batch_generator = create_batch_iterator(
    user_config=user_config,
    mode='training',
    cpus=4,
    iterator_class=WholeSlideDetectron2Iterator,
)

output_folder = Path('./outputs')
if not(os.path.isdir(output_folder)): os.mkdir (output_folder) 
cpus = 4

cfg = get_cfg()
# using faster rcnn architecture
cfg.merge_from_file(
    model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
)


cfg.DATASETS.TRAIN = ("detection_dataset2",)
cfg.DATASETS.TEST = () 
cfg.TEST.EVAL_PERIOD = 20  # 200イテレーション毎に評価を実施（例）
cfg.DATALOADER.NUM_WORKERS = 4

cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[16, 24, 32]]
cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[1.0]]
cfg.MODEL.ROI_HEADS.NAME = "MyFocalROIHeads"

cfg.SOLVER.IMS_PER_BATCH = 256
cfg.SOLVER.BASE_LR = 0.002  # pick a good LR
cfg.SOLVER.MAX_ITER = 1000  # 2000 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
cfg.SOLVER.STEPS = (500, 750)
cfg.SOLVER.WARMUP_ITERS = 100
cfg.SOLVER.WARMUP_FACTOR = 1.0/1000 
cfg.SOLVER.GAMMA = 0.5
cfg.SOLVER.CHECKPOINT_PERIOD = 200


cfg.INPUT.RANDOM_FLIP = "horizontal"  # "horizontal", "vertical", "none"など
cfg.INPUT.BRIGHTNESS = 0.8, 1.2
cfg.INPUT.CONTRAST = 0.8, 1.2
cfg.INPUT.SATURATION = 0.8, 1.2
cfg.INPUT.HUE = 0.8, 1.2


cfg.OUTPUT_DIR = str(output_folder)
output_folder.mkdir(parents=True, exist_ok=True)


model = build_model(cfg)
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Parameter Count:\n" + str(pytorch_total_params))

trainer = WholeSlideDectectron2Trainer(cfg, user_config=user_config, cpus=cpus)


trainer.resume_or_load(resume=False)
trainer.train()