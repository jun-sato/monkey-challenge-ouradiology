# ouradiology team approach of the MONKEY challenge: 
This repository contains all tutorials and code in connection with the MONKEY challenge run on [Grand Challenge](https://monkey.grand-challenge.org/)  
We greatly appreciate [Baseline code](https://github.com/computationalpathologygroup/monkey-challenge).

# Training Overview

This repository demonstrates how to train a Faster R-CNN model for lymphocyte detection in whole slide images (WSIs) using [WholeSlideData](https://github.com/ComputationalPathologyGroup/wholeslidedata) and [Detectron2](https://github.com/facebookresearch/detectron2). It includes a pipeline setup for creating patches from WSIs, preparing annotations, and training a customized Faster R-CNN model.

## What is the difference between our approach and baseline?
image size: [96, 96] → [224, 224]  
long training epoch 200 → 2000  
large learning rate 200 → 2000  
ensemble: none → wbf


## Main Libraries & Tools

- **WholeSlideData**  
  Handles patch sampling, annotation parsing, and data iteration for WSIs.
- **Detectron2**  
  A flexible object detection library. Here we use a Faster R-CNN model from the [model_zoo](https://github.com/facebookresearch/detectron2/tree/main/configs).

---

## Configuration Details

### WholeSlideData (`user_config`)

```yaml
wholeslidedata:
  default:
    yaml_source: "./configs/training_sample.yml"
    image_backend: "asap"
    labels:
      ROI: 0
      lymphocytes: 1

    batch_shape:
      batch_size: 10
      spacing: 0.5
      shape: [256, 256, 3]
      y_shape: [1000, 6]

    annotation_parser:
      sample_label_names: ['roi']

    point_sampler_name: "RandomPointSampler"
    point_sampler:
      buffer:
        spacing: ${batch_shape.spacing}
        value: -64

    patch_label_sampler_name: "DetectionPatchLabelSampler"
    patch_label_sampler:
      max_number_objects: 1000
      detection_labels: ['lymphocytes']
```

- **`batch_shape.batch_size`**: Number of patches per batch (10)  
- **`batch_shape.shape`**: Size of each patch (256×256×3)  
- **`point_sampler_name`**: Randomly samples patch locations  
- **`patch_label_sampler.max_number_objects`**: Maximum objects per patch (1000)

### Detectron2 Config

```python
cfg.merge_from_file(
    model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
)
cfg.DATASETS.TRAIN = ("detection_dataset2",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 1

cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[16, 24, 32]]

cfg.SOLVER.IMS_PER_BATCH = 10
cfg.SOLVER.BASE_LR = 0.001
cfg.SOLVER.MAX_ITER = 2000
cfg.SOLVER.STEPS = (10, 100, 250)
cfg.SOLVER.WARMUP_ITERS = 0
cfg.SOLVER.GAMMA = 0.5
```

- **Model**: `faster_rcnn_X_101_32x8d_FPN_3x`  
- **ROI_HEADS.NUM_CLASSES**: Number of classes (1; for lymphocytes)  
- **SOLVER.IMS_PER_BATCH**: Mini-batch size (10)  
- **SOLVER.BASE_LR**: Initial learning rate (0.001)  
- **SOLVER.MAX_ITER**: Maximum training iterations (2000)  
- **SOLVER.STEPS**: Iteration milestones for LR decay (10, 100, 250)  
- **ANCHOR_GENERATOR.SIZES**: Anchor sizes ([16, 24, 32])

---

## Training Process

1. **Batch Iterator Creation**  
   The `create_batch_iterator` function from WholeSlideData generates patch data according to the `user_config`.

2. **Model Configuration**  
   The Detectron2 config (`cfg`) loads the Faster R-CNN architecture from the model zoo.

3. **Trainer Execution**  
   A `WholeSlideDectectron2Trainer` uses both the `cfg` and `user_config` to start the training process.

4. **Output**  
   - Logs, model checkpoints, and final model weights are saved in the specified `OUTPUT_DIR` (default: `./outputs`).

---

## Model Parameter Count

```python
model = build_model(cfg)
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Parameter Count:\n" + str(pytorch_total_params))
```

- This snippet displays the total number of trainable parameters in the Faster R-CNN model.

---

## How to Run

```bash
git clone <this_repository>
cd <this_repository>
python training.py
```

- Training outputs (model weights, logs) are saved in `./outputs` by default.

---

## License

This repository is released under the [MIT License](LICENSE). Please see the [LICENSE](LICENSE) file for details.



## Creating the inference docker image
The folder `docker` contains the code to create the inference docker image that can be submitted to the MONKEY challenge.
`test_run.sh` lets you test the docker image locally. After that you can save it with `save.sh` and submit it to the MONKEY challenge.
If you want to see the log files of your submission, you have to submit to the Debugging Phase of the challenge.

## Results evaluation
The folder `evaluation` contains the code to evaluate the results of the MONKEY challenge. The exact same script is used for the leaderboard evaluation computation.

### How to use
Use the `get_froc_vals()`

1. Put the ground truth json files in the folder `evaluation/ground_truth/` with the file name format `case-id_inflammatory-cells.json`,
`case-id_lympocytes.json` and `case-id_monocytes.json` for the respective cell types. These files are provided along with the
`xml` files for the ground truth annotations ([how to access the data](https://monkey.grand-challenge.org/dataset-details/)).

2. Put the output of your algorithm in the folder `evaluation/test/` in a separate folder for each case with a subfolder `output` i.e. `case-id/output/` 
as folder names. In each of these folders, put the json files with the detection output and the name format 
`detected-inflammatory-cells.json`, `detected-lympocytes.json` and `detected-monocytes.json` for the respective cell types.
Additionally, you will need to provide the json file `evaluation/test/output/predictions.json`, which helps to distribute
the jobs.

3. Run `evaluation.py`. The script will compute the evaluation metrics for each case as well as overall and save them to 
`evaluation/test/output/metrics.json`.

The examples provided are the three files that are used for the evaluation phase of the debugging phase.

```angular2html
.
├── ground_truth/
│   ├── A_P000001_inflammatory-cells.json
│   ├── A_P000001_lymphocytes.json
│   ├── A_P000001_monocytes.json
│   └── (...)
└── test/
    ├── input/
    │   ├── A_P000001/
    │   │   └── output/
    │   │       ├── detected-inflammatory-cells.json
    │   │       ├── detected-lymphocytes.json
    │   │       └── detected-monocytes.json
    │   ├── A_P000002/
    │   │   └── (...)
    │   └── (...)
    ├── predictions.json
    └── output/
        └── metrics.json
```

## Other useful scripts
The folder `utils` contains other useful functions
- `json_to_xml.py`: Script that converts the output json files from grand-challenge back to xml files compatible with ASAP
There is also an optional `prob_cutoff` argument, that lets you filter out annotations with a threshold.
Helpful for visualising your results in ASAP.
- `plot_froc.py`: Script that plots the FROC curve from the metrics.json file generated by the evaluation script 
(this is also available for download on grand-challenge for the validation set cases).


