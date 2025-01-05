"""
It is meant to run within a container.

To run it locally, you can call the following bash script:

  ./test_run.sh

This will start the inference and reads from ./test/input and outputs to ./test/output

To save the container and prep it for upload to Grand-Challenge.org you can call:

  ./save.sh

Any container that shows the same behavior will do, this is purely an example of how one COULD do it.

Happy programming!
"""

from pathlib import Path
from glob import glob
import os
import json
from tqdm import tqdm
import numpy as np

from wholeslidedata.image.wholeslideimage import WholeSlideImage
from wholeslidedata.iterators import create_patch_iterator, PatchConfiguration
from wholeslidedata.annotation.labels import Label

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
RESOURCE_PATH = Path("resources")
Model_PATH = Path("/opt/ml/model")

from wsdetectron2 import Detectron2DetectionPredictor, apply_wbf_from_arrays
from structures import Point


def run():
    # Read the input

    image_paths = glob(os.path.join(INPUT_PATH, "images/kidney-transplant-biopsy-wsi-pas/*.tif"))
    mask_paths = glob(os.path.join(INPUT_PATH, "images/tissue-mask/*.tif"))

    image_path = image_paths[0]
    mask_path = mask_paths[0]

    output_path = OUTPUT_PATH
    json_filename_lymphocytes = "detected-lymphocytes.json"
    weight_root = os.path.join(Model_PATH, "model_final.pth")

    # Process the inputs: any way you'd like
    _show_torch_cuda_info()

    patch_shape = (224, 224, 3)
    spacings = (0.25,)
    overlap = (112, 112)
    offset = (0, 0)
    center = False

    patch_configuration = PatchConfiguration(patch_shape=patch_shape,
                                             spacings=spacings,
                                             overlap=overlap,
                                             offset=offset,
                                             center=center)

    model = Detectron2DetectionPredictor(
        output_dir=output_path,
        threshold=0.2,
        nms_threshold=1,
        weight_root=weight_root
    )

    iterator = create_patch_iterator(image_path=image_path,
                                     mask_path=mask_path,
                                     patch_configuration=patch_configuration,
                                     cpus=4,
                                     backend='asap')

    # Save your output
    inference(
        iterator=iterator,
        predictor=model,
        spacing=spacings[0],
        image_path=image_path,
        output_path=output_path,
        json_filename=json_filename_lymphocytes,
    )

    iterator.stop()

    location_detected_lymphocytes_all = glob(os.path.join(OUTPUT_PATH, "*.json"))
    location_detected_lymphocytes = location_detected_lymphocytes_all[0]
    print(location_detected_lymphocytes_all)
    print(location_detected_lymphocytes)
    # Secondly, read the results
    result_detected_lymphocytes = load_json_file(
        location=location_detected_lymphocytes,
    )

    return 0


def px_to_mm(px: int, spacing: float):
    return px * spacing / 1000


def to_wsd(points):
    """Convert list of coordinates into WSD points"""
    new_points = []
    for i, point in enumerate(points):
        p = Point(
            index=i,
            label=Label("lymphocyte", 1, color="blue"),
            coordinates=[point],
        )
        new_points.append(p)
    return new_points


def inference(iterator, predictor, spacing, image_path, output_path, json_filename):
    print("predicting...")
    output_dict = {
        "name": "lymphocytes",
        "type": "Multiple points",
        "version": {"major": 1, "minor": 0},
        "points": [],
    }

    output_dict_monocytes = {
        "name": "monocytes",
        "type": "Multiple points",
        "version": {"major": 1, "minor": 0},
        "points": [],
    }

    output_dict_inflammatory_cells = {
        "name": "inflammatory-cells",
        "type": "Multiple points",
        "version": {"major": 1, "minor": 0},
        "points": [],
    }
    class_thresholds={"lymphocyte": 0.4, "monocyte": 0.15}
    annotations = []
    counter = 0

    spacing_min = 0.25
    ratio = spacing / spacing_min
    with WholeSlideImage(image_path) as wsi:
        spacing = wsi.get_real_spacing(spacing_min)
        x_size, y_size = wsi.get_shape_from_spacing(spacing_min)
    
    all_boxes = []   # [[x1_global, y1_global, x2_global, y2_global], ...]
    all_scores = []  # [confidence1, confidence2, ...]
    all_labels = []  # [1 or 2, 1 or 2, ...] (lymphocyte->1, monocyte->2)
    all_predictions = []  # 全予測結果を入れる

    for x_batch, y_batch, info in iterator:
        x_batch = x_batch.squeeze(0)
        y_batch = y_batch.squeeze(0)

        predictions = predictor.predict_on_batch(x_batch)
        
        c = info['x']
        r = info['y']
            
        for idx, prediction in enumerate(predictions):

            for detections in prediction:
                x, y, label, confidence,x1,y1,x2,y2 = detections.values()
                
                if confidence < class_thresholds[label]:
                    continue

                # if x == 128 or y == 128:
                #     continue

                if y_batch[idx][y][x] == 0:
                    continue

                x = x * ratio + c  # x is in spacing= 0.5 but c is in spacing = 0.25
                y = y * ratio + r
                
                # グローバルスケールに変換
                x1_global = x1 * ratio + c
                y1_global = y1 * ratio + r
                x2_global = x2 * ratio + c
                y2_global = y2 * ratio + r
                lbl_id = 1 if label == "lymphocyte" else 2
                
                all_boxes.append([x1_global, y1_global, x2_global, y2_global])
                all_scores.append(confidence)
                all_labels.append(lbl_id)
                
                
    print(f"Raw detection count: {len(all_boxes)}")
    wbf_boxes_abs, wbf_scores, wbf_labels = apply_wbf_from_arrays(
        all_boxes,
        all_scores,
        all_labels,
        x_size,
        y_size,
        iou_thr=0.3,
        skip_box_thr=0.0001
    )
    # WBF後のボックスを点に変換（中心点計算）
    for i, (box, score, lbl) in enumerate(zip(wbf_boxes_abs, wbf_scores, wbf_labels)):
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        # JSONに保存する形式を作成
        prediction_record = {
            "name": "Point "+str(i),
            "point": [
                px_to_mm(cx, spacing),
                px_to_mm(cy, spacing),
                0.24199951445730394,
            ],
            "probability": float(score),
        }
        if lbl == 1:  # lymphocyte
            output_dict["points"].append(prediction_record)
            output_dict_inflammatory_cells["points"].append(prediction_record)
        elif lbl == 2:  # monocyte
            output_dict_monocytes["points"].append(prediction_record)
            output_dict_inflammatory_cells["points"].append(prediction_record)

        annotations.append((cx, cy))

    print(len(output_dict["points"]), len(output_dict_monocytes["points"]), len(output_dict_inflammatory_cells["points"]))
    print(f"Predicted {len(annotations)} points after WBF")
    print("saving predictions...")

    # saving json file
    output_path_json = os.path.join(output_path, json_filename)
    write_json_file(
        location=output_path_json,
        content=output_dict
    )

    json_filename_monocytes = "detected-monocytes.json"
    # it should be replaced with correct json files
    output_path_json = os.path.join(output_path, json_filename_monocytes)
    write_json_file(
        location=output_path_json,
        content=output_dict_monocytes
    )

    json_filename_inflammatory_cells = "detected-inflammatory-cells.json"
    # it should be replaced with correct json files
    output_path_json = os.path.join(output_path, json_filename_inflammatory_cells)
    write_json_file(
        location=output_path_json,
        content=output_dict_inflammatory_cells
    )

    print("finished!")


def write_json_file(*, location, content):
    # Writes a json file
    with open(location, 'w') as f:
        f.write(json.dumps(content, indent=4))


def load_json_file(*, location):
    # Reads a json file
    with open(location) as f:
        return json.loads(f.read())


def _show_torch_cuda_info():
    import torch

    print("=+=" * 10)
    print("Collecting Torch CUDA information")
    print(f"Torch CUDA is available: {(available := torch.cuda.is_available())}")
    if available:
        print(f"\tnumber of devices: {torch.cuda.device_count()}")
        print(f"\tcurrent device: {(current_device := torch.cuda.current_device())}")
        print(f"\tproperties: {torch.cuda.get_device_properties(current_device)}")
    print("=+=" * 10)


if __name__ == "__main__":
    raise SystemExit(run())