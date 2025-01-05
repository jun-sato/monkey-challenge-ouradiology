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
import torch

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
    weight_roots = [os.path.join(Model_PATH, "model_newloss.pth"),
                    os.path.join(Model_PATH, "model_03.pth")]
    class_thresholds_list = [
        {"lymphocyte": 0.4, "monocyte": 0.15},
        {"lymphocyte": 0.3, "monocyte": 0.25}]
    # Process the inputs: any way you'd like
    _show_torch_cuda_info()

    patch_shape = (224, 224, 3)
    spacings = (0.25,)
    overlap = (112, 112)
    offset = (0, 0)
    center = False

    # Save your output
    (annotations_count, lymph_count, mono_count, inflam_count) = ensemble_inference(
        model_weight_list=weight_roots,
        image_path=image_path,
        mask_path=mask_path,
        output_path=output_path,
        spacing=0.25,
        json_filename="detected-lymphocytes.json",
        class_thresholds_list=class_thresholds_list,
        iou_threshold_for_cluster=0.5,
        detectron_threshold=0.15,
        nms_threshold=1
    )


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

def ensemble_inference(
    model_weight_list,
    # 以下は元のinferenceに準拠
    image_path,
    mask_path,
    output_path,
    spacing,
    json_filename,
    class_thresholds_list,
    iou_threshold_for_cluster=0.5,
    # Detectron2の基本threshold
    detectron_threshold=0.15,
    nms_threshold=1.0
):
    """
    複数モデルを順番にロードし、推論結果をまとめてアンサンブル(WBF)して出力する。
    """

    # 出力用JSONを作成
    output_dict_lymph = {
        "name": "lymphocytes",
        "type": "Multiple points",
        "version": {"major": 1, "minor": 0},
        "points": [],
    }
    output_dict_mono = {
        "name": "monocytes",
        "type": "Multiple points",
        "version": {"major": 1, "minor": 0},
        "points": [],
    }
    output_dict_inflammatory = {
        "name": "inflammatory-cells",
        "type": "Multiple points",
        "version": {"major": 1, "minor": 0},
        "points": [],
    }

    # ここに全モデルの全ボックスを集約
    ensemble_boxes = []
    ensemble_scores = []
    ensemble_labels = []
    
    spacing_min = 0.3
    # 画像の元サイズを取得 (WBFに必要)
    with WholeSlideImage(image_path) as wsi:
        spacing = wsi.get_real_spacing(spacing_min)  # 使わないなら spacing_ = spacing_min としてもOK
        x_size, y_size = wsi.get_shape_from_spacing(spacing_min)

    # ====== 複数モデルを順番に推論 ======
    for model_weight_path, class_thresholds in zip(model_weight_list,class_thresholds_list):
        print(f"\n--- Loading model: {model_weight_path} ---")
        print(f"Class thresholds: {class_thresholds}")
        # ---- 1) モデルロード ----
        if "03" in model_weight_path:
            sp = 0.3
            ov = (112, 0)
        else:
            sp = 0.25
            ov = (0, 112)
        patch_configuration = PatchConfiguration(
                patch_shape=(224,224,3),
                spacings=(sp,),
                overlap=ov,
                offset=(0, 0),
                center=False
            )

        # spacing_min = 0.25 などの部分は元コードを流用
        spacing_min = sp
        ratio = sp / spacing_min
        
        model = Detectron2DetectionPredictor(
            output_dir=output_path,
            threshold=detectron_threshold,
            nms_threshold=nms_threshold,
            weight_root=model_weight_path
        )
        
        iterator = create_patch_iterator(
            image_path=image_path,
            mask_path=mask_path,
            patch_configuration=patch_configuration,
            cpus=4,
            backend='asap'
        )

        # ---- 3) このモデルで全パッチ推論 ----
        for x_batch, y_batch, info in tqdm(iterator):
            x_batch = x_batch.squeeze(0)
            y_batch = y_batch.squeeze(0)

            predictions = model.predict_on_batch(x_batch)

            c = info['x']
            r = info['y']

            for idx, prediction in enumerate(predictions):
                for detections in prediction:
                    x, y, label, confidence, x1, y1, x2, y2 = detections.values()
                    if confidence < class_thresholds[label]:
                        continue
                    if y_batch[idx][y][x] == 0:
                        continue

                    # グローバル座標に変換
                    x = x * ratio + c
                    y = y * ratio + r
                    x1_global = x1 * ratio + c
                    y1_global = y1 * ratio + r
                    x2_global = x2 * ratio + c
                    y2_global = y2 * ratio + r

                    if label == "lymphocyte":
                        lbl_id = 1
                    else:
                        lbl_id = 2


                    ensemble_boxes.append([x1_global, y1_global, x2_global, y2_global])
                    ensemble_scores.append(confidence)
                    ensemble_labels.append(lbl_id)

        # ---- 4) イテレータ停止 (リソース開放) ----
        iterator.stop()

        # # ---- 5) モデルアンロード (GPUメモリ対策) ----
        # del model
        # torch.cuda.empty_cache()

    print(f"Raw detection count from all models: {len(ensemble_boxes)}")

    # ====== 全部の推論結果をWBF ======
    wbf_boxes_abs, wbf_scores, wbf_labels = apply_wbf_from_arrays(
        ensemble_boxes,
        ensemble_scores,
        ensemble_labels,
        x_size,
        y_size,
        iou_thr=0.3,
        skip_box_thr=0.0001
    )

    # ====== WBF後のボックスを中心点にしてJSONに保存 ======
    annotations = []
    for i, (box, score, lbl) in enumerate(zip(wbf_boxes_abs, wbf_scores, wbf_labels)):
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        prediction_record = {
            "name": f"Point {i}",
            "point": [
                px_to_mm(cx, spacing),  # spacing_minでpx->mm
                px_to_mm(cy, spacing), 
                0.24199951445730394,
            ],
            "probability": float(score),
        }
        if lbl == 1:  # lymphocyte
            output_dict_lymph["points"].append(prediction_record)
            output_dict_inflammatory["points"].append(prediction_record)
        elif lbl == 2:  # monocyte
            output_dict_mono["points"].append(prediction_record)
            output_dict_inflammatory["points"].append(prediction_record)

        annotations.append((cx, cy))

    print(
        len(output_dict_lymph["points"]), 
        len(output_dict_mono["points"]), 
        len(output_dict_inflammatory["points"])
    )
    print(f"Predicted {len(annotations)} points after WBF for ensemble.")
    print("saving predictions...")
    

    # ==== JSONファイル ====
    # lymphocytes
    output_path_json = os.path.join(output_path, json_filename)
    write_json_file(location=output_path_json, content=output_dict_lymph)

    # monocytes
    json_filename_monocytes = "detected-monocytes.json"
    output_path_json = os.path.join(output_path, json_filename_monocytes)
    write_json_file(location=output_path_json, content=output_dict_mono)

    # inflammatory-cells
    json_filename_inflammatory_cells = "detected-inflammatory-cells.json"
    output_path_json = os.path.join(output_path, json_filename_inflammatory_cells)
    write_json_file(location=output_path_json, content=output_dict_inflammatory)

    return (len(annotations),
            len(output_dict_lymph["points"]),
            len(output_dict_mono["points"]),
            len(output_dict_inflammatory["points"]))
    



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