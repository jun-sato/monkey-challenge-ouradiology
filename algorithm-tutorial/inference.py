import os
import json
import torch
import creationism
from tqdm import tqdm
from itertools import product
import subprocess
import csv
import glob
import shutil

from wholeslidedata.interoperability.asap.annotationwriter import write_point_set
from wholeslidedata.image.wholeslideimage import WholeSlideImage
from wholeslidedata.iterators import create_patch_iterator, PatchConfiguration
from wholeslidedata.annotation.labels import Label

from utils.wsdetectron2 import Detectron2DetectionPredictor
from utils.structures import Point
from utils.tta import *

print(f"Pytorch GPU available: {torch.cuda.is_available()}")

# ==== 修正1: 複数の候補リスト ====

## patch sizeの変更
## anchor sizeの変更
## overlap??

patch_shapes = [
    #(96, 96, 3)
    (224, 224, 3)
]
overlaps = [
    # (0, 0),
    # (32, 32),
    (0, 0),
    # (64, 64)
]
image_paths = [
    "./data/images/pas-cpg/A_P000003_PAS_CPG.tif",
    "./data/images/pas-cpg/B_P000002_PAS_CPG.tif",
    "./data/images/pas-cpg/C_P000022_PAS_CPG.tif",
    "./data/images/pas-cpg/D_P000002_PAS_CPG.tif",
]
mask_paths = [
    "./data/images/tissue-masks/A_P000003_mask.tif",
    "./data/images/tissue-masks/B_P000002_mask.tif",
    "./data/images/tissue-masks/C_P000022_mask.tif",
    "./data/images/tissue-masks/D_P000002_mask.tif",
]
thresholds = [0.1]     # Detectron2内部のthreshold
nms_thresholds = [1]  # Detectron2内部のnms_threshold
class_thresholds_list = [
    # {"lymphocyte": 0.7, "monocyte": 0.3},
    # {"lymphocyte": 0.6, "monocyte": 0.25},
    # {"lymphocyte": 0.5, "monocyte": 0.2},
    {"lymphocyte": 0.4, "monocyte": 0.15},
    # {"lymphocyte": 0.3, "monocyte": 0.1}
]
weight_root = "./outputs/model_0000999.pth"

##best 128, 0.6,0.25,0.15,(0,0)
##best 224, 0.4,0.15,0.15,(8,0)
def px_to_mm(px: int, spacing: float):
    return px * spacing / 1000

def to_wsd(points):
    """Convert list of coordinates into WSD points"""
    new_points = []
    for i, point in enumerate(points):
        p = Point(
            index=i,
            label=Label("lymphocyted", 1, color="blue"),
            coordinates=[point],
        )
        new_points.append(p)
    return new_points

def write_json_file(*, location, content):
    # Writes a json file
    with open(location, 'w') as f:
        f.write(json.dumps(content, indent=4))
        
def inference(iterator, predictor, spacing, image_path, output_path, json_filename, class_thresholds,iou_threshold_for_cluster=0.5):
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

    annotations = []
    counter = 0
    
    spacing_min = 0.25
    ratio = spacing/spacing_min
    with WholeSlideImage(image_path) as wsi:
        spacing = wsi.get_real_spacing(spacing_min)
        x_size, y_size = wsi.get_shape_from_spacing(spacing_min)
    # 全パッチの検出を集約
    all_boxes = []   # [[x1_global, y1_global, x2_global, y2_global], ...]
    all_scores = []  # [confidence1, confidence2, ...]
    all_labels = []  # [1 or 2, 1 or 2, ...] (lymphocyte->1, monocyte->2)
    all_predictions = []  # 全予測結果を入れる
    
    for x_batch, y_batch, info in tqdm(iterator):
        x_batch = x_batch.squeeze(0)
        y_batch = y_batch.squeeze(0)

        predictions = predictor.predict_on_batch(x_batch)
        
        c = info['x']
        r = info['y']
        
        for idx, prediction in enumerate(predictions):

            for detections in prediction:
                x, y, label, confidence,x1,y1,x2,y2 = detections.values()
                #print(f'x: {x}, y: {y}, label: {label}, confidence: {confidence},{y_batch[idx].shape}')
                if confidence < class_thresholds[label]:
                    continue
                
                if y_batch[idx][y][x] == 0:
                    continue
                
                x = x*ratio + c # x is in spacing= 0.5 but c is in spacing = 0.25
                y = y*ratio + r

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
                # prediction_record = {
                #     "name" : "Point "+str(counter),
                #     "point": [
                #         px_to_mm(x, spacing),
                #         px_to_mm(y, spacing),
                #         0.24199951445730394,
                #     ],
                #     "probability": confidence,
                # }
                
                # if label == 'lymphocyte':
                #     output_dict["points"].append(prediction_record)
                #     output_dict_inflammatory_cells["points"].append(prediction_record)
                # elif label == 'monocyte':
                #     output_dict_monocytes["points"].append(prediction_record)
                #     output_dict_inflammatory_cells["points"].append(prediction_record)
                
                # annotations.append((x, y))
                # counter += 1

        # print(len(output_dict["points"]),len(output_dict_monocytes["points"]),len(output_dict_inflammatory_cells["points"]))  

        # print(f"Predicted {len(annotations)} points")
        # print("saving predictions...")

    # saving xml file
    annotations_wsd = to_wsd(annotations)
    xml_filename = 'points_results.xml'
    output_path_xml = os.path.join(output_path,xml_filename)
    write_point_set(
        annotations_wsd,
        output_path_xml,
        label_color="blue",
    )

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
    return (len(annotations),
            len(output_dict["points"]),
            len(output_dict_monocytes["points"]),
            len(output_dict_inflammatory_cells["points"]))

# ==== 修正2: CSV出力 ====
csv_file = "hyperparam_tuning.csv"
fieldnames = [
    "patch_shape",
    "overlap",
    "threshold",
    "nms_threshold",
    "class_thresholds",
    "annotations_count",
    "lymph_count",
    "mono_count",
    "inflammatory_count",
    "evaluation_stdout"
]


# CSVファイルが無ければヘッダを書き込む
write_header = not os.path.exists(csv_file)
with open(csv_file, "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()

    # ==== 修正3: itertools.productで全組み合わせ生成 ====
    for patch_shape, overlap, threshold, nms_threshold, class_thresholds in product(
        patch_shapes, overlaps, thresholds, nms_thresholds, class_thresholds_list
    ):
        for image_path, mask_path in zip(image_paths, mask_paths):
            # 条件ごとの出力フォルダ
            output_path = f"./outputs/results_{patch_shape[0]}x{patch_shape[1]}_ov{overlap[0]}x{overlap[1]}_{os.path.basename(image_path)}_{os.path.basename(mask_path)}_th{threshold}_nms{nms_threshold}"
            os.makedirs(output_path, exist_ok=True)

            patch_configuration = PatchConfiguration(
                patch_shape=patch_shape,
                spacings=(0.25,),
                overlap=overlap,
                offset=(0,0),
                center=False
            )

            model = Detectron2DetectionPredictor(
                output_dir=output_path,
                threshold= threshold,
                nms_threshold=nms_threshold,
                weight_root = weight_root
            )

            iterator = create_patch_iterator(
                image_path=image_path,
                mask_path=mask_path,
                patch_configuration=patch_configuration,
                cpus=4,
                backend='asap'
            )

            annotations_count, lymph_count, mono_count, inflam_count = inference(
                iterator=iterator,
                predictor=model,
                spacing = 0.25,
                image_path=image_path,
                output_path=output_path,
                json_filename="detected-lymphocytes.json",
                class_thresholds=class_thresholds
            )

            iterator.stop()
            
            #evaluateのためにjsonをコピー
            dst_dir = f"/mnt/hdd2/monkey-challenge/evaluation/test/input/{os.path.basename(image_path).split('_PAS')[0]}/output/"
            for file_path in glob.glob(output_path + "/*.json"):
                shutil.copy(file_path, dst_dir)

        # ==== 修正4: evaluate.py 実行と標準出力取得 ====
        # 想定として、evaluate.pyは結果を標準出力に出力
        # subprocessで呼び出し、stdoutをキャプチャ
        evaluate_command = ["python", "/mnt/hdd2/monkey-challenge/evaluation/evaluate.py"]
        # ここはevaluate.pyのインターフェースに合わせて引数変更してください
        result = subprocess.run(evaluate_command, capture_output=True, text=True)
        evaluation_stdout = result.stdout.strip()
        print('==================',result,evaluation_stdout)

        # CSVへ書き込み
        writer.writerow({
            "patch_shape": patch_shape,
            "overlap": overlap,
            "threshold": threshold,
            "nms_threshold": nms_threshold,
            "class_thresholds": json.dumps(class_thresholds),
            "annotations_count": annotations_count,
            "lymph_count": lymph_count,
            "mono_count": mono_count,
            "inflammatory_count": inflam_count,
            "evaluation_stdout": evaluation_stdout
        })