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

##########################
# 既存のパラメータ類
##########################
patch_shapes = [
    (224, 224, 3)
]
overlaps = [
    (112, 112),
]
image_paths = [
    "./data/images/pas-cpg/A_P000002_PAS_CPG.tif",
    "./data/images/pas-cpg/B_P000001_PAS_CPG.tif",
    "./data/images/pas-cpg/C_P000021_PAS_CPG.tif",
    "./data/images/pas-cpg/D_P000001_PAS_CPG.tif",
]
mask_paths = [
    "./data/images/tissue-masks/A_P000002_mask.tif",
    "./data/images/tissue-masks/B_P000001_mask.tif",
    "./data/images/tissue-masks/C_P000021_mask.tif",
    "./data/images/tissue-masks/D_P000001_mask.tif",
]
thresholds = [0.15]      # Detectron2内部のthreshold
nms_thresholds = [1]     # Detectron2内部のnms_threshold
class_thresholds_lists = [[
    {"lymphocyte": 0.4, "monocyte": 0.15},
    {"lymphocyte": 0.4, "monocyte": 0.15},
    {"lymphocyte": 0.4, "monocyte": 0.15},]
]

# 「複数モデルをアンサンブルする」場合の重みリストを指定
# 例: modelA, modelBなど
model_weight_list = [
    "./outputs/model_newloss.pth",
    "./outputs/model_03_30.pth",
    "./outputs/model_03_20.pth",
]

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


########################################################
# 変更1: ensemble_inference 関数を新規作成
########################################################
def ensemble_inference(
    model_weight_list,
    # 以下は元のinferenceに準拠
    patch_configuration,
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
            ov = (112, 112)
        else:
            sp = 0.25
            ov = (112, 112)
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
                    
                    if "20" in model_weight_path:
                        w = (x2_global - x1_global) / 2
                        h = (y2_global - y1_global) / 2
                        center_x = x1_global + w
                        center_y = y1_global + h
                        x1_global = center_x - w*(3/2)
                        x2_global = center_x + w*(3/2)
                        y1_global = center_y - h*(3/2)
                        y2_global = center_y + h*(3/2)

                    if label == "lymphocyte":
                        lbl_id = 1
                    else:
                        lbl_id = 2


                    ensemble_boxes.append([x1_global, y1_global, x2_global, y2_global])
                    ensemble_scores.append(confidence)
                    ensemble_labels.append(lbl_id)
        # if "03" in model_weight_path:
        #     ensemble_scores = list(np.array(ensemble_scores)-0.05)
        # ---- 4) イテレータ停止 (リソース開放) ----
        iterator.stop()

        # ---- 5) モデルアンロード (GPUメモリ対策) ----
        del model
        torch.cuda.empty_cache()

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
    ## infammatory cellsのタスクのためのwbf
    wbf_boxes_abs_all, wbf_scores_all, wbf_labels_all = apply_wbf_from_arrays(
        ensemble_boxes,
        ensemble_scores,
        [1]*len(ensemble_labels),
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
            #output_dict_inflammatory["points"].append(prediction_record)
        elif lbl == 2:  # monocyte
            output_dict_mono["points"].append(prediction_record)
            #output_dict_inflammatory["points"].append(prediction_record)

        annotations.append((cx, cy))
        
    for i, (box, score, lbl) in enumerate(zip(wbf_boxes_abs_all, wbf_scores_all, wbf_labels_all)):
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
        output_dict_inflammatory["points"].append(prediction_record)


    print(
        len(output_dict_lymph["points"]), 
        len(output_dict_mono["points"]), 
        len(output_dict_inflammatory["points"])
    )
    print(f"Predicted {len(annotations)} points after WBF for ensemble.")
    print("saving predictions...")
    
    annotations_wsd = to_wsd(annotations)
    xml_filename = 'points_results.xml'
    output_path_xml = os.path.join(output_path,xml_filename)
    write_point_set(
        annotations_wsd,
        output_path_xml,
        label_color="blue",
    )
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


###################################
# メイン処理 (CSV書き込みループなど)
###################################
write_header = not os.path.exists(csv_file)
with open(csv_file, "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()

    for patch_shape, overlap, threshold, nms_threshold, class_thresholds_list in product(
        patch_shapes, overlaps, thresholds, nms_thresholds, class_thresholds_lists
    ):
        for image_path, mask_path in zip(image_paths, mask_paths):
            # 条件ごとの出力フォルダ
            output_path = f"./outputs/results_{patch_shape[0]}x{patch_shape[1]}_ov{overlap[0]}x{overlap[1]}_{os.path.basename(image_path)}_{os.path.basename(mask_path)}_th{threshold}_nms{nms_threshold}"
            os.makedirs(output_path, exist_ok=True)

            patch_configuration = PatchConfiguration(
                patch_shape=patch_shape,
                spacings=(0.25,),
                overlap=overlap,
                offset=(0, 0),
                center=False
            )

            # --- ensemble_inference を呼び出す ---
            (annotations_count, lymph_count, mono_count, inflam_count) = ensemble_inference(
                model_weight_list=model_weight_list,
                patch_configuration=patch_configuration,
                image_path=image_path,
                mask_path=mask_path,
                output_path=output_path,
                spacing=0.25,
                json_filename="detected-lymphocytes.json",
                class_thresholds_list=class_thresholds_list,
                iou_threshold_for_cluster=0.5,
                detectron_threshold=threshold,
                nms_threshold=nms_threshold
            )

            # evaluateのためにjsonをコピー
            dst_dir = f"/mnt/hdd2/monkey-challenge/evaluation/test/input/{os.path.basename(image_path).split('_PAS')[0]}/output/"
            os.makedirs(dst_dir, exist_ok=True)
            for file_path in glob.glob(output_path + "/*.json"):
                shutil.copy(file_path, dst_dir)

        # ==== 修正: evaluate.py 実行と標準出力取得 ====
        evaluate_command = ["python", "/mnt/hdd2/monkey-challenge/evaluation/evaluate.py"]
        result = subprocess.run(evaluate_command, capture_output=True, text=True)
        evaluation_stdout = result.stdout.strip()
        print('==================', result, evaluation_stdout)

        writer.writerow({
            "patch_shape": patch_shape,
            "overlap": overlap,
            "threshold": threshold,
            "nms_threshold": nms_threshold,
            "class_thresholds": json.dumps(class_thresholds_list),
            "annotations_count": annotations_count,
            "lymph_count": lymph_count,
            "mono_count": mono_count,
            "inflammatory_count": inflam_count,
            "evaluation_stdout": evaluation_stdout
        })
