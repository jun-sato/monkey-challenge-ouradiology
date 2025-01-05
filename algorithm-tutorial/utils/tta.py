import math
import numpy as np
from ensemble_boxes import weighted_boxes_fusion

def augmentation1(image):
    # Vertical flip
    # image: (H, W, C)
    aug_image = np.flipud(image)
    return aug_image

def augmentation2(image):
    # Horizontal flip
    # image: (H, W, C)
    aug_image = np.fliplr(image)
    return aug_image

def revert_vertical_flip(predictions, height):
    """
    predictions: list[dict] with keys 'x', 'y', 'bbox'
    bbox: [x1, y1, x2, y2]
    vertical flipで反転した座標を元に戻す
    """
    reverted = []
    for p in predictions:
        # y座標を反転前に戻す
        y_new = (height - 1) - p['y']
        x_new = p['x']  # xはそのまま

        # bboxも反転戻し
        x1, y1, x2, y2 = p['bbox']
        y1_new = (height - 1) - y2
        y2_new = (height - 1) - y1

        reverted.append({
            "x": x_new,
            "y": y_new,
            "label": p['label'],
            "confidence": p['confidence'],
            "bbox": [x1, y1_new, x2, y2_new]
        })
    return reverted

def revert_horizontal_flip(predictions, width):
    """
    horizontal flipで反転した座標を元に戻す
    """
    reverted = []
    for p in predictions:
        x_new = (width - 1) - p['x']
        y_new = p['y']

        x1, y1, x2, y2 = p['bbox']
        x1_new = (width - 1) - x2
        x2_new = (width - 1) - x1

        reverted.append({
            "x": x_new,
            "y": y_new,
            "label": p['label'],
            "confidence": p['confidence'],
            "bbox": [x1_new, y1, x2_new, y2]
        })
    return reverted

def iou(boxA, boxB):
    # box: [x1, y1, x2, y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0])*(boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0])*(boxB[3] - boxB[1])
    iou_val = interArea / float(boxAArea + boxBArea - interArea + 1e-5)
    return iou_val

def cluster_predictions(predictions, iou_threshold=0.5):
    # 簡易的な貪欲クラスタリング例
    # 同じlabel同士でまとめる
    clusters = []
    used = set()
    for i, p in enumerate(predictions):
        if i in used:
            continue
        cluster = [p]
        used.add(i)
        for j, q in enumerate(predictions[i+1:], i+1):
            if j in used:
                continue
            if p['label'] == q['label']:
                # IoUが閾値以上なら同一クラスタ
                if iou(p['bbox'], q['bbox']) > iou_threshold:
                    cluster.append(q)
                    used.add(j)
        clusters.append(cluster)
    return clusters

def average_cluster_confidence(cluster):
    # クラスタ内のconfidence平均を取る単純な例
    label = cluster[0]['label']
    avg_x = sum(p['x'] for p in cluster)/len(cluster)
    avg_y = sum(p['y'] for p in cluster)/len(cluster)
    avg_conf = sum(p['confidence'] for p in cluster)/len(cluster)

    # bboxも平均的に計算（より精度を求めるなら最大IoU箱など別戦略も可）
    avg_bbox = [
        sum(p['bbox'][0] for p in cluster)/len(cluster),
        sum(p['bbox'][1] for p in cluster)/len(cluster),
        sum(p['bbox'][2] for p in cluster)/len(cluster),
        sum(p['bbox'][3] for p in cluster)/len(cluster),
    ]

    return {
        "x": avg_x,
        "y": avg_y,
        "label": label,
        "confidence": avg_conf,
        "bbox": avg_bbox
    }

def average_cluster_iou(cluster):
    # クラスタ内で最もconfidenceが高い検出を基準とし、
    # その検出と他の検出のIoUで重み付けする例
    label = cluster[0]['label']
    # 基準: 一番confidenceが高いもの
    ref = max(cluster, key=lambda x: x['confidence'])
    
    weights = []
    x_sum = 0.0
    y_sum = 0.0
    conf_sum = 0.0
    bbox_x1_sum = 0.0
    bbox_y1_sum = 0.0
    bbox_x2_sum = 0.0
    bbox_y2_sum = 0.0
    weight_sum = 0.0
    
    for p in cluster:
        w = iou(ref['bbox'], p['bbox'])  # IoUを重みとする
        weights.append(w)
        x_sum += p['x'] * w
        y_sum += p['y'] * w
        conf_sum += p['confidence'] * w
        bbox_x1_sum += p['bbox'][0]*w
        bbox_y1_sum += p['bbox'][1]*w
        bbox_x2_sum += p['bbox'][2]*w
        bbox_y2_sum += p['bbox'][3]*w
        weight_sum += w

    if weight_sum < 1e-5:
        # 全くIoUがなかった場合は単純平均
        return average_cluster_confidence(cluster)

    return {
        "x": x_sum / weight_sum,
        "y": y_sum / weight_sum,
        "label": label,
        "confidence": conf_sum / weight_sum,
        "bbox": [
            bbox_x1_sum/weight_sum,
            bbox_y1_sum/weight_sum,
            bbox_x2_sum/weight_sum,
            bbox_y2_sum/weight_sum
        ]
    }




def apply_wbf_from_arrays(all_boxes, all_scores, all_labels, image_width, image_height, iou_thr=0.5, skip_box_thr=0.0001):
    """
    all_boxes: [[x1, y1, x2, y2], [x1, y1, x2, y2], ...]
    all_scores: [score1, score2, ...]
    all_labels: [label_id1, label_id2, ...]
    image_width, image_height: 全体画像サイズ
    iou_thr: WBFで使用するIoU閾値
    skip_box_thr: スコアがこれ未満のボックスを除外
    """
    # 0~1への正規化
    boxes_norm = []
    for box in all_boxes:
        x1, y1, x2, y2 = box
        boxes_norm.append([
            x1 / image_width,
            y1 / image_height,
            x2 / image_width,
            y2 / image_height
        ])

    boxes_list = [boxes_norm]        # 単一モデルのリストとして渡す
    scores_list = [all_scores]
    labels_list = [all_labels]

    wbf_boxes, wbf_scores, wbf_labels = weighted_boxes_fusion(
        boxes_list, scores_list, labels_list,
        weights=None,
        iou_thr=iou_thr,
        skip_box_thr=skip_box_thr
    )

    # 元スケールに戻す
    wbf_boxes_abs = []
    for (x1, y1, x2, y2) in wbf_boxes:
        wbf_boxes_abs.append([
            x1 * image_width,
            y1 * image_height,
            x2 * image_width,
            y2 * image_height
        ])

    return wbf_boxes_abs, wbf_scores, wbf_labels