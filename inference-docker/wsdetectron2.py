import os

import detectron2.data.transforms as T
import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from ensemble_boxes import weighted_boxes_fusion

SIZE = 224
AUG = T.FixedSizeCrop((SIZE, SIZE), pad_value=0)
inv_label_map = {
    0: "lymphocyte",
    1: "monocyte",
}


def transform(image):
    image = AUG.get_transform(image).apply_image(image)
    return image


class BatchPredictor(DefaultPredictor):
    """Run d2 on a list of images."""

    def __call__(self, images):

        input_images = []
        for image in images:
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                image = image[:, :, ::-1]
            height, width = image.shape[:2]
            #new_image = transform(image)
            new_image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

            input_images.append({"image": new_image, "height": height, "width": width})

        with torch.no_grad():
            preds = self.model(input_images)
        return preds


class Detectron2DetectionPredictor:
    def __init__(self, output_dir, threshold, nms_threshold, weight_root):
        cfg = get_cfg()
        if '999' in weight_root:
            cfg.merge_from_file(
                model_zoo.get_config_file(
                    "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
                )
            )
        else:
            cfg.merge_from_file(
                model_zoo.get_config_file(
                    "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"
                )
            )


        cfg.DATASETS.TRAIN = ("detection_dataset2",)
        cfg.DATASETS.TEST = ()
        cfg.DATALOADER.NUM_WORKERS = 1

        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
        cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[16, 24, 32]]
        #cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[1.0]]


        cfg.SOLVER.IMS_PER_BATCH = 256
        cfg.SOLVER.BASE_LR = 0.002  # pick a good LR
        cfg.SOLVER.MAX_ITER = 1000  # 2000 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
        cfg.SOLVER.STEPS = (500, 750)
        cfg.SOLVER.WARMUP_ITERS = 100
        cfg.SOLVER.WARMUP_FACTOR = 1.0/1000 
        cfg.SOLVER.GAMMA = 0.5
        cfg.SOLVER.CHECKPOINT_PERIOD = 200

        ###
        ###



        cfg.OUTPUT_DIR = str(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        cfg.MODEL.WEIGHTS = weight_root
        

        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
        cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = nms_threshold
        cfg.MODEL.RPN.NMS_THRESH = nms_threshold

        self._predictor = BatchPredictor(cfg)

    def predict_on_batch(self, x_batch):
        # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        outputs = self._predictor(x_batch)
        predictions = []
        for output in outputs:
            predictions.append([])
            pred_boxes = output["instances"].get("pred_boxes")

            scores = output["instances"].get("scores")
            classes = output["instances"].get("pred_classes")
            centers = pred_boxes.get_centers()
            for idx, center in enumerate(centers):
                x, y = center.cpu().detach().numpy()
                pred_box = pred_boxes[idx].tensor.cpu().detach().numpy()[0]
                confidence = scores[idx].cpu().detach().numpy()
                label = inv_label_map[int(classes[idx].cpu().detach())]
                # if confidence >= class_thresholds[label]:
                prediction_record = {
                    "x": int(x),
                    "y": int(y),
                    "label": str(label),
                    "confidence": float(confidence),
                    "x1": pred_box[0],
                    "y1": pred_box[1],
                    "x2": pred_box[2],
                    "y2": pred_box[3]
                }
                predictions[-1].append(prediction_record)
        return predictions
    
    
def apply_wbf_from_arrays(all_boxes, all_scores, all_labels, image_width, image_height, iou_thr=0.5, skip_box_thr=0.001):
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