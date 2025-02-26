import torch
import torch.nn.functional as F
from detectron2.modeling.roi_heads import StandardROIHeads, ROI_HEADS_REGISTRY
from utils.vis import FocalLoss  # もとのFocalLossの定義位置に応じてパスを調整



from detectron2.modeling.roi_heads import StandardROIHeads
import torch
import torch.nn as nn
from detectron2.modeling.roi_heads import StandardROIHeads
from detectron2.modeling.roi_heads import ROI_HEADS_REGISTRY

# @ROI_HEADS_REGISTRY.register()
class MyFocalROIHeads(StandardROIHeads):
    # def __init__(self, cfg, input_shape):
    #     super().__init__(cfg, input_shape)
    #     # ドロップアウトレイヤーを定義 (p=0.5は例)
    #     self.dropout = nn.Dropout(p=0.5)
    
    # def _forward_box(self, features, proposals):
    #     if self.training:
    #         # トレーニング時
    #         pred_class_logits, pred_proposal_deltas = super()._forward_box(features, proposals)
    #         pred_class_logits = self.dropout(pred_class_logits)
    #         return pred_class_logits, pred_proposal_deltas
    #     else:
    #         # 推論時は1つだけ返る
    #         pred_instances = super()._forward_box(features, proposals)
    #         # 推論時はドロップアウト不要、またpred_instancesは既に最終出力なのでそのまま返す
    #         return pred_instances

    
    def losses(self, predictions, proposals):
        """
        FROCを最適化する分類損失と回帰損失を含むカスタムROIヘッド。
        
        Args:
            predictions: モデルの出力
                - pred_class_logits: 分類スコア (N, num_classes)
                - pred_proposal_deltas: ボックス回帰のデルタ (N, 4)
            proposals: RoIAlign後の提案ボックス
                - gt_classes: 各ボックスの正解クラス
                - gt_boxes: 各ボックスの正解座標
        Returns:
            dict: 計算した損失を含む辞書
        """
        pred_class_logits, pred_proposal_deltas = predictions

        # GTクラスとGTボックスを抽出
        gt_classes = torch.cat([p.gt_classes for p in proposals], dim=0)
        gt_boxes = torch.cat([p.gt_boxes.tensor for p in proposals], dim=0)
        
        # 提案ボックスとスコアを抽出
        pred_boxes = self.box_predictor.predict_boxes((None, pred_proposal_deltas), proposals)
        pred_scores = self.box_predictor.predict_probs((pred_class_logits, None), proposals)
        
        # 提案ボックスとスコアを結合 (flatten)
        pred_boxes = torch.cat(pred_boxes, dim=0)  # (N, 4)
        pred_scores = torch.cat(pred_scores, dim=0)  # (N,) - 正例スコアを取得
        
        # multi_class_froc_lossでクラスごとのTP/FP判定を行う
        froc_loss_cls = multi_class_froc_loss(
            pred_boxes=pred_boxes,
            pred_scores=pred_scores,   # (N,3)
            gt_boxes=gt_boxes,         # (M,4)
            gt_classes=gt_classes,     # (M,)
            distance_thresholds={1: 16.0, 2: 20.0},  # px単位 or mm単位に合わせて要調整
        )

        # --- 回帰損失 (FROCに基づく) ---
        froc_loss_box_reg = multi_class_froc_bbox_loss(
            pred_boxes=pred_boxes,
            pred_scores=pred_scores,
            gt_boxes=gt_boxes,
            gt_classes=gt_classes,
            distance_thresholds={1: 16.0, 2: 20.0},  # px単位 or mm単位に合わせて要調整
        )

        return {
            "loss_cls": froc_loss_cls,
            "loss_box_reg": froc_loss_box_reg,
        }

        # 分類損失 (FROCに基づく)
        froc_loss_cls = froc_loss(pred_boxes, pred_scores, gt_boxes)

        # 回帰損失 (FROCに基づく)
        froc_loss_box_reg = froc_bbox_loss(pred_boxes, gt_boxes, distance_threshold=20)

        return {
            "loss_cls": froc_loss_cls,        # 分類損失
            "loss_box_reg": froc_loss_box_reg  # 回帰損失
        }

def multi_class_froc_loss(
    pred_boxes,
    pred_scores,
    gt_boxes,
    gt_classes,
    distance_thresholds,
    eps=1e-6,
):
    """
    多クラス対応のFROCベース分類損失。
    pred_scores: (N, num_classes=3) -> [背景, lymphocyte, monocyte]
    """
    device = pred_boxes.device
    num_preds = pred_boxes.size(0)
    if num_preds == 0:
        # 予測が全く無い場合 -> 全てのGTが未検出
        # ここでは Σ(GT数) をロスとする形にしても良いし、0にしても良い
        # 適宜チューニングしてください
        return gt_boxes.size(0) * 1.0

    # まず予測クラス(= argmax)とその確信度(= max)を取り出す
    # ※「クラスごとのスコア全部」を使ってロスを構築する方法もありますが、
    #   ここではシンプルに「予測クラス=argmax / そのスコア=max」でTP/FP判定します
    pred_probs, pred_labels = torch.max(pred_scores, dim=1)  # (N,)

    # バウンディングボックス中心
    pred_centers = (pred_boxes[:, :2] + pred_boxes[:, 2:]) / 2  # (N, 2)
    gt_centers   = (gt_boxes[:, :2]   + gt_boxes[:, 2:])   / 2  # (M, 2)

    # 統合的にロスを加算していく
    total_loss = torch.zeros(1, device=device)

    # ====== クラス1(lymphocyte), クラス2(monocyte) のそれぞれでFROCロスを計算 ======
    # 背景(0)は "正例として検出してはいけない" クラス扱い
    for cls_id, dist_thresh in distance_thresholds.items():
        # cls_id の予測だけを抽出
        cls_mask_pred = (pred_labels == cls_id)
        # cls_id のGTだけを抽出
        cls_mask_gt = (gt_classes == cls_id)

        pred_centers_cls = pred_centers[cls_mask_pred]  # (N_cls, 2)
        pred_probs_cls   = pred_probs[cls_mask_pred]    # (N_cls,)
        num_preds_cls    = pred_centers_cls.size(0)

        gt_centers_cls = gt_centers[cls_mask_gt]        # (M_cls, 2)
        num_gts_cls    = gt_centers_cls.size(0)

        # GTが無い場合
        if num_gts_cls == 0:
            # このクラスとしてのGTは存在しない -> このクラスで出した予測は全てFP
            fp_loss = pred_probs_cls.sum()  # スコアの総和をペナルティ
            total_loss += fp_loss
            continue

        # 予測が無い場合
        if num_preds_cls == 0:
            # このクラスのGTが全部未検出 -> 全部FN
            # ここでは "GT数ぶんのロス" にしてみる
            fn_loss = torch.tensor(num_gts_cls, device=device, dtype=torch.float32)
            total_loss += fn_loss
            continue

        # 中心間距離
        dist_matrix = torch.cdist(pred_centers_cls, gt_centers_cls)  # (N_cls, M_cls)

        # TP判定: 距離が閾値以下
        # ※ 通常は1つのGTにつき1つのpredとのみマッチさせるが
        #   簡易実装として "どれか1つでも閾値内ならTP" という判定にする
        #   より厳密にやりたい場合はハンガリアンマッチング等を使用してください
        tp_mask = (dist_matrix <= dist_thresh).any(dim=1).float()  # (N_cls,)
        fp_mask = 1 - tp_mask

        # 損失計算 (二値分類的なFROC)
        # - TP: 高スコアになるほどロス減 -> -log(prob)
        # - FP: スコアが高いほど罰則 -> + prob
        tp_loss = -torch.log(pred_probs_cls + eps) * tp_mask
        fp_loss = pred_probs_cls * fp_mask

        loss_cls = tp_loss.sum() + fp_loss.sum()
        total_loss += loss_cls

    # 予測数 (全クラス含む) で正規化するなど、好みに応じて調整
    return total_loss / (num_preds + eps)


def multi_class_froc_bbox_loss(
    pred_boxes,
    pred_scores,
    gt_boxes,
    gt_classes,
    distance_thresholds
):
    """
    多クラス対応のFROCベース回帰損失。
    クラスごとに距離閾値を変えて中心点の距離をペナルティ化。
    """
    device = pred_boxes.device
    num_preds = pred_boxes.size(0)
    if num_preds == 0:
        # 予測が無いなら回帰損失は0にしてもよいし、一定ペナルティにしても良い
        return torch.tensor(0.0, device=device)

    # 予測クラス(=argmax) をとる
    pred_probs, pred_labels = torch.max(pred_scores, dim=1)  # (N,)

    # バウンディングボックス中心
    pred_centers = (pred_boxes[:, :2] + pred_boxes[:, 2:]) / 2  # (N, 2)
    gt_centers   = (gt_boxes[:, :2] + gt_boxes[:, 2:])   / 2    # (M, 2)

    total_loss = torch.zeros(1, device=device)

    for cls_id, dist_thresh in distance_thresholds.items():
        cls_mask_pred = (pred_labels == cls_id)
        cls_mask_gt   = (gt_classes == cls_id)

        pred_centers_cls = pred_centers[cls_mask_pred]
        gt_centers_cls   = gt_centers[cls_mask_gt]
        num_preds_cls    = pred_centers_cls.size(0)
        num_gts_cls      = gt_centers_cls.size(0)

        if num_preds_cls == 0 or num_gts_cls == 0:
            # 予測もしくはGTが無いなら損失=0
            continue

        # (N_cls, M_cls)
        distances = torch.cdist(pred_centers_cls, gt_centers_cls)

        # シンプル化: 「どれか1つでも閾値内の GT があれば、それと距離を計算」
        #   -> 予測ボックス i に対して最小距離を取る
        min_dists, _ = torch.min(distances, dim=1)  # (N_cls,)

        # FROC流: 閾値以下 -> (dist^2), 閾値超過 -> (dist - thresh)^2
        loss_cls = torch.where(
            min_dists <= dist_thresh,
            min_dists**2,
            (min_dists - dist_thresh)**2
        ).mean()

        total_loss += loss_cls

    return total_loss



def froc_loss(pred_boxes, pred_scores, gt_boxes, distance_threshold=5.0):
    """
    FROC最適化のための分類損失。
    """
    device = pred_boxes.device
    num_preds = pred_boxes.size(0)
    num_gts = gt_boxes.size(0)
    
    if num_preds == 0:
        return torch.tensor(num_gts, dtype=torch.float32, device=device)

    # バウンディングボックス中心の計算
    pred_centers = (pred_boxes[:, :2] + pred_boxes[:, 2:]) / 2  # (N, 2)
    gt_centers = (gt_boxes[:, :2] + gt_boxes[:, 2:]) / 2  # (M, 2)
    
    # 中心間の距離計算
    dists = torch.cdist(pred_centers, gt_centers)  # (N, M)
    
    # TP判定: 距離が閾値以下
    tp_mask = (dists <= distance_threshold)  # (N, M)
    tp = tp_mask.any(dim=1).float()  # 各予測がTPなら1, それ以外は0
    fp = 1 - tp  # TPでない予測はFP

    # 損失計算
    tp_loss = -torch.log(pred_scores + 1e-6) * tp
    fp_loss = pred_scores * fp

    # 総損失
    loss = tp_loss.sum() + fp_loss.sum()
    return loss / num_preds


def froc_bbox_loss(pred_boxes, gt_boxes, distance_threshold=5.0):
    """
    FROC最適化のための回帰損失。
    """
    device = pred_boxes.device
    num_preds = pred_boxes.size(0)
    num_gts = gt_boxes.size(0)

    if num_preds == 0 or num_gts == 0:
        return torch.tensor(0.0, dtype=torch.float32, device=device)

    # バウンディングボックス中心の計算
    pred_centers = (pred_boxes[:, :2] + pred_boxes[:, 2:]) / 2  # (N, 2)
    gt_centers = (gt_boxes[:, :2] + gt_boxes[:, 2:]) / 2  # (M, 2)

    # 中心間の距離計算
    distances = torch.cdist(pred_centers, gt_centers)  # (N, M)

    # 距離が閾値以下の場合のマスク
    distance_mask = distances <= distance_threshold

    # 損失計算
    loss = torch.where(distance_mask, distances ** 2, (distances - distance_threshold) ** 2)
    return loss.mean()

# @ROI_HEADS_REGISTRY.register()
# class MyFocalROIHeads(StandardROIHeads):
#     def losses(self, predictions, proposals):
#         pred_class_logits, pred_proposal_deltas = predictions
#         gt_classes = torch.cat([p.gt_classes for p in proposals], dim=0)
        
#         focal_loss_fn = FocalLoss(gamma=2.0, alpha=0.25)
#         loss_cls = focal_loss_fn(pred_class_logits, gt_classes)

#         box_transform = self.box_predictor.box2box_transform
#         loss_box_reg = self.box_predictor.losses((None, pred_proposal_deltas), proposals, box_transform)['loss_box_reg']

#         scores = F.softmax(pred_class_logits, dim=1) 
#         foreground_scores = scores[:,1:]  # lymphocyte(0), monocyte(1)
#         max_scores, pred_cls = foreground_scores.max(dim=1)

#         proposal_boxes = torch.cat([p.proposal_boxes.tensor for p in proposals], dim=0)
#         pred_deltas_all = pred_proposal_deltas.view(-1, (1 + self.num_classes)*4)
        
#         batch_idx = torch.arange(pred_cls.shape[0], device=pred_cls.device)
#         class_delta_indices = pred_cls + 1
#         pred_deltas_selected = torch.stack([
#             pred_deltas_all[batch_idx, class_delta_indices*4 + i] for i in range(4)
#         ], dim=1)

#         pred_boxes = self.box_predictor.box2box_transform.apply_deltas(pred_deltas_selected, proposal_boxes)

#         gt_boxes = torch.cat([p.gt_boxes.tensor for p in proposals], dim=0)
#         G = gt_boxes.shape[0]

#         if G == 0:
#             loss_froc = torch.tensor(0.0, device=gt_boxes.device)
#         else:
#             iou_matrix = pairwise_iou(pred_boxes, gt_boxes)
#             class_match_mask = (pred_cls[:, None] == gt_classes[None, :])
#             iou_matrix = iou_matrix * class_match_mask.float()

#             max_iou_vals, _ = iou_matrix.max(dim=1)
#             temp = 0.1
#             tp_indicator = torch.sigmoid((max_iou_vals - 0.5)/temp)
#             fp_indicator = 1 - tp_indicator

#             thresholds = torch.tensor([0.1,0.3,0.5,0.7,0.9], device=scores.device)
#             froc_loss_all = []

#             for t in thresholds:
#                 detect_prob = torch.sigmoid((max_scores - t)/temp)
#                 expected_TP = (detect_prob * tp_indicator).sum()
#                 expected_FP = (detect_prob * fp_indicator).sum()

#                 sensitivity = expected_TP / G
#                 fp_weight = torch.sigmoid((2 - expected_FP)/0.5)

#                 froc_loss_t = -sensitivity * fp_weight
#                 froc_loss_all.append(froc_loss_t)

#             loss_froc = torch.mean(torch.stack(froc_loss_all)) if len(froc_loss_all)>0 else 0.0

#         total_loss = loss_cls + loss_box_reg + 0.5 * loss_froc
#         return {
#             "loss_cls": loss_cls,
#             "loss_box_reg": loss_box_reg,
#             "loss_froc": loss_froc,
#             "total_loss": total_loss
#         }


from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers
from detectron2.structures import Boxes

from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers
from detectron2.modeling.roi_heads import select_foreground_proposals

import torch
import torch.nn.functional as F
from detectron2.layers import cat
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers
from detectron2.structures import Boxes
from detectron2.modeling.roi_heads import select_foreground_proposals

def _log_classification_stats(pred_logits, gt_classes):
    """
    Detectron2標準実装の `_log_classification_stats` と同等のログ用関数
    """
    # 必要に応じて統計表示の実装を入れる
    pass


def froc_loss(pred_boxes, pred_scores, gt_boxes, distance_threshold=5.0):
    """
    FROCを用いた簡易的な分類ロス（1クラス想定）。
    pred_scores: (N,) 0～1の検出スコア (前景確信度)
    pred_boxes:  (N, 4)
    gt_boxes:    (M, 4)
    distance_threshold: 中心間距離の閾値 (pxやmm等)
    
    戻り値: ロス値 (スカラー)
    """
    device = pred_boxes.device
    num_preds = pred_boxes.size(0)
    num_gts = gt_boxes.size(0)

    # GTが存在するのに予測がない -> FN = GT数ぶんペナルティ
    if num_preds == 0:
        # ここでは単純に "未検出のGT数" をロスに加える
        return torch.tensor(float(num_gts), device=device)

    # バウンディングボックス中心 (N, 2) / (M, 2)
    pred_centers = (pred_boxes[:, :2] + pred_boxes[:, 2:]) / 2
    gt_centers   = (gt_boxes[:, :2]   + gt_boxes[:, 2:])   / 2

    # 中心間距離 (N, M)
    dists = torch.cdist(pred_centers, gt_centers)
    
    # TP判定: 少なくとも1つのGTと distance_threshold 以下であればTP
    tp_mask = (dists <= distance_threshold).any(dim=1).float()  # (N,)
    fp_mask = 1.0 - tp_mask  # TPでない -> FP

    # ロス計算例：
    #  - TP → スコアが高いほどロス小さく: -log(score)
    #  - FP → スコアが高いほどペナルティ大きく: + score
    eps = 1e-6
    tp_loss = -torch.log(pred_scores + eps) * tp_mask
    fp_loss = pred_scores * fp_mask

    loss = tp_loss.sum() + fp_loss.sum()
    return loss / num_preds


class FocalLoss(torch.nn.Module):
    """
    例: シンプルなマルチクラスFocalLoss
    """
    def __init__(self, gamma=4.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        """
        logits: [N, C] (未softmax)
        targets: [N]   (0 <= targets < C)
        """
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal = self.alpha * (1 - pt)**self.gamma * ce_loss
        return focal.mean()

def diou_loss(pred_boxes, gt_boxes, eps=1e-7):
    """
    Distance IoU Loss の例
    pred_boxes, gt_boxes: [N, 4], (x1, y1, x2, y2)
    戻り値: shape [N] (各サンプルの Diou loss)
    """
    x1p, y1p, x2p, y2p = pred_boxes[:, 0], pred_boxes[:, 1], pred_boxes[:, 2], pred_boxes[:, 3]
    x1g, y1g, x2g, y2g = gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 2], gt_boxes[:, 3]

    # IoU
    inter_x1 = torch.max(x1p, x1g)
    inter_y1 = torch.max(y1p, y1g)
    inter_x2 = torch.min(x2p, x2g)
    inter_y2 = torch.min(y2p, y2g)
    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    area_p = (x2p - x1p).clamp(min=0) * (y2p - y1p).clamp(min=0)
    area_g = (x2g - x1g).clamp(min=0) * (y2g - y1g).clamp(min=0)
    union_area = area_p + area_g - inter_area + eps
    iou = inter_area / union_area

    # 中心点距離
    px = (x1p + x2p) / 2
    py = (y1p + y2p) / 2
    gx = (x1g + x2g) / 2
    gy = (y1g + y2g) / 2
    center_dist_sq = (px - gx)**2 + (py - gy)**2

    # 外接矩形の対角線長さ
    enclose_x1 = torch.min(x1p, x1g)
    enclose_y1 = torch.min(y1p, y1g)
    enclose_x2 = torch.max(x2p, x2g)
    enclose_y2 = torch.max(y2p, y2g)
    enclose_w = (enclose_x2 - enclose_x1).clamp(min=0)
    enclose_h = (enclose_y2 - enclose_y1).clamp(min=0)
    c2 = enclose_w**2 + enclose_h**2 + eps

    diou = iou - (center_dist_sq / c2)*5
    diou = torch.clamp(diou, min=-1.0, max=1.0)
    return 1.0 - diou




class MyFastRCNNOutputLayers(FastRCNNOutputLayers):
    """
    Detectron2のFastRCNNOutputLayersを継承し、
    - 分類ロス: FocalLoss
    - ボックス回帰ロス: DIoU
    の組み合わせに差し替える例。
    """
    def losses(self, predictions, proposals):
        """
        predictions: (scores, proposal_deltas)
            scores: shape [N, C]
            proposal_deltas: shape [N, 4*C]
        proposals (list[Instances]):
            Each must have "proposal_boxes", and (if foreground) "gt_boxes", "gt_classes".
            If no gt_boxes => negative proposals => not used for regression.
        
        Returns: Dict[str, Tensor]
        """
        scores, proposal_deltas = predictions

        # 1) parse classification outputs
        # proposals が空なら empty tensor
        if len(proposals):
            gt_classes = cat([p.gt_classes for p in proposals], dim=0)
        else:
            gt_classes = torch.empty(0, device=scores.device, dtype=torch.long)

        # Detectron2の標準関数: 分類に関する統計ログを出す（任意）
        _log_classification_stats(scores, gt_classes)

        # 2) parse box regression outputs
        if len(proposals):
            # Nx4
            proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim=0)
            # gt_boxesが無いものは proposal_boxes を代わりに使う（= 負例扱い）
            gt_boxes_ = cat(
                [(p.gt_boxes if p.has("gt_boxes") else p.proposal_boxes).tensor for p in proposals],
                dim=0,
            )
        else:
            proposal_boxes = gt_boxes_ = torch.empty((0, 4), device=scores.device)

        # ----- 分類ロス: Focal Loss -----
        if len(gt_classes) > 0:
            focal_loss_fn = FocalLoss(gamma=1, alpha=0.25)
            loss_cls = focal_loss_fn(scores, gt_classes)*5
        else:
            # 提案が無い場合はロス0（または空）
            loss_cls = scores.sum() * 0.0

        # ----- ボックス回帰ロス: DIoU -----
        # Detectron2のデフォルト挙動:
        #   "もし gt_boxes が無い（= 負例）なら そのproposalは回帰ロスに含めない"
        #   "つまり gt_classes < 0 or >= num_classes も除外"
        
        loss_box_reg = self.diou_box_loss(proposal_boxes, gt_boxes_, proposal_deltas, gt_classes)

        losses = {
            "loss_cls": loss_cls,
            "loss_box_reg": loss_box_reg,
        }
        # self.loss_weight: 親クラスで {"loss_cls": 1.0, "loss_box_reg": 1.0} などを定義してる場合がある
        return {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}

    def diou_box_loss(self, proposal_boxes, gt_boxes, proposal_deltas, gt_classes):
        """
        Detectron2 の標準 'box_reg_loss' を置き換えて DIoU計算を行う。
        - 負例(gt_class < 0 or gt_class >= num_classes) は除外
        - 正例だけでDIoUを計算
        """
        # 1) 正例インデックスを抽出 (Detectron2標準は "gt_class >= 0 & < num_classes" で判定)
        device = proposal_deltas.device
        num_classes = proposal_deltas.shape[1] // 4
        fg_inds = torch.nonzero((gt_classes >= 0) & (gt_classes < num_classes)).squeeze(1)
        if fg_inds.numel() == 0:
            return proposal_deltas.sum() * 0.0  # ロス0

        # 2) proposal_deltas から正例分だけ抜き出し
        pred_deltas_fg = proposal_deltas[fg_inds, :]  # shape [FG, 4*C]
        gt_classes_fg = gt_classes[fg_inds]           # shape [FG]
        proposal_boxes_fg = proposal_boxes[fg_inds, :]  # shape [FG, 4]
        gt_boxes_fg = gt_boxes[fg_inds, :]

        # 3) "クラスに対応する 4次元" を選択
        #    例: if gt_class=2 => index = 4*2..4*2+3
        batch_idx = torch.arange(len(fg_inds), device=device)
        pred_deltas_for_gt_class = []
        for i in range(4):
            pred_deltas_for_gt_class.append(
                pred_deltas_fg[batch_idx, 4 * gt_classes_fg + i]
            )
        pred_deltas_for_gt_class = torch.stack(pred_deltas_for_gt_class, dim=1)  # [FG, 4]

        # 4) box2box_transform.apply_deltas で座標を復元
        pred_boxes_fg = self.box2box_transform.apply_deltas(
            pred_deltas_for_gt_class, proposal_boxes_fg
        )

        # 5) DIoU loss
        diou_vals = diou_loss(pred_boxes_fg, gt_boxes_fg)
        return diou_vals.mean()

@ROI_HEADS_REGISTRY.register()
class MyFocalROIHeads(StandardROIHeads):
    def losses(self, predictions, proposals):
        """
        predictions: (pred_class_logits, pred_proposal_deltas)
        proposals: list[Instances], 各Instancesに gt_classes, gt_boxes 等が入っている
        """
        pred_class_logits, pred_proposal_deltas = predictions
        gt_classes = torch.cat([p.gt_classes for p in proposals], dim=0)

        # 1. 分類ロス (FocalLoss)
        focal_loss_fn = FocalLoss(gamma=4.0, alpha=0.25)
        loss_cls = focal_loss_fn(pred_class_logits, gt_classes)

        # 2. IoUベースのBox回帰ロスを計算
        #    (1) Proposals から box座標と GT box座標をまとめて取得
        proposal_boxes = []
        gt_boxes_list = []
        for prop in proposals:
            proposal_boxes.append(prop.proposal_boxes.tensor)  # shape [num_props, 4]
            gt_boxes_list.append(prop.gt_boxes.tensor)         # shape [num_props, 4]
        proposal_boxes = torch.cat(proposal_boxes, dim=0)  # shape [R, 4]
        gt_boxes = torch.cat(gt_boxes_list, dim=0)         # shape [R, 4]

        #    (2) pred_proposal_deltas: shape [R, 4 * num_classes]
        #        GTクラスに対応する4次元を抜き出し、proposalと合成して予測ボックスに変換
        box_transform = self.box_predictor.box2box_transform
        # R個分のインデックス
        batch_idx = torch.arange(len(gt_classes), device=gt_classes.device)
        pred_deltas_for_gt_class = []
        for i in range(4):
            # クラスごとにオフセットを取り出す (4 * gt_class + i)
            pred_deltas_for_gt_class.append(
                pred_proposal_deltas[batch_idx, 4 * gt_classes + i]
            )
        pred_deltas_for_gt_class = torch.stack(pred_deltas_for_gt_class, dim=1)  # shape [R, 4]

        #    (3) proposal_boxes + pred_deltas_for_gt_class から予測ボックス (x1, y1, x2, y2) を取得
        pred_boxes = box_transform.apply_deltas(pred_deltas_for_gt_class, proposal_boxes)

        #    (4) IoUロス計算
        iou_loss_vals = diou_loss(pred_boxes, gt_boxes)  # shape [R]
        loss_box_reg = iou_loss_vals.mean()

        return {
            "loss_custom_cls": loss_cls*0,
            "loss_custom_box_reg": loss_box_reg
        }

@ROI_HEADS_REGISTRY.register()
class MyStandardROIHeads(StandardROIHeads):
    def __init__(self, cfg, input_shape, box_predictor=None):
        super().__init__(cfg, input_shape)

        if box_predictor is not None:
            self.box_predictor = box_predictor
        else:
            self.box_predictor = MyFastRCNNOutputLayers(
                input_shape=self.box_head.output_shape,
                box2box_transform=self.box_predictor.box2box_transform,
                num_classes=self.num_classes,
            )

