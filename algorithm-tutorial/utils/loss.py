import torch
import torch.nn.functional as F
from detectron2.modeling.roi_heads import StandardROIHeads, ROI_HEADS_REGISTRY
from utils.vis import FocalLoss  # もとのFocalLossの定義位置に応じてパスを調整

def pairwise_iou(boxes1, boxes2):
    # boxes1, boxes2: shape [N,4], [M,4]
    # return: IoU matrix of shape [N,M]
    area1 = (boxes1[:,2]-boxes1[:,0])*(boxes1[:,3]-boxes1[:,1])
    area2 = (boxes2[:,2]-boxes2[:,0])*(boxes2[:,3]-boxes2[:,1])

    lt_x = torch.max(boxes1[:,None,0], boxes2[:,0])  # [N,M]
    lt_y = torch.max(boxes1[:,None,1], boxes2[:,1])
    rb_x = torch.min(boxes1[:,None,2], boxes2[:,2])
    rb_y = torch.min(boxes1[:,None,3], boxes2[:,3])

    inter_w = (rb_x - lt_x).clamp(min=0)
    inter_h = (rb_y - lt_y).clamp(min=0)
    inter = inter_w * inter_h

    iou = inter / (area1[:,None] + area2 - inter)
    return iou

@ROI_HEADS_REGISTRY.register()
class MyFocalROIHeads(StandardROIHeads):
    def losses(self, predictions, proposals):
        pred_class_logits, pred_proposal_deltas = predictions
        gt_classes = torch.cat([p.gt_classes for p in proposals], dim=0)
        
        focal_loss_fn = FocalLoss(gamma=2.0, alpha=0.25)
        loss_cls = focal_loss_fn(pred_class_logits, gt_classes)

        box_transform = self.box_predictor.box2box_transform
        loss_box_reg = self.box_predictor.losses((None, pred_proposal_deltas), proposals, box_transform)['loss_box_reg']

        scores = F.softmax(pred_class_logits, dim=1) 
        foreground_scores = scores[:,1:]  # lymphocyte(0), monocyte(1)
        max_scores, pred_cls = foreground_scores.max(dim=1)

        proposal_boxes = torch.cat([p.proposal_boxes.tensor for p in proposals], dim=0)
        pred_deltas_all = pred_proposal_deltas.view(-1, (1 + self.num_classes)*4)
        
        batch_idx = torch.arange(pred_cls.shape[0], device=pred_cls.device)
        class_delta_indices = pred_cls + 1
        pred_deltas_selected = torch.stack([
            pred_deltas_all[batch_idx, class_delta_indices*4 + i] for i in range(4)
        ], dim=1)

        pred_boxes = self.box_predictor.box2box_transform.apply_deltas(pred_deltas_selected, proposal_boxes)

        gt_boxes = torch.cat([p.gt_boxes.tensor for p in proposals], dim=0)
        G = gt_boxes.shape[0]

        if G == 0:
            loss_froc = torch.tensor(0.0, device=gt_boxes.device)
        else:
            iou_matrix = pairwise_iou(pred_boxes, gt_boxes)
            class_match_mask = (pred_cls[:, None] == gt_classes[None, :])
            iou_matrix = iou_matrix * class_match_mask.float()

            max_iou_vals, _ = iou_matrix.max(dim=1)
            temp = 0.1
            tp_indicator = torch.sigmoid((max_iou_vals - 0.5)/temp)
            fp_indicator = 1 - tp_indicator

            thresholds = torch.tensor([0.1,0.3,0.5,0.7,0.9], device=scores.device)
            froc_loss_all = []

            for t in thresholds:
                detect_prob = torch.sigmoid((max_scores - t)/temp)
                expected_TP = (detect_prob * tp_indicator).sum()
                expected_FP = (detect_prob * fp_indicator).sum()

                sensitivity = expected_TP / G
                fp_weight = torch.sigmoid((2 - expected_FP)/0.5)

                froc_loss_t = -sensitivity * fp_weight
                froc_loss_all.append(froc_loss_t)

            loss_froc = torch.mean(torch.stack(froc_loss_all)) if len(froc_loss_all)>0 else 0.0

        total_loss = loss_cls + loss_box_reg + 0.5 * loss_froc
        return {
            "loss_cls": loss_cls,
            "loss_box_reg": loss_box_reg,
            "loss_froc": loss_froc,
            "total_loss": total_loss
        }



# @ROI_HEADS_REGISTRY.register()
# class MyFocalROIHeads(StandardROIHeads):
#     def losses(self, predictions, proposals):
#         pred_class_logits, pred_proposal_deltas = predictions
#         gt_classes = torch.cat([p.gt_classes for p in proposals], dim=0)

#         focal_loss_fn = FocalLoss(gamma=2.0, alpha=0.25)
#         loss_cls = focal_loss_fn(pred_class_logits, gt_classes)

#         # ボックス回帰損失はデフォルトのメソッドを使用する
#         box_transform = self.box_predictor.box2box_transform
#         loss_box_reg = self.box_predictor.losses((None, pred_proposal_deltas), proposals, box_transform)['loss_box_reg']

#         return {
#             "loss_cls": loss_cls,
#             "loss_box_reg": loss_box_reg
#         }
        
        
