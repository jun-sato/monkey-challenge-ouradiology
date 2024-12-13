import matplotlib.patches as pltpatches
import matplotlib.pyplot as plt

def plot_boxes(
    boxes, max_width, max_height, axes=None, output_shape=None, color_map=plt.cm.prism
):
    if axes is None:
        _, ax = plt.subplots(1, 1)
    else:
        ax = axes

    for box in boxes:
        x1, y1, x2, y2, label_value, confidence = box
        color = color_map(int(label_value))

        if (x1, y1, x2, y2) != (0, 0, 0, 0):
            rect = pltpatches.Rectangle(
                (x1, y1),
                min(max_width, max(0, x2 - x1)),
                min(max_height, max(0, y2 - y1)),
                linewidth=2,
                edgecolor=color,
                facecolor="none",
            )
            ax.add_patch(rect)

    if axes is None:
        plt.show()


import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss for classification
    gamma: 焦点パラメータ
    alpha: クラス不均衡を調整するための重み（floatまたはTensor）
    """
    def __init__(self, gamma=2.0, alpha=0.25, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: [N, C] のlogits（クラススコア）
        # targets: [N] の正解クラスインデックス
        log_probs = F.log_softmax(inputs, dim=1)
        probs = torch.exp(log_probs)
        
        # gatherによってtargetsに応じたクラスの確率を取り出す
        gather_log_probs = log_probs.gather(1, targets.unsqueeze(1))
        gather_probs = probs.gather(1, targets.unsqueeze(1))
        
        # focal loss 計算
        focal_weight = (1 - gather_probs) ** self.gamma
        if isinstance(self.alpha, float):
            alpha_factor = self.alpha if targets.dim() == 0 else torch.ones_like(targets, dtype=log_probs.dtype) * self.alpha
            alpha_factor = alpha_factor.unsqueeze(1)
        else:
            # クラスごとにalpha指定する場合はここで処理
            alpha_factor = self.alpha[targets].unsqueeze(1)
        
        loss = -alpha_factor * focal_weight * gather_log_probs

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
