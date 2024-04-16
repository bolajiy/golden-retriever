import torch


class MaskedXentMarginLoss:
    def __init__(self, pad_value=-1000, zero_weight=1., one_weight=1., margin=0.7, eps=1e-7, **kwargs):
        self.pad_value = pad_value
        self.margin = margin
        self.zero_weight = zero_weight
        self.one_weight = one_weight
        self.eps = eps

    def __call__(self, pred, labels, sample_weights=None):
        pred = torch.clamp(pred, min=self.eps, max=1 - self.eps)
        loss_mat = -self.one_weight * labels * torch.log(pred) * (pred <= self.margin).float()
        loss_mat -= self.zero_weight * (1 - labels) * torch.log(1-pred) * (pred >= 1 - self.margin).float()
        if sample_weights is not None:
           loss_mat = loss_mat * sample_weights
        mask = ((labels == 0) + (labels == 1))
        loss = loss_mat[mask].sum()  # / mask.sum()
        return loss


_losses = {'MaskedXentMarginLoss': MaskedXentMarginLoss,
           }


def get_loss(loss_name):
    return _losses[loss_name]
