"""Scar-targeted Tversky+CE nnU-Net trainer (recall-favoring), 100 epochs.

WHY: OOF scar is false-negative-dominated (recall ~0.49). Tversky with beta>alpha
up-weights false negatives in the gradient -> trades precision for recall on the rare
scar class, while KEEPING nnU-Net's full recipe (heavy aug, deep supervision, full-res
224x224, batch_dice, oversampling). That recipe is what generalized when bespoke ScarNet
overfit, so we change ONLY the region-loss term.

First run = PLAIN Tversky (gamma=1.0, no focal modulation) per the verified recommendation
(focal can be unstable on a tiny class at N=40). alpha=0.3 (FP weight), beta=0.7 (FN weight).

DEPLOY: this file is discovered ONLY when copied to
  site-packages/nnunetv2/training/nnUNetTrainer/variants/loss/nnUNetTrainerScarTversky.py
A nnunetv2 reinstall/upgrade wipes it — re-copy this repo file after any upgrade.
"""
import numpy as np
import torch
from torch import nn

from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from nnunetv2.training.nnUNetTrainer.variants.training_length.nnUNetTrainer_Xepochs import nnUNetTrainer_100epochs
from nnunetv2.utilities.ddp_allgather import AllGatherGrad
from nnunetv2.utilities.helpers import softmax_helper_dim1


class MemoryEfficientFocalTverskyLoss(nn.Module):
    """Tversky index (Salehi 2017) with optional focal modulation (Abraham 2018).

    NOTE on symbol convention: here `alpha` multiplies FP and `beta` multiplies FN
    (the OPPOSITE of the paper's alpha=FN labelling). With alpha=0.3 < beta=0.7 the
    behaviour is recall-favoring (FN penalized harder). Do not "fix" the names.
    Mirrors MemoryEfficientSoftDiceLoss plumbing (do_bg, batch_dice, ddp, smooth).
    """
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=False, smooth=1e-5,
                 ddp=True, alpha=0.3, beta=0.7, gamma=1.0):
        super().__init__()
        self.apply_nonlin = apply_nonlin
        self.batch_dice = batch_dice
        self.do_bg = do_bg
        self.smooth = smooth
        self.ddp = ddp
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, x, y, loss_mask=None):
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)
        axes = tuple(range(2, x.ndim))
        tp, fp, fn, _ = get_tp_fp_fn_tn(x, y, axes, loss_mask, False)
        if not self.do_bg:
            tp, fp, fn = tp[:, 1:], fp[:, 1:], fn[:, 1:]
        if self.batch_dice:
            if self.ddp:
                tp = AllGatherGrad.apply(tp).sum(0)
                fp = AllGatherGrad.apply(fp).sum(0)
                fn = AllGatherGrad.apply(fn).sum(0)
            tp, fp, fn = tp.sum(0), fp.sum(0), fn.sum(0)
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth).clamp_min(1e-8)
        # gamma=1.0 -> plain Tversky loss (1 - TI); gamma>1 amplifies hard (low-TI) classes
        ft = torch.pow((1.0 - tversky).clamp_min(0.0), 1.0 / self.gamma)
        return ft.mean()


class DC_FT_and_CE_loss(nn.Module):
    """Compound = weight_ce*CE + weight_dice*FocalTversky (drop-in for DC_and_CE_loss)."""
    def __init__(self, tversky_kwargs, ce_kwargs, weight_ce=1.0, weight_dice=1.0, ignore_label=None):
        super().__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice
        self.ignore_label = ignore_label
        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = MemoryEfficientFocalTverskyLoss(apply_nonlin=softmax_helper_dim1, **tversky_kwargs)

    def forward(self, net_output, target):
        if self.ignore_label is not None:
            assert target.shape[1] == 1
            mask = target != self.ignore_label
            target_dice = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None
        dc = self.dc(net_output, target_dice, loss_mask=mask) if self.weight_dice != 0 else 0
        ce = self.ce(net_output, target[:, 0]) if (self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0)) else 0
        return self.weight_ce * ce + self.weight_dice * dc


class nnUNetTrainerScarTversky(nnUNetTrainer_100epochs):
    """Plain Tversky (alpha0.3 beta0.7 gamma1.0) + CE, deep-supervision wrapped.
    Recipe otherwise identical to the 100-epoch baseline."""
    def _build_loss(self):
        assert not self.label_manager.has_regions, 'regions not supported by this trainer'
        loss = DC_FT_and_CE_loss(
            {'batch_dice': self.configuration_manager.batch_dice, 'smooth': 1e-5, 'do_bg': False,
             'ddp': self.is_ddp, 'alpha': 0.3, 'beta': 0.7, 'gamma': 1.0},
            {}, weight_ce=1.0, weight_dice=1.0, ignore_label=self.label_manager.ignore_label)
        if self.enable_deep_supervision:
            dss = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(dss))])
            weights[-1] = 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)
        return loss
