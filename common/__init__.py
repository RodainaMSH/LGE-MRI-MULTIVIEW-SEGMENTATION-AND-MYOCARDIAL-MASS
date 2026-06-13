"""Shared library used identically by all three tracks (unet_vanilla,
unet_efficientnet, nnunet).

Keeping the metrics code in ONE place is what makes the three-track comparison
provably fair: every track imports the exact same functions rather than each
carrying its own copy.

Modules:
    common.metrics    — Dice / vPCC / RAE / patient mass (pure functions)
    common.seg_export — native-volume prediction + mask saving (predict_native_volume, save_mask)
"""
