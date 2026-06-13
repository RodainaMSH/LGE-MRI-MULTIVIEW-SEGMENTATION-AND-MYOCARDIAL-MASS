"""ImageNet-pretrained encoder swap into the nnU-Net recipe — a SIZE SWEEP to MEASURE
where capacity starts overfitting our N=40 data (instead of assuming).

Each subclass = one smp encoder (ImageNet weights). nnU-Net's recipe is otherwise intact:
native-spacing resampling, Z-score, heavy aug, Dice+CE, foreground oversampling, mirror-TTA,
poly-LR/SGD, 100 epochs. Deep supervision is OFF (smp.Unet is single-output) — so compare the
encoders to nnUNetTrainerPlainNoDS_100epochs (PlainConvUNet, deep-sup also OFF) for a FAIR
encoder-only delta, and to the stock 0.733 nnU-Net (deep-sup ON) as the production reference.

smp adapts the first conv for in_channels=1 while reusing ImageNet weights, so nnU-Net's
1-channel Z-score input is preserved (better for MRI than ImageNet RGB normalization).

DEPLOY: copy to site-packages/nnunetv2/training/nnUNetTrainer/variants/network_architecture/
(re-copy after any nnunetv2 upgrade — the install tree is the only searched location).
"""
import torch
import segmentation_models_pytorch as smp
from nnunetv2.training.nnUNetTrainer.variants.training_length.nnUNetTrainer_Xepochs import nnUNetTrainer_100epochs


class _SMPEncoderTrainer(nnUNetTrainer_100epochs):
    ENCODER = "efficientnet-b0"  # overridden per subclass
    ENCODER_WEIGHTS = "imagenet"  # overridden per subclass (e.g. "noisy-student", "advprop")

    # NOTE: explicit signature required — nnUNetTrainer.__init__ introspects these param
    # names and does locals()[k]; *args/**kwargs breaks it (KeyError: 'args').
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.enable_deep_supervision = False  # smp.Unet returns a single map

    def set_deep_supervision_enabled(self, enabled: bool):
        # smp.Unet has no nnU-Net-style decoder.deep_supervision flag -> no-op (avoids AttributeError)
        return

    @classmethod
    def build_network_architecture(cls, plans_manager, configuration_manager,
                                   num_input_channels, num_output_channels,
                                   enable_deep_supervision: bool = True):
        # Ignore nnU-Net's PlainConv arch kwargs; build an ImageNet-pretrained smp U-Net instead.
        return smp.Unet(
            encoder_name=cls.ENCODER,
            encoder_weights=cls.ENCODER_WEIGHTS,
            in_channels=num_input_channels,
            classes=num_output_channels,
        )


class nnUNetTrainerSMP_effb0(_SMPEncoderTrainer):
    ENCODER = "efficientnet-b0"


class nnUNetTrainerSMP_effb3(_SMPEncoderTrainer):
    ENCODER = "efficientnet-b3"


class nnUNetTrainerSMP_effb5(_SMPEncoderTrainer):
    ENCODER = "efficientnet-b5"


class nnUNetTrainerSMP_effb7(_SMPEncoderTrainer):
    ENCODER = "efficientnet-b7"


# --- bigger-than-B7 encoders (probe whether the still-climbing size curve keeps rising) ---
class nnUNetTrainerSMP_convnextB(_SMPEncoderTrainer):
    ENCODER = "tu-convnext_base"      # ~93M


class nnUNetTrainerSMP_effv2l(_SMPEncoderTrainer):
    ENCODER = "tu-tf_efficientnetv2_l"  # ~120M


class nnUNetTrainerSMP_convnextL(_SMPEncoderTrainer):
    ENCODER = "tu-convnext_large"     # ~203M (may OOM at batch 17; orchestrator continues on failure)


class nnUNetTrainerPlainNoDS_100epochs(nnUNetTrainer_100epochs):
    """Control: stock PlainConvUNet but with deep supervision OFF, so the smp encoders above are
    compared like-for-like (encoder effect isolated from the deep-supervision ingredient)."""
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.enable_deep_supervision = False


# ===== FAMILY SWEEP (2026-06-08): beat EffNetV2-L 0.7587 by trying other families / weights =====
# EfficientNet better weights (same winning family):
class nnUNetTrainerSMP_effb7ns(_SMPEncoderTrainer):
    ENCODER = "timm-efficientnet-b7"; ENCODER_WEIGHTS = "noisy-student"   # JFT-distilled, often best transfer


class nnUNetTrainerSMP_effv2l21k(_SMPEncoderTrainer):
    ENCODER = "tu-tf_efficientnetv2_l_in21k"; ENCODER_WEIGHTS = "imagenet"  # ImageNet-21k pretrain of our winner


class nnUNetTrainerSMP_effv2m(_SMPEncoderTrainer):
    ENCODER = "tu-tf_efficientnetv2_m"; ENCODER_WEIGHTS = "imagenet"      # ~54M, smaller V2 (less overfit at N=32?)


class nnUNetTrainerSMP_effv2xl(_SMPEncoderTrainer):
    ENCODER = "tu-tf_efficientnetv2_xl"; ENCODER_WEIGHTS = "imagenet"     # ~209M (may OOM)


class nnUNetTrainerSMP_effv2xl_RASwarm(_SMPEncoderTrainer):
    """RAS specialist WARM-STARTED from the cardiac-LGE generalist (ImageNet -> cardiac-LGE -> RAS).
    Builds the same EffNetV2-XL smp.Unet, then loads the trained generalist
    (nnUNetTrainerSMP_effv2xl on Dataset000) encoder+decoder via shape-matched strict=False.
    The generalist's input stem is ALREADY 1-channel (grayscale LGE) so it transfers as-is;
    only the final 2-class segmentation head is fresh. Set RASWARM_CKPT to pick the source fold."""
    ENCODER = "tu-tf_efficientnetv2_xl"; ENCODER_WEIGHTS = None

    @classmethod
    def build_network_architecture(cls, plans_manager, configuration_manager,
                                   num_input_channels, num_output_channels,
                                   enable_deep_supervision: bool = True):
        import os
        net = smp.Unet(encoder_name=cls.ENCODER, encoder_weights=None,
                       in_channels=num_input_channels, classes=num_output_channels)
        ckpt = os.environ.get("RASWARM_CKPT",
            "/home/youssef/projects/research/CMR-MULTI/nnunet/nnUNet_results/"
            "Dataset000_LGEgeneralist/nnUNetTrainerSMP_effv2xl__nnUNetPlans__2d/fold_0/checkpoint_final.pth")
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)["network_weights"]
        msd = net.state_dict()
        keep = {k: v for k, v in sd.items() if k in msd and v.shape == msd[k].shape}
        dropped = [k for k in sd if k not in keep]
        net.load_state_dict(keep, strict=False)
        print(f"[RASwarm] warm-start from generalist {ckpt}")
        print(f"[RASwarm] transferred {len(keep)}/{len(sd)} tensors; FRESH (shape-mismatch/dropped): {dropped}")
        return net


class nnUNetTrainerSMP_effv2xl21k(_SMPEncoderTrainer):
    # 2026-06-08: highest-prior same-family extension of the V2-L win (0.7587). ImageNet-21k pretrain
    # of EfficientNetV2-XL (~209M, build-verified). Pure smp, no unofficial weights. ConvNeXt-large
    # (203M) trained fine at batch 17/patch 224, so this should fit 24GB.
    ENCODER = "tu-tf_efficientnetv2_xl_in21k"; ENCODER_WEIGHTS = "imagenet"


# Best BatchNorm CNN families (share EffNet's favorable BN/ImageNet transfer):
class nnUNetTrainerSMP_regnety120(_SMPEncoderTrainer):
    ENCODER = "tu-regnety_120"; ENCODER_WEIGHTS = "imagenet"


class nnUNetTrainerSMP_seresnext101(_SMPEncoderTrainer):
    ENCODER = "se_resnext101_32x4d"; ENCODER_WEIGHTS = "imagenet"


class nnUNetTrainerSMP_resnest200(_SMPEncoderTrainer):
    ENCODER = "timm-resnest200e"; ENCODER_WEIGHTS = "imagenet"


# Transformer contrast (SegFormer MiT backbone):
class nnUNetTrainerSMP_mitb4(_SMPEncoderTrainer):
    ENCODER = "mit_b4"; ENCODER_WEIGHTS = "imagenet"


# ImageNet CONTROLS for the RadImageNet comparison (same arch, ImageNet pretrain) — isolates the
# "medical pretraining" effect from the "ResNet/DenseNet architecture" effect.
class nnUNetTrainerSMP_resnet50(_SMPEncoderTrainer):
    ENCODER = "resnet50"; ENCODER_WEIGHTS = "imagenet"


class nnUNetTrainerSMP_densenet121(_SMPEncoderTrainer):
    ENCODER = "densenet121"; ENCODER_WEIGHTS = "imagenet"   # ~8M


class nnUNetTrainerSMP_densenet169(_SMPEncoderTrainer):
    ENCODER = "densenet169"; ENCODER_WEIGHTS = "imagenet"   # ~14M


class nnUNetTrainerSMP_densenet201(_SMPEncoderTrainer):
    ENCODER = "densenet201"; ENCODER_WEIGHTS = "imagenet"   # ~20M


class nnUNetTrainerSMP_densenet161(_SMPEncoderTrainer):
    ENCODER = "densenet161"; ENCODER_WEIGHTS = "imagenet"   # ~28M (widest)


# ===== RadImageNet (medical-pretrained, official BMEII-AI PyTorch weights, MIT) =====
# Pretrained on 1.35M radiology images (CT/MR/US). Official weights = an nn.Sequential-wrapped
# torchvision backbone; we remap to smp encoder key names + fold the RGB conv to 1-channel (sum),
# matching smp's in_channels=1 reduction. Verified: 318/318 (resnet50) & 725/725 (densenet121)
# tensors map with 0 missing / 0 unexpected / 0 shape-mismatch.
import os

# Absolute repo path (this trainer is deployed into site-packages, so __file__-relative paths break).
# Override with env var RIN_DIR if the repo moves.
_RIN_DIR = os.environ.get(
    "RIN_DIR",
    "/home/youssef/projects/research/CMR-MULTI/weights/radimagenet/RadImageNet_pytorch")


def _remap_radimagenet(arch: str, raw: dict, target_sd: dict) -> dict:
    if arch == "resnet50":
        idxmap = {"0": "conv1", "1": "bn1", "4": "layer1", "5": "layer2", "6": "layer3", "7": "layer4"}
        out = {}
        for k, v in raw.items():
            rest = k[len("backbone."):]
            head, tail = rest.split(".", 1)
            if head not in idxmap:   # relu/maxpool carry no params
                continue
            out[idxmap[head] + "." + tail] = v
        first = "conv1.weight"
    elif arch == "densenet121":
        out = {"features." + k[len("backbone.0."):]: v for k, v in raw.items()}
        first = "features.conv0.weight"
    else:
        raise ValueError(arch)
    # fold RGB(3) first conv -> 1 channel to match smp in_channels=1 (sum across input channels)
    if out[first].shape[1] == 3 and target_sd[first].shape[1] == 1:
        out[first] = out[first].sum(dim=1, keepdim=True)
    return out


class _RadImageNetTrainer(_SMPEncoderTrainer):
    ENCODER = "resnet50"           # smp encoder_name (built with weights=None, then RIN loaded)
    RIN_ARCH = "resnet50"          # which remap to use
    RIN_FILE = "ResNet50.pt"

    @classmethod
    def build_network_architecture(cls, plans_manager, configuration_manager,
                                   num_input_channels, num_output_channels,
                                   enable_deep_supervision: bool = True):
        net = smp.Unet(encoder_name=cls.ENCODER, encoder_weights=None,
                       in_channels=num_input_channels, classes=num_output_channels)
        raw = torch.load(os.path.join(_RIN_DIR, cls.RIN_FILE), map_location="cpu", weights_only=False)
        tgt = net.encoder.state_dict()
        remap = _remap_radimagenet(cls.RIN_ARCH, raw, tgt)
        missing = [k for k in tgt if k not in remap]
        assert not missing, f"RadImageNet load incomplete, missing {len(missing)}: {missing[:5]}"
        # smp's ResNet/DenseNet encoder subclasses override load_state_dict and drop the `strict`
        # kwarg — so call without it. Completeness is already guaranteed by the assert above.
        probe_key = "conv1.weight" if cls.RIN_ARCH == "resnet50" else "features.conv0.weight"
        net.encoder.load_state_dict(remap)
        loaded = net.encoder.state_dict()[probe_key]
        assert torch.equal(loaded, remap[probe_key]), "RadImageNet weights did not land (value mismatch)"
        print(f"[RadImageNet] loaded {len(remap)} tensors into smp {cls.ENCODER} encoder "
              f"(in_ch={num_input_channels}) from {cls.RIN_FILE} [verified]")
        return net


class nnUNetTrainerRIN_resnet50(_RadImageNetTrainer):
    ENCODER = "resnet50"; RIN_ARCH = "resnet50"; RIN_FILE = "ResNet50.pt"


class nnUNetTrainerRIN_densenet121(_RadImageNetTrainer):
    ENCODER = "densenet121"; RIN_ARCH = "densenet121"; RIN_FILE = "DenseNet121.pt"


# ===== MedSAM2 (SAM2.1 Hiera-tiny) MEDICAL-pretrained transformer encoder in the nnU-Net recipe =====
# Evidence: the challenge organizers' own paper (Qu et al. 2026), Table D5 SAX-LGE — MedSAM2 got the
# BEST scar/LGE DSC (65.37), beating their model (60.83) and nnU-Net (60.19). We load the MedSAM2 trunk
# (SAM2.1 Hiera-tiny, depths [1,2,7,2], dims 96-192-384-768) into a timm hiera_tiny smp U-Net.
# Verified surgery: 152/153 encoder tensors load (timm `model.model.<x>` -> SAM2 `<x>`; mlp.fc1/fc2 ->
# mlp.layers.0/1; patch_embed RGB->1ch summed); only `pos_embed` stays at init (input is fixed 224, so
# the pretrained positional grid would match, but SAM2's global+window scheme differs in storage; it
# fine-tunes). NOTE: transformer => fixed 224 input (matches nnU-Net patch 224). Override env MEDSAM2_CKPT.
_MEDSAM2_CKPT = os.environ.get(
    "MEDSAM2_CKPT",
    "/home/youssef/projects/research/CMR-MULTI/specialists/MedSAM2/checkpoints/MedSAM2_latest.pt")


class nnUNetTrainerHieraMedSAM2(_SMPEncoderTrainer):
    ENCODER = "tu-hiera_tiny_224"

    @classmethod
    def build_network_architecture(cls, plans_manager, configuration_manager,
                                   num_input_channels, num_output_channels,
                                   enable_deep_supervision: bool = True):
        net = smp.Unet(encoder_name=cls.ENCODER, encoder_weights=None,
                       in_channels=num_input_channels, classes=num_output_channels)
        ck = torch.load(_MEDSAM2_CKPT, map_location="cpu", weights_only=False)
        sd = ck.get("model", ck) if isinstance(ck, dict) else ck
        trunk = {k.replace("image_encoder.trunk.", ""): v
                 for k, v in sd.items() if k.startswith("image_encoder.trunk.")}
        tgt = net.encoder.state_dict()

        def src(short):
            return short.replace("mlp.fc1", "mlp.layers.0").replace("mlp.fc2", "mlp.layers.1")

        remap = {}
        for tk in tgt:
            short = src(tk.replace("model.model.", ""))
            if short in trunk and trunk[short].shape == tgt[tk].shape:
                remap[tk] = trunk[short]
            elif (short == "patch_embed.proj.weight" and short in trunk
                  and trunk[short].shape[1] == 3 and tgt[tk].shape[1] == 1):
                remap[tk] = trunk[short].sum(1, keepdim=True)
        assert len(remap) >= 150, f"MedSAM2-Hiera load too sparse: {len(remap)}/{len(tgt)}"
        net.encoder.load_state_dict(remap, strict=False)
        print(f"[MedSAM2-Hiera] loaded {len(remap)}/{len(tgt)} MEDICAL trunk tensors into "
              f"tu-hiera_tiny (in_ch={num_input_channels}); pos_embed left at init")
        return net


# ---- SAM2.1 / Hiera-LARGE (214M) as a U-Net encoder in the nnU-Net recipe ----
# timm ships these natively (no state-dict surgery): the SMP tu-encoder builds the SAME timm
# model, so we just load the explicitly-tagged pretrained weights into net.encoder.model
# (verified: 586/586 tensors, 0 missing/unexpected). in_chans=1 -> timm folds RGB->1ch.
class _TimmPretrainedLargeTrainer(_SMPEncoderTrainer):
    ENCODER = None        # e.g. "tu-sam2_hiera_large"
    TIMM_NAME = None      # e.g. "sam2_hiera_large.fb_r1024_2pt1"
    TAG = "?"

    @classmethod
    def build_network_architecture(cls, plans_manager, configuration_manager,
                                   num_input_channels, num_output_channels,
                                   enable_deep_supervision: bool = True):
        import timm
        net = smp.Unet(encoder_name=cls.ENCODER, encoder_weights=None,
                       in_channels=num_input_channels, classes=num_output_channels)
        pre = timm.create_model(cls.TIMM_NAME, pretrained=True, features_only=True,
                                in_chans=num_input_channels)
        r = net.encoder.model.load_state_dict(pre.state_dict(), strict=False)
        n = len(net.encoder.model.state_dict())
        assert not r.missing_keys and not r.unexpected_keys, \
            f"{cls.TIMM_NAME} load mismatch: miss={len(r.missing_keys)} unexp={len(r.unexpected_keys)}"
        print(f"[{cls.TAG}] loaded {n}/{n} pretrained tensors into {cls.ENCODER} "
              f"(in_ch={num_input_channels})")
        return net


class nnUNetTrainerSAM2large(_TimmPretrainedLargeTrainer):
    # SAM2.1 Hiera-large (video/natural-image promptable-seg pretrain), 214M trunk
    ENCODER = "tu-sam2_hiera_large"
    TIMM_NAME = "sam2_hiera_large.fb_r1024_2pt1"
    TAG = "SAM2.1-large"


class nnUNetTrainerHieraLargeMAE(_TimmPretrainedLargeTrainer):
    # Hiera-large MAE-pretrained on ImageNet, NATIVE 224 (clean res match) -- domain control vs SAM2.1
    ENCODER = "tu-hiera_large_224"
    TIMM_NAME = "hiera_large_224.mae_in1k_ft_in1k"
    TAG = "Hiera-large-MAE"


# ===== DenseNet-264 (PaddleClas ImageNet-1k) — the LARGEST pretrained densenet that exists (2026-06-09) =====
# Capacity test (Youssef: "8M too small"). DenseNet-264 is NOT a native smp/timm encoder, so we (a) register
# a densenet264 entry in smp's encoder registry, and (b) load the PaddleClas ImageNet-1k weights converted to
# torchvision layout (nnunet/convert_densenet264.py; Samoyed top-1 0.768 verified). SAME nnU-Net recipe as
# every other _SMPEncoderTrainer (smp U-Net, SGD/poly-LR, 100 ep, deep-sup off). RGB stem summed -> 1ch.
from segmentation_models_pytorch.encoders import encoders as _SMP_ENC
from segmentation_models_pytorch.encoders.densenet import DenseNetEncoder as _DN_ENC
if "densenet264" not in _SMP_ENC:
    _SMP_ENC["densenet264"] = {
        "encoder": _DN_ENC, "pretrained_settings": {},
        "params": {"out_channels": [3, 64, 256, 512, 2304, 2688], "num_init_features": 64,
                   "growth_rate": 32, "block_config": (6, 12, 64, 48)},
    }
_DN264_CKPT = os.environ.get(
    "DN264_CKPT",
    "/home/youssef/projects/research/CMR-MULTI/weights/densenet264/densenet264_in1k_torch.pth")


class nnUNetTrainerSMP_densenet264(_SMPEncoderTrainer):
    ENCODER = "densenet264"; ENCODER_WEIGHTS = None  # built weightless; ImageNet-1k loaded manually below

    @classmethod
    def build_network_architecture(cls, plans_manager, configuration_manager,
                                   num_input_channels, num_output_channels,
                                   enable_deep_supervision: bool = True):
        net = smp.Unet(encoder_name="densenet264", encoder_weights=None,
                       in_channels=num_input_channels, classes=num_output_channels)
        sd = dict(torch.load(_DN264_CKPT, map_location="cpu", weights_only=True))
        if num_input_channels == 1:
            sd["features.conv0.weight"] = sd["features.conv0.weight"].sum(1, keepdim=True)  # RGB->1ch (smp's rule)
        elif num_input_channels != 3:
            sd["features.conv0.weight"] = sd["features.conv0.weight"][:, :num_input_channels]
        net.encoder.load_state_dict(sd)  # smp DenseNetEncoder pops classifier + fixes key names
        probe = net.encoder.state_dict()["features.conv0.weight"]
        assert torch.equal(probe, sd["features.conv0.weight"]), "densenet264 weights did not land"
        print(f"[DenseNet264] ImageNet-1k loaded into smp densenet264 encoder "
              f"(in_ch={num_input_channels}, enc {sum(p.numel() for p in net.encoder.parameters())/1e6:.1f}M) [verified]")
        return net


# ============================================================================
# DECODER STUDY (2026-06-10) — vary the DECODER/arch, hold encoder + recipe constant.
# We only ever swapped the ENCODER; the smp U-Net decoder (3.4M w/ XL) was always the same,
# trained-from-scratch (no pretraining transfers to a decoder). Testbed encoder = EfficientNet-B7
# (baseline = nnUNetTrainerSMP_effb7 / plain smp.Unet: ① OOF DSC 0.7365 / scar 0.3995; ② 0.7464 / 0.4768).
# Same _SMPEncoderTrainer recipe (100ep, SGD+poly-LR, deep-sup off). GATE: 5-fold OOF (our+their),
# beat the B7-Unet baseline scar/DSC by >+0.01 (clear fold-noise); watch overfit (decoder is from-scratch @ N=40).
# Winner decoder -> graft onto EffNetV2-XL (best encoder) + 5-fold to confirm the real model.
# ============================================================================
class _SMPVariantTrainer(_SMPEncoderTrainer):
    """Decoder/arch variant: same recipe + encoder, swap the smp architecture/decoder."""
    SMP_MODEL = smp.Unet      # override: smp.UnetPlusPlus / smp.MAnet / smp.FPN / smp.DeepLabV3Plus
    DECODER_KW = {}           # override: {"decoder_attention_type":"scse"} / {"decoder_channels":(...)}
    ENCODER = "efficientnet-b7"; ENCODER_WEIGHTS = "imagenet"

    @classmethod
    def build_network_architecture(cls, plans_manager, configuration_manager,
                                   num_input_channels, num_output_channels,
                                   enable_deep_supervision: bool = True):
        net = cls.SMP_MODEL(encoder_name=cls.ENCODER, encoder_weights=cls.ENCODER_WEIGHTS,
                            in_channels=num_input_channels, classes=num_output_channels, **cls.DECODER_KW)
        dec = sum(p.numel() for p in net.decoder.parameters()) / 1e6
        print(f"[DecoderStudy] {cls.__name__}: {cls.SMP_MODEL.__name__} enc={cls.ENCODER} "
              f"decoder={dec:.1f}M kw={cls.DECODER_KW}")
        return net

# --- B7 testbed: decoder/arch variants (compare vs nnUNetTrainerSMP_effb7 baseline) ---
class nnUNetTrainerSMP_b7_scse(_SMPVariantTrainer):
    DECODER_KW = {"decoder_attention_type": "scse"}                          # U-Net + squeeze-&-excite attention
class nnUNetTrainerSMP_b7_uxx(_SMPVariantTrainer):
    SMP_MODEL = smp.UnetPlusPlus                                             # nested dense skips (best for focal scar)
class nnUNetTrainerSMP_b7_uxxscse(_SMPVariantTrainer):
    SMP_MODEL = smp.UnetPlusPlus; DECODER_KW = {"decoder_attention_type": "scse"}
class nnUNetTrainerSMP_b7_manet(_SMPVariantTrainer):
    SMP_MODEL = smp.MAnet                                                    # multi-scale attention decoder
class nnUNetTrainerSMP_b7_wide(_SMPVariantTrainer):
    DECODER_KW = {"decoder_channels": (512, 256, 128, 64, 32)}              # wider decoder (capacity; overfit risk)
class nnUNetTrainerSMP_b7_fpn(_SMPVariantTrainer):
    SMP_MODEL = smp.FPN                                                      # multi-scale feature pyramid (1.9M, light, diff paradigm)
class nnUNetTrainerSMP_b7_dlv3p(_SMPVariantTrainer):
    SMP_MODEL = smp.DeepLabV3Plus                                            # atrous spatial pyramid pooling (1.3M, light, diff paradigm)

# --- XL grafts: run the WINNING decoder on the best encoder once known (EffNetV2-XL) ---
class nnUNetTrainerSMP_xl_uxx(_SMPVariantTrainer):
    ENCODER = "tu-tf_efficientnetv2_xl"; SMP_MODEL = smp.UnetPlusPlus
class nnUNetTrainerSMP_xl_scse(_SMPVariantTrainer):
    ENCODER = "tu-tf_efficientnetv2_xl"; DECODER_KW = {"decoder_attention_type": "scse"}


# ============================================================================
# DenseNet-264 DECODER GRAFTS (2026-06-10) — Youssef's call: run the WINNING decoder's 5-fold
# directly on DenseNet-264 (our ② Task2-leader encoder), skipping the B7 5-fold. Same ImageNet-1k
# weight-loading as nnUNetTrainerSMP_densenet264 (encoder-only; decoder from scratch). NOTE decoders
# are MUCH bigger here (DN channels wide): Unet dec 13.4M, UnetPlusPlus 35.8M, MAnet 216.9M, wide 29.2M.
# DeepLabV3+ is UNAVAILABLE on densenet (smp: no dilated mode).
# ============================================================================
class _DN264VariantTrainer(nnUNetTrainerSMP_densenet264):
    """DenseNet-264 encoder (ImageNet-1k) + a chosen decoder/arch. Encoder weights load exactly as the
    base densenet264 trainer; only SMP_MODEL/DECODER_KW change the decoder (trained from scratch)."""
    SMP_MODEL = smp.Unet
    DECODER_KW = {}

    @classmethod
    def build_network_architecture(cls, plans_manager, configuration_manager,
                                   num_input_channels, num_output_channels,
                                   enable_deep_supervision: bool = True):
        net = cls.SMP_MODEL(encoder_name="densenet264", encoder_weights=None,
                            in_channels=num_input_channels, classes=num_output_channels, **cls.DECODER_KW)
        sd = dict(torch.load(_DN264_CKPT, map_location="cpu", weights_only=True))
        if num_input_channels == 1:
            sd["features.conv0.weight"] = sd["features.conv0.weight"].sum(1, keepdim=True)  # RGB->1ch
        elif num_input_channels != 3:
            sd["features.conv0.weight"] = sd["features.conv0.weight"][:, :num_input_channels]
        net.encoder.load_state_dict(sd)  # encoder-only; decoder stays from-scratch
        probe = net.encoder.state_dict()["features.conv0.weight"]
        assert torch.equal(probe, sd["features.conv0.weight"]), "densenet264 weights did not land"
        dec = sum(p.numel() for p in net.decoder.parameters()) / 1e6
        print(f"[DN264-Decoder] {cls.__name__}: {cls.SMP_MODEL.__name__} dec={dec:.1f}M "
              f"kw={cls.DECODER_KW} [enc imagenet-1k loaded, verified]")
        return net

class nnUNetTrainerSMP_dn264_uxxscse(_DN264VariantTrainer):
    SMP_MODEL = smp.UnetPlusPlus; DECODER_KW = {"decoder_attention_type": "scse"}   # best B7 paired tilt
class nnUNetTrainerSMP_dn264_uxx(_DN264VariantTrainer):
    SMP_MODEL = smp.UnetPlusPlus
class nnUNetTrainerSMP_dn264_scse(_DN264VariantTrainer):
    DECODER_KW = {"decoder_attention_type": "scse"}
class nnUNetTrainerSMP_dn264_manet(_DN264VariantTrainer):
    SMP_MODEL = smp.MAnet                                                            # 216.9M decoder (capacity)
class nnUNetTrainerSMP_dn264_wide(_DN264VariantTrainer):
    DECODER_KW = {"decoder_channels": (512, 256, 128, 64, 32)}
