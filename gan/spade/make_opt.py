"""Builds the SPADE `opt` namespace EXACTLY as Sina Amirrajab's CMRISynthSeg used it for LGE.

Every value here is sourced from their code, not guessed:
  - generator/loss defaults  -> options/base_options.py + options/train_options.py
  - the LGE-specific overrides -> cavity_train_debug.py  (netG=stylespade, ngf=32,
    resnet_n_downsample=4, resnet_n_blocks=2, crop=128, num_upsampling_layers='few', niter 150+50)
  - the dataset defaults      -> data/cmrcavityLGEAug_dataset.py (label_nc=5, output_nc=1, no_instance)

⚠ resnet_n_downsample MUST be 4 (the debug override), NOT the StyleSPADE modify_commandline_options
default of 2 — the style encoder's flatten is hardwired to 16*ngf*8*8, which only holds when the encoder
downsamples 128 -> 8 (i.e. 2^4) with ngf*2^4 = 512 = 16*ngf channels. With 2 it dim-mismatches and crashes.
"""
from argparse import Namespace


def make_opt(is_train=True, gpu=True, **overrides):
    o = Namespace()
    # --- experiment / io ---
    o.gpu_ids = [0] if gpu else []
    o.isTrain = is_train
    o.semantic_nc = 5            # = label_nc (no_instance, no dontcare); set explicitly below too
    # --- input / output sizes (their LGE config) ---
    o.label_nc = 5              # bg, LVcav, LVmyo, scar, RVcav  (our scheme == their {BG,LV_Blood,MYO,Scar,NO_reflow})
    o.output_nc = 1            # single-channel LGE
    o.contain_dontcare_label = False
    o.no_instance = True
    o.crop_size = 128
    o.aspect_ratio = 1.0
    # --- generator (stylespade, debug overrides) ---
    o.netG = 'stylespade'
    o.ngf = 32
    o.norm_G = 'spectralspadesyncbatch3x3'   # StyleSPADE sets this default
    o.norm_mode = 'spade'
    o.num_upsampling_layers = 'few'
    o.resnet_n_downsample = 4    # <-- debug override (NOT the modify_commandline_options default of 2)
    o.resnet_n_blocks = 2
    o.resnet_kernel_size = 3
    o.resnet_initial_kernel_size = 7
    o.init_type = 'xavier'
    o.init_variance = 0.02
    o.z_dim = 256
    o.use_vae = False
    o.use_noise = False
    o.no_BG = False
    o.add_dist = False
    # --- discriminator (multiscale) ---
    o.netD = 'multiscale'
    o.netD_subarch = 'n_layer'
    o.num_D = 2
    o.n_layers_D = 4
    o.ndf = 64
    o.norm_D = 'spectralinstance'
    o.norm_E = 'spectralinstance'   # for the VAE ConvEncoder (v3)
    # --- losses ---
    o.gan_mode = 'hinge'
    o.no_ganFeat_loss = False
    o.no_vgg_loss = False
    o.lambda_feat = 10.0
    o.lambda_vgg = 10.0
    o.lambda_kld = 0.05
    o.lambda_L1 = 100.0
    # --- optimisation (TTUR on: G_lr=lr/2, D_lr=lr*2) ---
    o.lr = 0.0002
    o.beta1 = 0.0
    o.beta2 = 0.9
    o.no_TTUR = False
    o.optimizer = 'adam'
    o.D_steps_per_G = 1
    # --- misc the model touches ---
    o.use_amp = False
    o.gpu = gpu
    for k, v in overrides.items():
        setattr(o, k, v)
    # plain SPADE (label-only, no style image) uses the standard 5-upsample stack from a 4x4 seg downsample,
    # NOT stylespade's fixed 'few' (which assumes the 8x8 style-encoder bottleneck). Auto-fix unless overridden.
    if o.netG == "spade" and "num_upsampling_layers" not in overrides:
        o.num_upsampling_layers = "normal"
    # keep semantic_nc consistent if label_nc was overridden
    o.semantic_nc = o.label_nc + (1 if o.contain_dontcare_label else 0) + (0 if o.no_instance else 1)
    return o
