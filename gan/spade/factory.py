"""Network factory — replaces their models/networks/__init__.py find_class_in_module machinery
with a direct construction (we only ever use stylespade G + multiscale D). init exactly as theirs:
create -> .cuda() -> init_weights(init_type, init_variance)  (models/networks/__init__.py:create_network).
"""
import torch
from .generator import StyleSPADEGenerator, SPADEGenerator, Pix2PixHDGenerator
from .discriminator import MultiscaleDiscriminator

_G = {'stylespade': StyleSPADEGenerator, 'spade': SPADEGenerator, 'pix2pixhd': Pix2PixHDGenerator}


def define_G(opt):
    net = _G[opt.netG](opt)
    if len(opt.gpu_ids) > 0:
        assert torch.cuda.is_available()
        net.cuda()
    net.init_weights(opt.init_type, opt.init_variance)
    return net


def define_D(opt):
    net = MultiscaleDiscriminator(opt)
    if len(opt.gpu_ids) > 0:
        assert torch.cuda.is_available()
        net.cuda()
    net.init_weights(opt.init_type, opt.init_variance)
    return net
