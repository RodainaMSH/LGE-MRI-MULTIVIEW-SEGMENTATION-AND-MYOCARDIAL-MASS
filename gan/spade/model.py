"""SpadeModel — faithful port of CMRISynthSeg models/pix2pix_model.py (loss/forward math VERBATIM).
Supports the three generator regimes we use:
  - netG='stylespade'           : style-encoder conditioned on a real slice (v1.5; ghosts at N=32)
  - netG='spade' (use_vae=False): label-only deterministic SPADE (v2 — chosen; ghost-free)
  - netG='spade' + use_vae=True : SPADE-VAE (v3) — encoder->latent at train (KLD), sample z~N(0,1) at
                                  generation for appearance DIVERSITY. No spatial style image -> cannot ghost.
AMP and the L1/VAE-recon extras are omitted (off in their LGE config). Checkpoint IO = plain torch.save.
"""
import torch
from .factory import define_G, define_D
from .encoder import ConvEncoder
from .loss import GANLoss, VGGLoss, KLDLoss


class SpadeModel(torch.nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.FloatTensor = torch.cuda.FloatTensor if self.use_gpu() else torch.FloatTensor

        self.netG = define_G(opt)
        self.netD = define_D(opt) if opt.isTrain else None
        self.netE = None
        if opt.use_vae:
            self.netE = ConvEncoder(opt)
            if self.use_gpu():
                self.netE.cuda()
            self.netE.init_weights(opt.init_type, opt.init_variance)

        if opt.isTrain:
            self.criterionGAN = GANLoss(opt.gan_mode, tensor=self.FloatTensor, opt=self.opt)
            self.criterionFeat = torch.nn.L1Loss()
            if not opt.no_vgg_loss:
                self.criterionVGG = VGGLoss(self.opt.gpu_ids)
            if opt.use_vae:
                self.KLDLoss = KLDLoss()

    # ----- entry point -----
    def forward(self, data, mode):
        input_semantics, real_image = self.preprocess_input(data)
        if mode == 'generator':
            return self.compute_generator_loss(input_semantics, real_image)
        elif mode == 'discriminator':
            return self.compute_discriminator_loss(input_semantics, real_image)
        elif mode == 'inference':
            with torch.no_grad():
                fake, _, _ = self.generate_fake(input_semantics, real_image)
            return fake
        elif mode == 'sample':
            # VAE generation: z ~ N(0,1) (appearance diversity), label drives anatomy, no style image
            with torch.no_grad():
                z = torch.randn(input_semantics.size(0), self.opt.z_dim,
                                dtype=torch.float32, device=input_semantics.device)
                fake = self.netG(input_semantics, z=z)
            return fake
        raise ValueError("|mode| is invalid")

    def create_optimizers(self, opt):
        G_params = list(self.netG.parameters())
        if opt.use_vae:
            G_params += list(self.netE.parameters())
        D_params = list(self.netD.parameters())
        beta1, beta2 = opt.beta1, opt.beta2
        G_lr, D_lr = (opt.lr, opt.lr) if opt.no_TTUR else (opt.lr / 2, opt.lr * 2)
        return (torch.optim.Adam(G_params, lr=G_lr, betas=(beta1, beta2)),
                torch.optim.Adam(D_params, lr=D_lr, betas=(beta1, beta2)))

    def save(self, path_prefix):
        torch.save(self.netG.state_dict(), f"{path_prefix}_G.pth")
        if self.netD is not None:
            torch.save(self.netD.state_dict(), f"{path_prefix}_D.pth")
        if self.netE is not None:
            torch.save(self.netE.state_dict(), f"{path_prefix}_E.pth")

    # ----- helpers (verbatim logic) -----
    def preprocess_input(self, data):
        data['label'] = data['label'].long()
        if self.use_gpu():
            data['label'] = data['label'].cuda()
            data['image'] = data['image'].cuda()
        label_map = data['label']
        bs, _, h, w = label_map.size()
        nc = self.opt.label_nc + 1 if self.opt.contain_dontcare_label else self.opt.label_nc
        input_label = self.FloatTensor(bs, nc, h, w).zero_()
        input_semantics = input_label.scatter_(1, label_map, 1.0)
        if self.opt.no_BG:
            input_semantics[:, 0, :, :] = 0
        return input_semantics, data['image']

    def encode_z(self, real_image):
        mu, logvar, _ = self.netE(real_image)
        std = torch.exp(0.5 * logvar)
        z = torch.randn_like(std).mul(std).add_(mu)
        return z, mu, logvar

    def generate_fake(self, input_semantics, real_image):
        KLD = None
        if self.opt.use_vae:
            z, mu, logvar = self.encode_z(real_image)
            fake = self.netG(input_semantics, z=z)
            if self.opt.isTrain:
                KLD = self.KLDLoss(mu, logvar) * self.opt.lambda_kld
        elif self.opt.netG == 'stylespade':
            fake = self.netG(input_semantics, real_image)
        else:
            fake = self.netG(input_semantics, z=None)
        return fake, KLD, None

    def compute_generator_loss(self, input_semantics, real_image):
        G_losses = {}
        fake_image, KLD, _ = self.generate_fake(input_semantics, real_image)
        if KLD is not None:
            G_losses['KLD'] = KLD
        pred_fake, pred_real = self.discriminate(input_semantics, fake_image, real_image)
        G_losses['GAN'] = self.criterionGAN(pred_fake, True, for_discriminator=False)
        if not self.opt.no_ganFeat_loss:
            num_D = len(pred_fake)
            GAN_Feat_loss = self.FloatTensor(1).fill_(0)
            for i in range(num_D):
                for j in range(len(pred_fake[i]) - 1):
                    GAN_Feat_loss += self.criterionFeat(pred_fake[i][j], pred_real[i][j].detach()) * self.opt.lambda_feat / num_D
            G_losses['GAN_Feat'] = GAN_Feat_loss
        if not self.opt.no_vgg_loss:
            G_losses['VGG'] = self.criterionVGG(fake_image, real_image) * self.opt.lambda_vgg
        return G_losses, fake_image

    def compute_discriminator_loss(self, input_semantics, real_image):
        D_losses = {}
        with torch.no_grad():
            fake_image, _, _ = self.generate_fake(input_semantics, real_image)
            fake_image = fake_image.detach()
            fake_image.requires_grad_()
        pred_fake, pred_real = self.discriminate(input_semantics, fake_image, real_image)
        D_losses['D_Fake'] = self.criterionGAN(pred_fake, False, for_discriminator=True)
        D_losses['D_real'] = self.criterionGAN(pred_real, True, for_discriminator=True)
        return D_losses

    def discriminate(self, input_semantics, fake_image, real_image):
        fake_concat = torch.cat([input_semantics, fake_image], dim=1)
        real_concat = torch.cat([input_semantics, real_image], dim=1)
        fake_and_real = torch.cat([fake_concat, real_concat], dim=0)
        discriminator_out = self.netD(fake_and_real)
        return self.divide_pred(discriminator_out)

    def divide_pred(self, pred):
        if type(pred) == list:
            fake, real = [], []
            for p in pred:
                fake.append([t[:t.size(0) // 2] for t in p])
                real.append([t[t.size(0) // 2:] for t in p])
        else:
            fake = pred[:pred.size(0) // 2]
            real = pred[pred.size(0) // 2:]
        return fake, real

    def use_gpu(self):
        return len(self.opt.gpu_ids) > 0
