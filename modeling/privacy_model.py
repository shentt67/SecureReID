import collections
import os

import torch
import torch.nn as nn

from . import networks, Baseline
from .layer import CrossEntropyLabelSmooth, TripletLoss, WeightedRegularizedTriplet, CenterLoss


class PrivacyModel(nn.Module):
    def __init__(self, cfg, num_classes):
        super(PrivacyModel, self).__init__()

        self.cfg = cfg
        self.train_mode = cfg.MODEL.MODE
        self.output_mode = cfg.MODEL.OUTPUT_MODE
        self.device_id = cfg.MODEL.DEVICE_ID
        str_ids = self.device_id.split(',')
        self.gpu_ids = []
        for id in range(len(str_ids)):
            self.gpu_ids.append(int(str_ids[id]))

        self.isTrain = cfg.TEST.EVALUATE_ONLY == 'off'
        os.environ['CUDA_VISIBLE_DEVICE'] = self.device_id
        # self.device = torch.device('cuda:0') if self.gpu_ids else torch.device('cpu')  # get device name: CPU or GPU
        self.save_dir = os.path.join(cfg.OUTPUT_DIR)  # save all the checkpoints to save_dir

        norm = 'batch'
        if self.train_mode == 'GCR':
            # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
            if self.isTrain:
                self.model_names = ['G', 'G_R', 'C', 'D1', 'D2']
            else:
                self.model_names = ['G', 'G_R', 'C']
            # define networks
            input_nc = output_nc = 3
            self.netG = networks.define_G(input_nc, output_nc, cfg.MODEL.NGF, cfg.MODEL.NETG, norm,
                                          not cfg.MODEL.NO_DROPOUT, cfg.MODEL.INIT_TYPE, cfg.MODEL.INIT_GAIN,
                                          self.gpu_ids, cfg.SOLVER.IMS_PER_BATCH, "anonymize", cfg.MODEL.DEVICE)
            self.netG_R = networks.define_G(input_nc, output_nc, cfg.MODEL.NGF, cfg.MODEL.NETG, norm,
                                            not cfg.MODEL.NO_DROPOUT, cfg.MODEL.INIT_TYPE, cfg.MODEL.INIT_GAIN,
                                            self.gpu_ids, cfg.SOLVER.IMS_PER_BATCH, "recovery", cfg.MODEL.DEVICE)
            self.netC = Baseline(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NAME,
                                 cfg.MODEL.GENERALIZED_MEAN_POOL, cfg.MODEL.PRETRAIN_CHOICE)
            if len(self.gpu_ids) > 0:
                assert (torch.cuda.is_available())
                self.netC.to(self.gpu_ids[0])
                self.netC = torch.nn.DataParallel(self.netC, self.gpu_ids)  # multi-GPUs
            if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
                input_nc_D = input_nc + output_nc
                self.netD1 = networks.define_D(input_nc_D, cfg.MODEL.NDF, cfg.MODEL.NETD,
                                               cfg.MODEL.N_LAYERS_D, norm, cfg.MODEL.INIT_TYPE, cfg.MODEL.INIT_GAIN,
                                               self.gpu_ids)
                self.netD2 = networks.define_D(input_nc_D, cfg.MODEL.NDF, cfg.MODEL.NETD,
                                               cfg.MODEL.N_LAYERS_D, norm, cfg.MODEL.INIT_TYPE, cfg.MODEL.INIT_GAIN,
                                               self.gpu_ids)
        elif self.train_mode == 'C':
            self.model_names = ['C']
            self.netC = Baseline(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NAME,
                                 cfg.MODEL.GENERALIZED_MEAN_POOL, cfg.MODEL.PRETRAIN_CHOICE)
            if len(self.gpu_ids) > 0:
                assert (torch.cuda.is_available())
                self.netC.to(self.gpu_ids[0])
                self.netC = torch.nn.DataParallel(self.netC, self.gpu_ids)  # multi-GPUs

    def forward(self, input, img_M, deviation):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.real_O = input
        self.real_M = img_M
        # for test
        if (not self.isTrain):
            feat_OI = self.netC(self.real_O)
            if self.train_mode == 'C':
                feat_PI = self.netC(self.real_M)
            elif self.train_mode == 'GCR':
                self.fake_P = self.netG(self.real_O)
                self.fake_P_encrypted = self.fake_P + deviation
                self.fake_P_decrypted = self.fake_P_encrypted - deviation
                if self.cfg.TEST.RECOVERY_WITH_KEY:
                    self.fake_R = self.netG_R(self.fake_P_decrypted)
                else:
                    self.fake_R = self.netG_R(self.fake_P_encrypted)
                if self.cfg.TEST.RETRIEVE_WITH_KEY:
                    feat_PI = self.netC(self.fake_P_decrypted)
                else:
                    feat_PI = self.netC(self.fake_P_encrypted)
            return feat_OI, feat_PI
        # for train
        if self.train_mode == 'C':
            if self.output_mode == 'protected':
                self.pred_PI, self.global_feat_PI = self.netC(self.real_M)
                return self.pred_PI, self.global_feat_PI
            elif self.output_mode == 'both':
                self.pred_OI, self.global_feat_OI = self.netC(self.real_O)
                self.pred_PI, self.global_feat_PI = self.netC(self.real_M)
                return self.pred_OI, self.global_feat_OI, self.pred_PI, self.global_feat_PI
            else:
                self.pred_OI, self.global_feat_OI = self.netC(self.real_O)
                return self.pred_OI, self.global_feat_OI
        elif self.train_mode == 'GCR':
            self.pred_OI, self.global_feat_OI = self.netC(self.real_O)
            self.fake_P = self.netG(self.real_O)
            self.fake_R = self.netG_R(self.fake_P)
            if self.output_mode == 'protected':
                self.pred_PI, self.global_feat_PI = self.netC(self.fake_P)
                return self.pred_PI, self.global_feat_PI
            elif self.output_mode == 'both':
                self.pred_PI, self.global_feat_PI = self.netC(self.fake_P)
                return self.pred_OI, self.global_feat_OI, self.pred_PI, self.global_feat_PI
            else:
                return self.pred_OI, self.global_feat_OI

    def get_optimizer(self, cfg, criterion):
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        optimizer = {}
        if self.train_mode == 'GCR':
            params_g = [{"params": self.netG.parameters(), "lr": lr, "weight_decay": weight_decay}]
            params_c = [{"params": self.netC.parameters(), "lr": lr, "weight_decay": weight_decay}]
            params_r = [{"params": self.netG_R.parameters(), "lr": lr, "weight_decay": weight_decay}]
            params_d1 = [{"params": self.netD1.parameters(), "lr": lr, "weight_decay": weight_decay}]
            params_d2 = [{"params": self.netD2.parameters(), "lr": lr, "weight_decay": weight_decay}]
            if cfg.SOLVER.OPTIMIZER_NAME == 'SGD':
                optimizer['g'] = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params_g, momentum=cfg.SOLVER.MOMENTUM)
                optimizer['c'] = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params_c, momentum=cfg.SOLVER.MOMENTUM)
                optimizer['r'] = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params_r, momentum=cfg.SOLVER.MOMENTUM)
                optimizer['d1'] = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params_d1,
                                                                                  momentum=cfg.SOLVER.MOMENTUM)
                optimizer['d2'] = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params_d2,
                                                                                  momentum=cfg.SOLVER.MOMENTUM)
            else:
                optimizer['g'] = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params_g, betas=(0.5, 0.999))
                optimizer['c'] = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params_c)
                optimizer['r'] = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params_r, betas=(0.5, 0.999))
                optimizer['d1'] = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params_d1, betas=(0.5, 0.999))
                optimizer['d2'] = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params_d2, betas=(0.5, 0.999))
        elif self.train_mode == 'C':
            params_c = [{"params": self.netC.parameters(), "lr": lr, "weight_decay": weight_decay}]
            if cfg.SOLVER.OPTIMIZER_NAME == 'SGD':
                optimizer['c'] = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params_c, momentum=cfg.SOLVER.MOMENTUM)
            else:
                optimizer['c'] = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params_c)
        if cfg.MODEL.CENTER_LOSS == 'on':
            optimizer['center'] = torch.optim.SGD(criterion['center'].parameters(), lr=cfg.SOLVER.CENTER_LR)
        return optimizer

    def get_creterion(self, cfg, num_classes):
        criterion = {}
        criterion['xent'] = CrossEntropyLabelSmooth(num_classes=num_classes)  # new add by luo

        print("Weighted Regularized Triplet:", cfg.MODEL.WEIGHT_REGULARIZED_TRIPLET)
        if cfg.MODEL.WEIGHT_REGULARIZED_TRIPLET == 'on':
            criterion['triplet'] = WeightedRegularizedTriplet()
        else:
            criterion['triplet'] = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss

        if cfg.MODEL.CENTER_LOSS == 'on':
            criterion['center'] = CenterLoss(num_classes=num_classes, feat_dim=cfg.MODEL.CENTER_FEAT_DIM,
                                             use_gpu=True)

        def criterion_reid(score, feat, target):
            loss = criterion['xent'](score, target) + criterion['triplet'](feat, target)[0]
            if cfg.MODEL.CENTER_LOSS == 'on':
                loss = loss + cfg.SOLVER.CENTER_LOSS_WEIGHT * criterion['center'](feat, target)
            return loss

        criterion['reid'] = criterion_reid

        # criterion['gan'] = networks.GANLoss('vanilla').cuda()
        criterion['gan'] = networks.GANLoss('vanilla').to("cuda:0")
        criterion['l1'] = torch.nn.L1Loss()

        def criterion_G_gan():
            fake_OP = torch.cat((self.real_O, self.fake_P), 1)
            fake_PR = torch.cat((self.fake_P, self.fake_R), 1)
            pred_fake_OP = self.netD1(fake_OP)
            pred_fake_PR = self.netD2(fake_PR)
            loss_g_gan = criterion['gan'](pred_fake_OP, True) + criterion['gan'](pred_fake_PR, True)
            return loss_g_gan

        def criterion_G_l1():
            loss_g_l1 = torch.nn.L1Loss()(self.fake_P, self.real_M)
            return loss_g_l1

        def criterion_R_l1():
            loss_g_r_l1 = torch.nn.L1Loss()(self.fake_R, self.real_O)
            return loss_g_r_l1

        def criterion_C(score1, feat1, score2, feat2, target):
            loss_reid1 = criterion_reid(score1, feat1, target)
            loss_reid2 = criterion_reid(score2, feat2, target)
            return loss_reid1 + loss_reid2

        def criterion_D1():
            # Fake; stop backprop to the generator by detaching fake_B
            fake_OP = torch.cat((self.real_O, self.fake_P),
                                1)  # we use conditional GANs; we need to feed both input and output to the discriminator
            pred_fake = self.netD1(fake_OP.detach())
            loss_d_fake = criterion['gan'](pred_fake, False)
            # Real
            real_OM = torch.cat((self.real_O, self.real_M), 1)
            pred_real = self.netD1(real_OM.detach())
            loss_d_real = criterion['gan'](pred_real, True)
            # combine loss and calculate gradients
            loss = loss_d_fake + loss_d_real
            return loss

        def criterion_D2():
            fake_PR = torch.cat((self.fake_P, self.fake_R),
                                1)  # we use conditional GANs; we need to feed both input and output to the discriminator
            pred_fake = self.netD2(fake_PR.detach())
            loss_d_fake = criterion['gan'](pred_fake, False)
            # Real
            real_PO = torch.cat((self.fake_P, self.real_O), 1)
            pred_real = self.netD2(real_PO.detach())
            loss_d_real = criterion['gan'](pred_real, True)
            # combine loss and calculate gradients
            loss = loss_d_fake + loss_d_real
            return loss

        criterion['G_gan'] = criterion_G_gan
        criterion['G_l1'] = criterion_G_l1
        criterion['R_l1'] = criterion_R_l1
        criterion['C'] = criterion_C
        criterion['D1'] = criterion_D1
        criterion['D2'] = criterion_D2

        return criterion

    def load_param(self, cfg):
        if isinstance(cfg.MODEL.EPOCH, int):
            epoch = 'epoch_%d' % cfg.MODEL.EPOCH
        else:
            epoch = 'epoch_' + cfg.MODEL.EPOCH
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (epoch, name)
                load_path = os.path.join(cfg.MODEL.PRETRAIN_DIR, load_filename)
                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    # if name != 'C':
                    #     net = net.module
                    net = net.module
                print('loading the model from %s' % load_path)
                # if you are using PyTorch newer than 0.4 (e.g., built from
                # GitHub source), you can remove str() on self.device
                state_dict = torch.load(load_path, map_location=str(torch.device('cuda:{}'.format(0))))
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata
                if name == 'C':
                    if not isinstance(state_dict, collections.OrderedDict):
                        state_dict = state_dict.state_dict()
                    for i in state_dict:
                        if 'classifier' in i:
                            continue
                        net.state_dict()[i].copy_(state_dict[i])
                else:
                    # patch InstanceNorm checkpoints prior to 0.4
                    for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                        self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
                    net.load_state_dict(state_dict)

    def save_networks(self, epoch):
        """Save all the networks to the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)

                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    if name == "C":
                        torch.save(net.module.state_dict(), save_path)
                        # net.cuda()
                        net.to("cuda:0")
                    else:
                        torch.save(net.module.cpu().state_dict(), save_path)
                        # net.cuda()
                        net.to("cuda:0")
                else:
                    if name == "C":
                        torch.save(net.cpu().state_dict(), save_path)
                    else:
                        torch.save(net.cpu().state_dict(), save_path)

    def eval(self):
        self.isTrain = False
        """Make models eval mode during test time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()

    # modified HERE
    def train(self):
        """Make models eval mode during test time"""
        self.isTrain = True
        for name in self.model_names:
            # Here
            # if name == 'C':
            #     continue

            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.train()

    def test(self):
        """Forward function used in test time.

        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
        It also calls <compute_visuals> to produce additional visualization results
        """
        with torch.no_grad():
            self.forward()
            self.compute_visuals()

    def compute_visuals(self):
        """Calculate additional output images for visdom and HTML visualization"""
        pass

    def get_image_paths(self):
        """ Return image paths that are used to load current data"""
        return self.image_paths

    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        old_lr = self.optimizers[0].param_groups[0]['lr']
        for scheduler in self.schedulers:
            if self.opt.lr_policy == 'plateau':
                scheduler.step(self.metric)
            else:
                scheduler.step()

        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate %.7f -> %.7f' % (old_lr, lr))

    def get_current_visuals(self):
        """Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    def get_current_losses(self):
        """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(
                    getattr(self, 'loss_' + name))  # float(...) works for both scalar tensor and float number
        return errors_ret

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        """Fix InstanceNorm checkpoints incompatibility (prior to 0.4)"""
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

    def print_networks(self, verbose):
        """Print the total number of parameters in the network and (if verbose) network architecture

        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
