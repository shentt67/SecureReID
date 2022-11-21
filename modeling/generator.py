from modeling import networks
import torch.nn as nn
import os
import torch


class Generator(nn.Module):
    def __init__(self, cfg):
        super(Generator, self).__init__()
        self.cfg = cfg
        self.device_id = cfg.MODEL.DEVICE_ID
        str_ids = self.device_id.split(',')
        gpu_ids = []
        for id in range(len(str_ids)):
            gpu_ids.append(int(str_ids[id]))
        os.environ['CUDA_VISIBLE_DEVICE'] = self.device_id
        input_nc = output_nc = 3
        norm = 'batch'
        self.netG = networks.define_G(input_nc, output_nc, cfg.MODEL.NGF, cfg.MODEL.NETG, norm,
                                      not cfg.MODEL.NO_DROPOUT, cfg.MODEL.INIT_TYPE, cfg.MODEL.INIT_GAIN, gpu_ids,
                                      cfg.SOLVER.IMS_PER_BATCH, cfg.MODEL.DEVICE)

    def forward(self, input):
        self.real_O = input
        self.fake_P = self.netG(self.real_O)
        return self.fake_P

    def load_param(self, cfg):
        name = 'G'
        load_filename = 'epoch_%s_net_%s.pth' % ('val_best', name)
        load_path = os.path.join(cfg.OUTPUT_DIR, load_filename)
        net = getattr(self, 'net' + name)
        if isinstance(net, torch.nn.DataParallel):
            net = net.module
        print('loading the model from %s' % load_path)
        # if you are using PyTorch newer than 0.4 (e.g., built from
        # GitHub source), you can remove str() on self.device
        state_dict = torch.load(load_path, map_location=str(torch.device('cuda:{}'.format(0))))
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata

        # patch InstanceNorm checkpoints prior to 0.4
        for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
            self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
        net.load_state_dict(state_dict)

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
