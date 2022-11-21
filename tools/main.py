# encoding: utf-8

import argparse
import os
import sys

import torch
from torch.backends import cudnn

sys.path.append('.')
from config import cfg
from data import make_data_loader
from modeling import build_model
from utils.lr_scheduler import WarmupMultiStepLR
from utils.logger import setup_logger
from tools.train import do_train
from tools.test import do_test


def main():
    parser = argparse.ArgumentParser(description="AGW Re-ID Baseline")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("reid_baseline", output_dir, 0)
    logger.info("Using {} GPUS".format(num_gpus))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DEVICE == "cuda":
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID  # new add by gu
    cudnn.benchmark = True

    if cfg.DATASETS.NAMES.split('_')[-1] != 'val':
        data_loader, num_query, num_classes = make_data_loader(cfg)
        num_val_query = 0
    else:
        data_loader, num_query, num_classes, num_val_query = make_data_loader(cfg)

    if not (cfg.TEST.EVALUATE_ONLY == 'on'):
        model = build_model(cfg, num_classes)
    else:
        model = build_model(cfg, num_classes)

    if 'cpu' not in cfg.MODEL.DEVICE:
        if torch.cuda.device_count() > 1:
            str_ids = cfg.MODEL.DEVICE_ID.split(',')
            gpu_ids = []
            for str_id in str_ids:
                id = int(str_id)
                if id >= 0:
                    gpu_ids.append(id)
            model = torch.nn.DataParallel(model).module

    if cfg.TEST.EVALUATE_ONLY == 'on':
        logger.info("Evaluate Only")
        model.load_param(cfg)
        do_test(cfg, model, data_loader, num_query)
        return

    criterion = model.get_creterion(cfg, num_classes)
    optimizer = model.get_optimizer(cfg, criterion)

    # Add for using self trained model
    if cfg.MODEL.PRETRAIN_CHOICE == 'self':
        start_epoch = eval(cfg.MODEL.PRETRAIN_PATH.split('/')[-1].split('.')[0].split('_')[-1])
        print('Start epoch:', start_epoch)
        path_to_optimizer = cfg.MODEL.PRETRAIN_PATH.replace('model', 'optimizer')
        print('Path to the checkpoint of optimizer:', path_to_optimizer)
        path_to_center_param = cfg.MODEL.PRETRAIN_PATH.replace('model', 'center_param')
        print('Path to the checkpoint of center_param:', path_to_center_param)
        path_to_optimizer_center = cfg.MODEL.PRETRAIN_PATH.replace('model', 'optimizer_center')
        print('Path to the checkpoint of optimizer_center:', path_to_optimizer_center)
        model.load_param(cfg)
        criterion['center'].load_state_dict(torch.load(path_to_center_param))
        optimizer['center'].load_state_dict(torch.load(path_to_optimizer_center))
        if cfg.MODEL.MODE == 'GCR':
            scheduler_g = WarmupMultiStepLR(optimizer['g'], cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA,
                                            cfg.SOLVER.WARMUP_FACTOR,
                                            cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD, start_epoch)
            scheduler_c = WarmupMultiStepLR(optimizer['c'], cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA,
                                            cfg.SOLVER.WARMUP_FACTOR,
                                            cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD, start_epoch)
            scheduler_r = WarmupMultiStepLR(optimizer['r'], cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA,
                                            cfg.SOLVER.WARMUP_FACTOR,
                                            cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD, start_epoch)
            scheduler_d1 = WarmupMultiStepLR(optimizer['d1'], cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA,
                                             cfg.SOLVER.WARMUP_FACTOR,
                                             cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD, start_epoch)
            scheduler_d2 = WarmupMultiStepLR(optimizer['d2'], cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA,
                                             cfg.SOLVER.WARMUP_FACTOR,
                                             cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD)
            schedulers = [scheduler_g, scheduler_c, scheduler_r, scheduler_d1, scheduler_d2]
        elif cfg.MODEL.MODE == 'C':
            scheduler_c = WarmupMultiStepLR(optimizer['c'], cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA,
                                            cfg.SOLVER.WARMUP_FACTOR,
                                            cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD)
            schedulers = [scheduler_c]
    elif cfg.MODEL.PRETRAIN_CHOICE == 'imagenet':
        start_epoch = 0
        if cfg.MODEL.MODE == 'GCR':
            scheduler_g = WarmupMultiStepLR(optimizer['g'], cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA,
                                            cfg.SOLVER.WARMUP_FACTOR,
                                            cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD)
            scheduler_c = WarmupMultiStepLR(optimizer['c'], cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA,
                                            cfg.SOLVER.WARMUP_FACTOR,
                                            cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD)
            scheduler_r = WarmupMultiStepLR(optimizer['r'], cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA,
                                            cfg.SOLVER.WARMUP_FACTOR,
                                            cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD)
            scheduler_d1 = WarmupMultiStepLR(optimizer['d1'], cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA,
                                             cfg.SOLVER.WARMUP_FACTOR,
                                             cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD)
            scheduler_d2 = WarmupMultiStepLR(optimizer['d2'], cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA,
                                             cfg.SOLVER.WARMUP_FACTOR,
                                             cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD)
            schedulers = [scheduler_g, scheduler_c, scheduler_r, scheduler_d1, scheduler_d2]
        elif cfg.MODEL.MODE == 'C':
            scheduler_c = WarmupMultiStepLR(optimizer['c'], cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA,
                                            cfg.SOLVER.WARMUP_FACTOR,
                                            cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD)
            schedulers = [scheduler_c]

    else:
        print('Only support pretrain_choice for imagenet and self, but got {}'.format(cfg.MODEL.PRETRAIN_CHOICE))

    do_train(cfg,
             model,
             data_loader,
             optimizer,
             schedulers,
             criterion,
             num_query,
             start_epoch,
             num_val_query=num_val_query
             )


if __name__ == '__main__':
    main()
