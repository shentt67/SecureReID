# encoding: utf-8

import logging
import torch
import torch.nn as nn
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, Timer
from ignite.metrics import RunningAverage

from utils.reid_metric import r1_mAP_mINP
from tools.test import create_supervised_evaluator
from utils import util
from modeling.generator import Generator
from torch.utils.tensorboard import SummaryWriter
import os

global ITER
ITER = 0


def create_supervised_trainer(model, generator, optimizer, criterion, cfg, metric, logger, device=None):
    """
    Factory function for creating a trainer for supervised models

    Args:
        model (`torch.nn.Module`): the model to train
        optimizer (dict - class:`torch.optim.Optimizer`): the optimizer to use
        criterion (dict - class:loss function): the loss function to use
        cetner_loss_weight (float, optional): the weight for cetner_loss_weight
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.

    Returns:
        Engine: a trainer engine with supervised update function
    """
    train_mode = cfg.MODEL.MODE
    output_mode = cfg.MODEL.OUTPUT_MODE
    cetner_loss_weight = cfg.SOLVER.CENTER_LOSS_WEIGHT
    r1 = [cfg.MODEL.VAL_R1]
    epsilon_psnr = cfg.MODEL.PSNR
    epsilon_ssim = cfg.MODEL.SSIM

    def _update(engine, batch):
        model.train()

        if 'center' in optimizer.keys():
            optimizer['center'].zero_grad()

        img_O, img_M, target, deviation,  _, _ = batch

        deviation = deviation.to(device) if torch.cuda.device_count() >= 1 else deviation
        img_O = img_O.to(device) if torch.cuda.device_count() >= 1 else img_O
        anonymous_enough = metric['psnr_OP'] < metric['psnr_OM'] + epsilon_psnr and metric['ssim_OP'] < metric['ssim_OM'] + epsilon_ssim
        if cfg.DATASETS.NAMES.split('_')[-1] == 'val' and anonymous_enough and metric['r1'] > cfg.MODEL.VAL_R1:
            if metric['r1'] > r1[0]:
                logger.info("update generator")
                generator.load_param(cfg)
                generator.eval()
                r1[0] = metric['r1']
            img_M = generator(img_O)
        img_M = img_M.to(device) if torch.cuda.device_count() >= 1 else img_M
        img_M = img_M.detach()
        target = target.to(device) if torch.cuda.device_count() >= 1 else target
        if train_mode == 'GCR':
            if output_mode == 'both':
                score_O, feat_O, score_P, feat_P = model(img_O, img_M, deviation)
                loss_C_O = criterion['reid'](score_O, feat_O, target) * 1
                loss_C_P = criterion['reid'](score_P, feat_P, target) * 1
                loss_C = loss_C_O + loss_C_P
                # loss_C = loss_C_P

            elif output_mode == 'protected':
                score_P, feat_P = model(img_O, img_M, deviation)
                loss_C_P = criterion['reid'](score_P, feat_P, target) * 1
                loss_C = loss_C_P
            else:
                score_O, feat_O = model(img_O, img_M, deviation)
                loss_C_O = criterion['reid'](score_O, feat_O, target) * 1
                loss_C = loss_C_O
            model.set_requires_grad(model.netD1, True) 
            model.set_requires_grad(model.netD2, True) 
            loss_D1 = criterion['D1']()
            loss_D2 = criterion['D2']()
            optimizer['d1'].zero_grad()
            optimizer['d2'].zero_grad()
            loss_D1.backward()
            loss_D2.backward()
            optimizer['d1'].step()
            optimizer['d2'].step()
            model.set_requires_grad(model.netD1, False) 
            model.set_requires_grad(model.netD2, False) 
            loss_G_gan = criterion['G_gan']() * 1
            loss_G_l1 = criterion['G_l1']() * 100
            loss_R_l1 = criterion['R_l1']() * 100
            # if cfg.DATASETS.NAMES.split('_')[-1] == 'val' and metric['psnr_OP'] < metric['psnr_OM'] + 0.5 and metric['ssim_OP'] < metric['ssim_OM'] + 0.01:
            #     loss_GCR = loss_G_gan + loss_R_l1 + loss_C
            # else:
            #     loss_GCR = loss_G_gan + loss_G_l1 + loss_R_l1 + loss_C

            loss_GCR = loss_G_gan + loss_G_l1 + loss_R_l1 + loss_C
            optimizer['g'].zero_grad()
            optimizer['c'].zero_grad()
            optimizer['r'].zero_grad()
            loss_GCR.backward()
            optimizer['g'].step()
            optimizer['c'].step()
            optimizer['r'].step()
            loss = loss_D1 + loss_D2 + loss_GCR
        elif train_mode == 'C':
            if output_mode == 'both':
                score_O, feat_O, score_P, feat_P = model(img_O, img_M, deviation)
                loss_C_O = criterion['reid'](score_O, feat_O, target) 
                loss_C_P = criterion['reid'](score_P, feat_P, target) 
                loss_C = loss_C_O + loss_C_P
            elif output_mode == 'protected':
                score_P, feat_P = model(img_O, img_M, deviation)
                loss_C_P = criterion['reid'](score_P, feat_P, target)
                loss_C = loss_C_P
            else:
                score_O, feat_O = model(img_O, img_M, deviation)
                loss_C_O = criterion['reid'](score_O, feat_O, target)
                loss_C = loss_C_O
            optimizer['c'].zero_grad()
            loss_C.backward()
            optimizer['c'].step()
            loss = loss_C

        if 'center' in optimizer.keys():
            for param in criterion['center'].parameters():
                param.grad.data *= (1. / cetner_loss_weight)
            optimizer['center'].step()

        # compute acc
        if output_mode == 'both':
            acc = ((score_O.max(1)[1] == target).float().mean() +  (score_P.max(1)[1] == target).float().mean()) / 2
        elif output_mode == 'protected':
            acc = (score_P.max(1)[1] == target).float().mean()
        else:
            acc = (score_O.max(1)[1] == target).float().mean()
        if train_mode == 'GCR':
            if output_mode == 'both':
                return loss.item(), acc.item(), loss_G_gan.item(), loss_G_l1.item(), loss_R_l1.item(), loss_D1.item(), loss_D2.item(), loss_C_O.item(), loss_C_P.item()
            elif output_mode == 'protected':
                return loss.item(), acc.item(), loss_G_gan.item(), loss_G_l1.item(), loss_R_l1.item(), loss_D1.item(), loss_D2.item(), loss_C_P.item()
            else:
                return loss.item(), acc.item(), loss_G_gan.item(), loss_G_l1.item(), loss_R_l1.item(), loss_D1.item(), loss_D2.item(), loss_C_O.item()
        elif train_mode == 'C':
            if output_mode == 'both':
                return loss.item(), acc.item(), loss_C_O.item(), loss_C_P.item()
            elif output_mode == 'protected':
                return loss.item(), acc.item(), loss_C_P.item()
            else:
                return loss.item(), acc.item(), loss_C_O.item()


    return Engine(_update)


def do_train(
        cfg,
        model,
        data_loader,
        optimizer,
        schedulers,
        criterion,
        num_query,
        start_epoch,
        num_val_query = 0
):
    # model.eval()
    model.train()
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD
    val_period = cfg.SOLVER.VAL_PERIOD
    output_dir = cfg.OUTPUT_DIR
    device = cfg.MODEL.DEVICE
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("reid_baseline")
    logger.info("Start training")

    generator = Generator(cfg)
    # generator.load_param(cfg)
    

    metric = {'r1': 0.0, 'r1_max':0.0, 'psnr_OM':0.0, 'ssim_OM':0.0, 'psnr_OP':100.0, 'ssim_OP':1.0}
    test_metric = {'r1': 0.0}
    trainer = create_supervised_trainer(model, generator, optimizer, criterion, cfg, metric, logger, device=device)

    # tensorboard writer
    loss_G_L1_writer = SummaryWriter(os.path.join(cfg.OUTPUT_DIR, 'runs', "loss", "G_L1"))
    loss_G_GAN_writer = SummaryWriter(os.path.join(cfg.OUTPUT_DIR, 'runs', "loss", "G_GAN"))
    loss_G_R_L1_writer = SummaryWriter(os.path.join(cfg.OUTPUT_DIR, 'runs', "loss", "G_R_L1"))
    loss_D1_writer = SummaryWriter(os.path.join(cfg.OUTPUT_DIR, 'runs', "loss", "D1"))
    loss_D2_writer = SummaryWriter(os.path.join(cfg.OUTPUT_DIR, 'runs', "loss", "D2"))
    loss_C_PI_writer = SummaryWriter(os.path.join(cfg.OUTPUT_DIR, 'runs', "loss", "C_PI"))
    loss_C_OI_writer = SummaryWriter(os.path.join(cfg.OUTPUT_DIR, 'runs', "loss", "C_OI"))
    r1_OI_writer = SummaryWriter(os.path.join(cfg.OUTPUT_DIR, 'runs', "metrics", "r1_OI"))
    map_OI_writer = SummaryWriter(os.path.join(cfg.OUTPUT_DIR, 'runs', "metrics", "mAP_OI"))
    minp_OI_writer = SummaryWriter(os.path.join(cfg.OUTPUT_DIR, 'runs', "metrics", "mINP_OI"))
    r1_PI_writer = SummaryWriter(os.path.join(cfg.OUTPUT_DIR, 'runs', "metrics", "r1_PI"))
    map_PI_writer = SummaryWriter(os.path.join(cfg.OUTPUT_DIR, 'runs', "metrics", "mAP_PI"))
    minp_PI_writer = SummaryWriter(os.path.join(cfg.OUTPUT_DIR, 'runs', "metrics", "mINP_PI"))

    valing_results = None    
    if cfg.MODEL.MODE == 'GCR':
        valing_results = {'batch_sizes': 0, 'mse_OM': 0, 'ssims_OM': 0, 'psnr_OM': 0, 'ssim_OM': 0, \
            'mse_OP': 0, 'ssims_OP': 0, 'psnr_OP': 0, 'ssim_OP': 0, \
            'mse_OR': 0, 'ssims_OR': 0, 'psnr_OR': 0, 'ssim_OR': 0}
    elif cfg.MODEL.MODE == 'C':
        valing_results = {'batch_sizes': 0, 'mse_OM': 0, 'ssims_OM': 0, 'psnr_OM': 0, 'ssim_OM': 0}
    if cfg.TEST.PARTIAL_REID == 'off':
        evaluator = create_supervised_evaluator(model, metrics={
            'r1_mAP_mINP': r1_mAP_mINP(num_query, gallery_mode=cfg.TEST.GALLERY_MODE, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)}, \
                cfg=cfg, valing_results=valing_results)
    elif cfg.TEST.PARTIAL_REID == 'on':
        evaluator_reid = create_supervised_evaluator(model, 
                                                     metrics={'r1_mAP_mINP': r1_mAP_mINP(300, gallery_mode=cfg.TEST.GALLERY_MODE, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)},
                                                     cfg=cfg, valing_results=valing_results)
        evaluator_ilids = create_supervised_evaluator(model, 
                                                      metrics={'r1_mAP_mINP': r1_mAP_mINP(119, gallery_mode=cfg.TEST.GALLERY_MODE, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)},
                                                      cfg=cfg, valing_results=valing_results)
    if cfg.DATASETS.NAMES.split('_')[-1] == 'val':
        evaluator_val = create_supervised_evaluator(model, metrics={
            'r1_mAP_mINP': r1_mAP_mINP(num_val_query, gallery_mode=cfg.TEST.GALLERY_MODE, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)},  cfg=cfg, valing_results=valing_results)

    checkpointer = ModelCheckpoint(output_dir, cfg.MODEL.NAME, checkpoint_period, n_saved=10, require_empty=False)
    if cfg.MODEL.MODE == 'GCR':
        trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {'model': model,
                                                                     'optimizer_g': optimizer['g'],
                                                                     'optimizer_c': optimizer['c'],
                                                                     'optimizer_r': optimizer['r'],
                                                                     'optimizer_d1': optimizer['d1'],
                                                                     'optimizer_d2': optimizer['d2'],
                                                                     'center_param': criterion['center'],
                                                                     'optimizer_center': optimizer['center']
                                                                    })
    elif cfg.MODEL.MODE == 'C':
        trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {'model': model,
                                                                     'optimizer_c': optimizer['c'],
                                                                     'center_param': criterion['center'],
                                                                     'optimizer_center': optimizer['center']
                                                                    })
    timer = Timer(average=True)
    timer.attach(trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                 pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)

    # average metric to attach on trainer
    if cfg.MODEL.MODE == 'GCR':
        if cfg.MODEL.OUTPUT_MODE == 'both':
            RunningAverage(output_transform=lambda x: x[0]).attach(trainer, 'avg_loss')
            RunningAverage(output_transform=lambda x: x[1]).attach(trainer, 'avg_acc')
            RunningAverage(output_transform=lambda x: x[2]).attach(trainer, 'loss_G_gan')
            RunningAverage(output_transform=lambda x: x[3]).attach(trainer, 'loss_G_l1')
            RunningAverage(output_transform=lambda x: x[4]).attach(trainer, 'loss_R_l1')
            RunningAverage(output_transform=lambda x: x[5]).attach(trainer, 'loss_D1')
            RunningAverage(output_transform=lambda x: x[6]).attach(trainer, 'loss_D2')
            RunningAverage(output_transform=lambda x: x[7]).attach(trainer, 'loss_C_O')
            RunningAverage(output_transform=lambda x: x[8]).attach(trainer, 'loss_C_P')
        elif cfg.MODEL.OUTPUT_MODE == 'protected':
            RunningAverage(output_transform=lambda x: x[0]).attach(trainer, 'avg_loss')
            RunningAverage(output_transform=lambda x: x[1]).attach(trainer, 'avg_acc')
            RunningAverage(output_transform=lambda x: x[2]).attach(trainer, 'loss_G_gan')
            RunningAverage(output_transform=lambda x: x[3]).attach(trainer, 'loss_G_l1')
            RunningAverage(output_transform=lambda x: x[4]).attach(trainer, 'loss_R_l1')
            RunningAverage(output_transform=lambda x: x[5]).attach(trainer, 'loss_D1')
            RunningAverage(output_transform=lambda x: x[6]).attach(trainer, 'loss_D2')
            RunningAverage(output_transform=lambda x: x[7]).attach(trainer, 'loss_C_P')
        else:
            RunningAverage(output_transform=lambda x: x[0]).attach(trainer, 'avg_loss')
            RunningAverage(output_transform=lambda x: x[1]).attach(trainer, 'avg_acc')
            RunningAverage(output_transform=lambda x: x[2]).attach(trainer, 'loss_G_gan')
            RunningAverage(output_transform=lambda x: x[3]).attach(trainer, 'loss_G_l1')
            RunningAverage(output_transform=lambda x: x[4]).attach(trainer, 'loss_R_l1')
            RunningAverage(output_transform=lambda x: x[5]).attach(trainer, 'loss_D1')
            RunningAverage(output_transform=lambda x: x[6]).attach(trainer, 'loss_D2')
            RunningAverage(output_transform=lambda x: x[7]).attach(trainer, 'loss_C_O')
    elif cfg.MODEL.MODE == 'C':
        if cfg.MODEL.OUTPUT_MODE == 'both':
            RunningAverage(output_transform=lambda x: x[0]).attach(trainer, 'avg_loss')
            RunningAverage(output_transform=lambda x: x[1]).attach(trainer, 'avg_acc') 
            RunningAverage(output_transform=lambda x: x[2]).attach(trainer, 'loss_C_O')
            RunningAverage(output_transform=lambda x: x[3]).attach(trainer, 'loss_C_P')
        elif cfg.MODEL.OUTPUT_MODE == 'protected':
            RunningAverage(output_transform=lambda x: x[0]).attach(trainer, 'avg_loss')
            RunningAverage(output_transform=lambda x: x[1]).attach(trainer, 'avg_acc') 
            RunningAverage(output_transform=lambda x: x[2]).attach(trainer, 'loss_C_P')
        else:
            RunningAverage(output_transform=lambda x: x[0]).attach(trainer, 'avg_loss')
            RunningAverage(output_transform=lambda x: x[1]).attach(trainer, 'avg_acc') 
            RunningAverage(output_transform=lambda x: x[2]).attach(trainer, 'loss_C_O')

    @trainer.on(Events.STARTED)
    def start_training(engine):
        engine.state.epoch = start_epoch

    @trainer.on(Events.EPOCH_STARTED)
    def adjust_learning_rate(engine):
        for scheduler in schedulers:
            scheduler.step()


    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        global ITER
        ITER += 1

        if ITER % log_period == 0:
            logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                        .format(engine.state.epoch, ITER, len(data_loader['train']),
                                engine.state.metrics['avg_loss'], engine.state.metrics['avg_acc'],
                                schedulers[0].get_lr()[0]))
            if cfg.MODEL.MODE == 'GCR':
                if cfg.MODEL.OUTPUT_MODE == 'both':
                    logger.info("Loss_G_gan: {:.3f}, Loss_G_l1: {:.3f}, Loss_R_l1: {:.3f}, Loss_D1: {:.3f}, Loss_D2: {:.3f}, Loss_C_O: {:.3f}, Loss_C_P: {:.3f}"
                        .format(engine.state.metrics['loss_G_gan'],engine.state.metrics['loss_G_l1'],
                                engine.state.metrics['loss_R_l1'], engine.state.metrics['loss_D1'],  engine.state.metrics['loss_D2'], 
                                engine.state.metrics['loss_C_O'], engine.state.metrics['loss_C_P']))
                    loss_G_L1_writer.add_scalar('loss', engine.state.metrics['loss_G_l1'], engine.state.epoch)
                    loss_G_GAN_writer.add_scalar('loss', engine.state.metrics['loss_G_gan'], engine.state.epoch)
                    loss_G_R_L1_writer.add_scalar('loss',engine.state.metrics['loss_R_l1'], engine.state.epoch)
                    loss_D1_writer.add_scalar('loss',engine.state.metrics['loss_D1'], engine.state.epoch)
                    loss_D2_writer.add_scalar('loss',engine.state.metrics['loss_D2'], engine.state.epoch)
                    loss_C_PI_writer.add_scalar('loss',engine.state.metrics['loss_C_P'], engine.state.epoch)
                    loss_C_OI_writer.add_scalar('loss',engine.state.metrics['loss_C_O'], engine.state.epoch)
                elif cfg.MODEL.OUTPUT_MODE == 'protected':
                    logger.info("Loss_G_gan: {:.3f}, Loss_G_l1: {:.3f}, Loss_R_l1: {:.3f}, Loss_D1: {:.3f}, Loss_D2: {:.3f}, Loss_C_P: {:.3f}"
                        .format(engine.state.metrics['loss_G_gan'],engine.state.metrics['loss_G_l1'],
                                engine.state.metrics['loss_R_l1'], engine.state.metrics['loss_D1'],  engine.state.metrics['loss_D2'],
                                engine.state.metrics['loss_C_P']))
                    loss_G_L1_writer.add_scalar('loss', engine.state.metrics['loss_G_l1'], engine.state.epoch)
                    loss_G_GAN_writer.add_scalar('loss', engine.state.metrics['loss_G_gan'], engine.state.epoch)
                    loss_G_R_L1_writer.add_scalar('loss',engine.state.metrics['loss_R_l1'], engine.state.epoch)
                    loss_D1_writer.add_scalar('loss',engine.state.metrics['loss_D1'], engine.state.epoch)
                    loss_D2_writer.add_scalar('loss',engine.state.metrics['loss_D2'], engine.state.epoch)
                    loss_C_PI_writer.add_scalar('loss',engine.state.metrics['loss_C_P'], engine.state.epoch)
                else:
                    logger.info("Loss_G_gan: {:.3f}, Loss_G_l1: {:.3f}, Loss_R_l1: {:.3f}, Loss_D1: {:.3f}, Loss_D2: {:.3f}, Loss_C_O: {:.3f}"
                        .format(engine.state.metrics['loss_G_gan'],engine.state.metrics['loss_G_l1'],
                                engine.state.metrics['loss_R_l1'], engine.state.metrics['loss_D1'],  engine.state.metrics['loss_D2'],
                                engine.state.metrics['loss_C_O']))
                    loss_G_L1_writer.add_scalar('loss', engine.state.metrics['loss_G_l1'], engine.state.epoch)
                    loss_G_GAN_writer.add_scalar('loss', engine.state.metrics['loss_G_gan'], engine.state.epoch)
                    loss_G_R_L1_writer.add_scalar('loss',engine.state.metrics['loss_R_l1'], engine.state.epoch)
                    loss_D1_writer.add_scalar('loss',engine.state.metrics['loss_D1'], engine.state.epoch)
                    loss_D2_writer.add_scalar('loss',engine.state.metrics['loss_D2'], engine.state.epoch)
                    loss_C_OI_writer.add_scalar('loss',engine.state.metrics['loss_C_O'], engine.state.epoch)
            elif cfg.MODEL.MODE == 'C':
                if cfg.MODEL.OUTPUT_MODE == 'both':
                    logger.info("Loss_C_O: {:.3f}, Loss_C_P: {:.3f}"
                        .format(engine.state.metrics['loss_C_O'], engine.state.metrics['loss_C_P']))
                    loss_C_PI_writer.add_scalar('loss',engine.state.metrics['loss_C_P'], engine.state.epoch)
                    loss_C_OI_writer.add_scalar('loss',engine.state.metrics['loss_C_O'], engine.state.epoch)
                elif cfg.MODEL.OUTPUT_MODE == 'protected':
                    logger.info("Loss_C_P: {:.3f}"
                        .format(engine.state.metrics['loss_C_P']))
                    loss_C_PI_writer.add_scalar('loss',engine.state.metrics['loss_C_P'], engine.state.epoch)
                else:
                    logger.info("Loss_C_O: {:.3f}"
                        .format(engine.state.metrics['loss_C_O']))
                    loss_C_OI_writer.add_scalar('loss',engine.state.metrics['loss_C_O'], engine.state.epoch)
        if len(data_loader['train']) == ITER:
            ITER = 0

    # adding handlers using `trainer.on` decorator API
    @trainer.on(Events.EPOCH_COMPLETED)
    def print_times(engine):
        logger.info('Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]'
                    .format(engine.state.epoch, timer.value() * timer.step_count,
                            data_loader['train'].batch_size / timer.value()))
        logger.info('-' * 10)
        timer.reset()

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        r1 = 0.0
        if engine.state.epoch % eval_period == 0:
            if cfg.TEST.PARTIAL_REID == 'off':
                evaluator.run(data_loader['eval'])
                cmc1, mAP1, mINP1, cmc2, mAP2, mINP2 = evaluator.state.metrics['r1_mAP_mINP']
                logger.info("Validation Results - Epoch: {}".format(engine.state.epoch))
                logger.info('test')
                if cfg.TEST.GALLERY_MODE == 'both':
                    logger.info('origin images:')
                    logger.info("mINP: {:.1%}".format(mINP1))
                    logger.info("mAP: {:.1%}".format(mAP1))
                    for r in [1, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc1[r - 1]))
                    logger.info('protected images:')
                    logger.info("mINP: {:.1%}".format(mINP2))
                    logger.info("mAP: {:.1%}".format(mAP2))
                    for r in [1, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc2[r - 1]))
                    r1_OI_writer.add_scalar('R1', cmc1[0], engine.state.epoch)
                    map_OI_writer.add_scalar('mAP', mAP1, engine.state.epoch)
                    minp_OI_writer.add_scalar('mINP', mINP1, engine.state.epoch)
                    r1_PI_writer.add_scalar('R1', cmc2[0], engine.state.epoch)
                    map_PI_writer.add_scalar('mAP', mAP2, engine.state.epoch)
                    minp_PI_writer.add_scalar('mINP', mINP2, engine.state.epoch)
                    r1 = cmc2[0]
                else:
                    logger.info('%s images'.format(cfg.TEST.GALLERY_MODE))
                    logger.info("mINP: {:.1%}".format(mINP1))
                    logger.info("mAP: {:.1%}".format(mAP1))
                    for r in [1, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc1[r - 1]))
                    r1_OI_writer.add_scalar('R1', cmc1[0], engine.state.epoch)
                    map_OI_writer.add_scalar('mAP', mAP1, engine.state.epoch)
                    minp_OI_writer.add_scalar('mINP', mINP1, engine.state.epoch)
                    r1 = cmc1[0]
            elif cfg.TEST.PARTIAL_REID == 'on':
                evaluator_reid.run(data_loader['eval_reid'])
                cmc1, mAP1, mINP1, cmc2, mAP2, mINP2 = evaluator_reid.state.metrics['r1_mAP_mINP']
                logger.info("Validation Results - Epoch: {}".format(engine.state.epoch))
                if cfg.TEST.GALLERY_MODE == 'both':
                    logger.info('origin images:')
                    logger.info("mINP: {:.1%}".format(mINP1))
                    logger.info("mAP: {:.1%}".format(mAP1))
                    for r in [1, 3, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc1[r - 1]))
                    logger.info('protected images:')
                    logger.info("mINP: {:.1%}".format(mINP2))
                    logger.info("mAP: {:.1%}".format(mAP2))
                    for r in [1, 3, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc2[r - 1]))
                    r1_OI_writer.add_scalar('R1', cmc1[0], engine.state.epoch)
                    map_OI_writer.add_scalar('mAP', mAP1, engine.state.epoch)
                    minp_OI_writer.add_scalar('mINP', mINP1, engine.state.epoch)
                    r1_PI_writer.add_scalar('R1', cmc2[0], engine.state.epoch)
                    map_PI_writer.add_scalar('mAP', mAP2, engine.state.epoch)
                    minp_PI_writer.add_scalar('mINP', mINP2, engine.state.epoch)
                    r1 = cmc2[0]
                else:
                    logger.info('%s images'.format(cfg.TEST.GALLERY_MODE))
                    logger.info("mINP: {:.1%}".format(mINP1))
                    logger.info("mAP: {:.1%}".format(mAP1))
                    for r in [1, 3, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc1[r - 1]))
                    r1_OI_writer.add_scalar('R1', cmc1[0], engine.state.epoch)
                    map_OI_writer.add_scalar('mAP', mAP1, engine.state.epoch)
                    minp_OI_writer.add_scalar('mINP', mINP1, engine.state.epoch)
                    r1 = cmc1[0]
        
            if r1 > test_metric['r1']:
                test_metric['r1'] = r1
                logger.info('saving the test best model (epoch %d)' % (engine.state.epoch))
                model.save_networks('epoch_' + 'best')


        if cfg.DATASETS.NAMES.split('_')[-1] == 'val':
            if engine.state.epoch % val_period == 0:
                if valing_results != None:
                    for k in valing_results:
                        valing_results[k] = 0

                evaluator_val.run(data_loader['val'])
                cmc1, mAP1, mINP1, cmc2, mAP2, mINP2 = evaluator_val.state.metrics['r1_mAP_mINP']
                logger.info("val: ")
                if cfg.TEST.GALLERY_MODE == 'both':
                    logger.info('origin images:')
                    logger.info("mINP: {:.1%}".format(mINP1))
                    logger.info("mAP: {:.1%}".format(mAP1))
                    for r in [1, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc1[r - 1]))
                    logger.info('protected images:')
                    logger.info("mINP: {:.1%}".format(mINP2))
                    logger.info("mAP: {:.1%}".format(mAP2))
                    for r in [1, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc2[r - 1]))
                    r1 = cmc2[0]
                else:
                    logger.info('%s images'.format(cfg.TEST.GALLERY_MODE))
                    logger.info("mINP: {:.1%}".format(mINP1))
                    logger.info("mAP: {:.1%}".format(mAP1))
                    for r in [1, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc1[r - 1]))
                    r1 = cmc1[0]
            
                # upgrade
                epsilon_psnr = cfg.MODEL.PSNR
                epsilon_ssim = cfg.MODEL.SSIM
                anonymous_enough = valing_results['psnr_OP'] < valing_results['psnr_OM'] + epsilon_psnr and valing_results['ssim_OP'] < valing_results['ssim_OM'] + epsilon_ssim
                if anonymous_enough and r1 > metric['r1_max']:
                    metric['r1_max'] = r1
                    logger.info('saving the val best model (epoch %d)' % (engine.state.epoch))
                    model.save_networks('epoch_' + 'val_best')

        if engine.state.epoch % eval_period == 0 or engine.state.epoch % val_period == 0: 
            # save the latest model
            logger.info('saving the latest model (epoch %d)' % (engine.state.epoch))
            model.save_networks('epoch_' + 'latest')
            if cfg.MODEL.MODE == 'GCR':
                logger.info('PSNR')
                logger.info('OM:{0} OP:{1} OR:{2}'.format(valing_results['psnr_OM'], valing_results['psnr_OP'], valing_results['psnr_OR']))
                logger.info('SSIM')
                logger.info('OM:{0} OP:{1} OR:{2}'.format(valing_results['ssim_OM'], valing_results['ssim_OP'], valing_results['ssim_OR']))
                logger.info("Saving images......")
                metric['psnr_OP'] = valing_results['psnr_OP']
                metric['ssim_OP'] = valing_results['ssim_OP']
                metric['psnr_OM'] = valing_results['psnr_OM']
                metric['ssim_OM'] = valing_results['ssim_OM']
                metric['r1'] = r1
                image_numpy = util.tensor2im(model.real_O[0], mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
                util.save_image(image_numpy, '{0}/{1}_real_O.jpg'.format(cfg.OUTPUT_DIR, engine.state.epoch))
                image_numpy = util.tensor2im(model.real_M[0], mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
                util.save_image(image_numpy, '{0}/{1}_real_M.jpg'.format(cfg.OUTPUT_DIR, engine.state.epoch))
                image_numpy = util.tensor2im(model.fake_P_encrypted[0], mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
                util.save_image(image_numpy, '{0}/{1}_fake_P_encrypted.jpg'.format(cfg.OUTPUT_DIR, engine.state.epoch))
                image_numpy = util.tensor2im(model.fake_P[0], mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
                util.save_image(image_numpy, '{0}/{1}_fake_P.jpg'.format(cfg.OUTPUT_DIR, engine.state.epoch))
                image_numpy = util.tensor2im(model.fake_R[0], mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
                util.save_image(image_numpy, '{0}/{1}_fake_R.jpg'.format(cfg.OUTPUT_DIR, engine.state.epoch))
            elif cfg.MODEL.MODE == 'C':
                logger.info('PSNR')
                logger.info('OM:{0}'.format(valing_results['psnr_OM']))
                logger.info('SSIM')
                logger.info('OM:{0}'.format(valing_results['ssim_OM']))
                logger.info("Saving images......")
                metric['psnr_OM'] = valing_results['psnr_OM']
                metric['ssim_OM'] = valing_results['ssim_OM']
                image_numpy = util.tensor2im(model.real_O[0], mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
                util.save_image(image_numpy, '{0}/{1}_real_O.jpg'.format(cfg.OUTPUT_DIR, engine.state.epoch))
                image_numpy = util.tensor2im(model.real_M[0], mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
                util.save_image(image_numpy, '{0}/{1}_real_M.jpg'.format(cfg.OUTPUT_DIR, engine.state.epoch))

    trainer.run(data_loader['train'], max_epochs=epochs)
