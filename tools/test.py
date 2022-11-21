# encoding: utf-8
import logging

import torch
import torch.nn as nn
from ignite.engine import Engine

from utils.reid_metric import r1_mAP_mINP, r1_mAP_mINP_reranking
from utils import util
from math import log10
from utils.image_metric import ssim

import os


def create_supervised_evaluator(model, metrics, cfg, valing_results=None):
    """
    Factory function for creating an evaluator for supervised models

    Args:
        model (`torch.nn.Module`): the model to evaluate
        metrics (dict of str - :class:`ignite.metrics.Metric`): a map of metric names to Metrics
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.
    Returns:
        Engine: an evaluator engine with supervised inference function
    """

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            img_O, img_M, pids, deviation, camids, img_paths = batch

            deviation = deviation.to(device) if torch.cuda.device_count() >= 1 else deviation

            img_O = img_O.to(device) if torch.cuda.device_count() >= 1 else img_O

            img_M = img_M.to(device) if torch.cuda.device_count() >= 1 else img_M
            img_M = img_M.detach()

            feat_O, feat_P = model(img_O, img_M, deviation)

            # store anonymized image
            # if not os.path.exists('{0}/{1}'.format(cfg.OUTPUT_DIR, img_paths[0].split('/')[-2])):
            #     os.mkdir('{0}/{1}'.format(cfg.OUTPUT_DIR, img_paths[0].split('/')[-2]))
            # for i in range(len(model.fake_P)):
            #     if not os.path.exists('{0}/{1}'.format(cfg.OUTPUT_DIR, img_paths[i].split('/')[-2])):
            #         os.mkdir('{0}/{1}'.format(cfg.OUTPUT_DIR, img_paths[i].split('/')[-2]))
            #     image_numpy = util.tensor2im(model.fake_P[i], mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
            #     util.save_image(image_numpy, '{0}/{1}/{2}'.format(cfg.OUTPUT_DIR, img_paths[i].split('/')[-2], img_paths[i].split('/')[-1]))

            # store recovery images
            # if not os.path.exists('{0}/{1}'.format(cfg.OUTPUT_DIR, img_paths[0].split('/')[-2])):
            #     os.mkdir('{0}/{1}'.format(cfg.OUTPUT_DIR, img_paths[0].split('/')[-2]))
            # for i in range(len(model.fake_R)):
            #     if not os.path.exists('{0}/{1}'.format(cfg.OUTPUT_DIR, img_paths[i].split('/')[-2])):
            #         os.mkdir('{0}/{1}'.format(cfg.OUTPUT_DIR, img_paths[i].split('/')[-2]))
            #     image_numpy = util.tensor2im(model.fake_R[i], mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
            #     util.save_image(image_numpy, '{0}/{1}/{2}'.format(cfg.OUTPUT_DIR, img_paths[i].split('/')[-2], img_paths[i].split('/')[-1]))

            # store encrypted images
            # if not os.path.exists('{0}/{1}'.format(cfg.OUTPUT_DIR, img_paths[0].split('/')[-2])):
            #     os.mkdir('{0}/{1}'.format(cfg.OUTPUT_DIR, img_paths[0].split('/')[-2]))
            # for i in range(len(model.fake_P_encrypted)):
            #     if not os.path.exists('{0}/{1}'.format(cfg.OUTPUT_DIR, img_paths[i].split('/')[-2])):
            #         os.mkdir('{0}/{1}'.format(cfg.OUTPUT_DIR, img_paths[i].split('/')[-2]))
            #     image_numpy = util.tensor2im(model.fake_P_encrypted[i], mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
            #     util.save_image(image_numpy, '{0}/{1}/{2}'.format(cfg.OUTPUT_DIR, img_paths[i].split('/')[-2], img_paths[i].split('/')[-1]))

            # store decrypted images
            # if not os.path.exists('{0}/{1}'.format(cfg.OUTPUT_DIR, img_paths[0].split('/')[-2])):
            #     os.mkdir('{0}/{1}'.format(cfg.OUTPUT_DIR, img_paths[0].split('/')[-2]))
            # for i in range(len(model.fake_P_decrypted)):
            #     if not os.path.exists('{0}/{1}'.format(cfg.OUTPUT_DIR, img_paths[i].split('/')[-2])):
            #         os.mkdir('{0}/{1}'.format(cfg.OUTPUT_DIR, img_paths[i].split('/')[-2]))
            #     image_numpy = util.tensor2im(model.fake_P_decrypted[i], mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
            #     util.save_image(image_numpy, '{0}/{1}/{2}'.format(cfg.OUTPUT_DIR, img_paths[i].split('/')[-2], img_paths[i].split('/')[-1]))

            if valing_results != None:
                batch_size = batch[0].shape[0]
                valing_results['batch_sizes'] += batch_size
                # normalize 
                if mean != None and std != None:
                    img_OI = img_O.mul(std).add(mean)
                    img_MI = img_M.mul(std).add(mean)
                    if cfg.MODEL.MODE == 'GCR':
                        img_PI = model.fake_P.mul(std).add(mean)
                        img_RI = model.fake_R.mul(std).add(mean)

                if cfg.MODEL.MODE == 'GCR':
                    batch_mse_OM = ((img_OI - img_MI) ** 2).data.mean()
                    batch_mse_OP = ((img_OI - img_PI) ** 2).data.mean()
                    batch_mse_OR = ((img_OI - img_RI) ** 2).data.mean()
                    valing_results['mse_OM'] += batch_mse_OM * batch_size
                    valing_results['mse_OP'] += batch_mse_OP * batch_size
                    valing_results['mse_OR'] += batch_mse_OR * batch_size

                    batch_ssim_OM = ssim(img_OI, img_MI, device=device).data
                    batch_ssim_OP = ssim(img_OI, img_PI, device=device).data
                    batch_ssim_OR = ssim(img_OI, img_RI, device=device).data
                    valing_results['ssims_OM'] += batch_ssim_OM * batch_size
                    valing_results['ssims_OP'] += batch_ssim_OP * batch_size
                    valing_results['ssims_OR'] += batch_ssim_OR * batch_size

                    valing_results['psnr_OM'] = 10 * log10(1 / (valing_results['mse_OM'] / valing_results['batch_sizes']))
                    valing_results['psnr_OP'] = 10 * log10(1 / (valing_results['mse_OP'] / valing_results['batch_sizes']))
                    valing_results['psnr_OR'] = 10 * log10(1 / (valing_results['mse_OR'] / valing_results['batch_sizes']))
                    valing_results['ssim_OM'] = valing_results['ssims_OM'] / valing_results['batch_sizes']
                    valing_results['ssim_OP'] = valing_results['ssims_OP'] / valing_results['batch_sizes']
                    valing_results['ssim_OR'] = valing_results['ssims_OR'] / valing_results['batch_sizes']
                elif cfg.MODEL.MODE == 'C':
                    batch_mse_OM = ((img_OI - img_MI) ** 2).data.mean()
                    valing_results['mse_OM'] += batch_mse_OM * batch_size
                    batch_ssim_OM = ssim(img_OI, img_MI, device=device).data
                    valing_results['ssims_OM'] += batch_ssim_OM * batch_size
                    valing_results['psnr_OM'] = 10 * log10(1 / (valing_results['mse_OM'] / valing_results['batch_sizes']))
                    valing_results['ssim_OM'] = valing_results['ssims_OM'] / valing_results['batch_sizes']

            return feat_O, feat_P, pids, camids

    mean, std = cfg.INPUT.PIXEL_MEAN , cfg.INPUT.PIXEL_STD
    cnt = [0]
    num = 1000
    device = cfg.MODEL.DEVICE
    if valing_results != None:
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        if mean.ndim == 1:
            mean = mean.view(-1, 1, 1).to(device)
        if std.ndim == 1:
            std = std.view(-1, 1, 1).to(device)
    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


def do_test(
        cfg,
        model,
        data_loader,
        num_query
):
    # model.save_networks(cfg.MODEL.EPOCH)
    device = cfg.MODEL.DEVICE

    logger = logging.getLogger("reid_baseline")
    logger.info("Enter inferencing")
    valing_results = None
    if cfg.MODEL.MODE == 'GCR':
        valing_results = {'batch_sizes': 0, 'mse_OM': 0, 'ssims_OM': 0, 'psnr_OM': 0, 'ssim_OM': 0, \
            'mse_OP': 0, 'ssims_OP': 0, 'psnr_OP': 0, 'ssim_OP': 0, \
            'mse_OR': 0, 'ssims_OR': 0, 'psnr_OR': 0, 'ssim_OR': 0}
    elif cfg.MODEL.MODE == 'C':
        valing_results = {'batch_sizes': 0, 'mse_OM': 0, 'ssims_OM': 0, 'psnr_OM': 0, 'ssim_OM': 0}
    if cfg.TEST.RE_RANKING == 'off':
        print("Create evaluator")
        evaluator = create_supervised_evaluator(model, metrics={'r1_mAP_mINP': r1_mAP_mINP(num_query, gallery_mode=cfg.TEST.GALLERY_MODE, query_mode=cfg.TEST.QUERY_MODE, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)},
                                                cfg=cfg, valing_results=valing_results)
    elif cfg.TEST.RE_RANKING == 'on':
        print("Create evaluator for reranking")
        evaluator = create_supervised_evaluator(model, metrics={'r1_mAP_mINP': r1_mAP_mINP_reranking(num_query, gallery_mode=cfg.TEST.GALLERY_MODE, query_mode=cfg.TEST.QUERY_MODE, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)},
                                                cfg=cfg, valing_results=valing_results)
    else:
        print("Unsupported re_ranking config. Only support for on or off, but got {}.".format(cfg.TEST.RE_RANKING))

    evaluator.run(data_loader['eval'])

    cmc1, mAP1, mINP1, cmc2, mAP2, mINP2 = evaluator.state.metrics['r1_mAP_mINP']
    
    logger.info('Validation Results')
    if cfg.TEST.GALLERY_MODE == 'both':
        logger.info('origin images:')
        logger.info("mINP: {:.1%}".format(mINP1))
        logger.info("mAP: {:.1%}".format(mAP1))
        if cfg.TEST.PARTIAL_REID == 'off':
            for r in [1, 5, 10]:
                logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc1[r - 1]))
        else:
            for r in [1, 3, 5, 10]:
                logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc1[r - 1]))
        logger.info('protected images:')
        logger.info("mINP: {:.1%}".format(mINP2))
        logger.info("mAP: {:.1%}".format(mAP2))
        if cfg.TEST.PARTIAL_REID == 'off':
            for r in [1, 5, 10]:
                logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc2[r - 1]))
        else:
            for r in [1, 3, 5, 10]:
                logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc2[r - 1]))
    else:
        logger.info('%s images'.format(cfg.TEST.GALLERY_MODE))
        logger.info("mINP: {:.1%}".format(mINP1))
        logger.info("mAP: {:.1%}".format(mAP1))
        if cfg.TEST.PARTIAL_REID == 'off':
            for r in [1, 5, 10]:
                logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc1[r - 1]))
        else:
            for r in [1, 3, 5, 10]:
                logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc1[r - 1]))

    if cfg.MODEL.MODE == 'GCR':
        logger.info('PSNR')
        logger.info('OM:{0} OP:{1} OR:{2}'.format(valing_results['psnr_OM'], valing_results['psnr_OP'], valing_results['psnr_OR']))
        logger.info('SSIM')
        logger.info('OM:{0} OP:{1} OR:{2}'.format(valing_results['ssim_OM'], valing_results['ssim_OP'], valing_results['ssim_OR']))
