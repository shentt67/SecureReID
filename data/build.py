# encoding: utf-8

import torch
from torch.utils.data import DataLoader

from .datasets import init_dataset, ImageTrainDataset, ImageTestDataset
from .triplet_sampler import RandomIdentitySampler
from .transforms import build_transforms


def train_collate_fn(batch):
    imgs_src, imgs_dst, pids, keys, camids, img_path = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    return torch.stack(imgs_src, dim=0), torch.stack(imgs_dst, dim=0), pids, torch.stack(keys, dim=0), camids, img_path

def val_collate_fn(batch):
    imgs_src, imgs_dst, pids, keys, camids, img_path = zip(*batch)
    return torch.stack(imgs_src, dim=0), torch.stack(imgs_dst, dim=0), pids, torch.stack(keys, dim=0), camids, img_path

def make_data_loader(cfg):
    transforms = build_transforms(cfg)
    dataset = init_dataset(cfg.DATASETS.NAMES, root=cfg.DATASETS.ROOT_DIR)

    num_classes = dataset.num_train_pids
    num_workers = cfg.DATALOADER.NUM_WORKERS
    train_set = ImageTrainDataset(dataset.train, cfg.INPUT.DIRECTION, cfg.INPUT.TYPE, cfg.INPUT.RADIUS, transforms['train'], cfg.INPUT.IMG_SIZE)
    data_loader={}
    if cfg.DATALOADER.PK_SAMPLER == 'on':
        data_loader['train'] = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
            sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
            num_workers=num_workers, collate_fn=train_collate_fn
        )
    else:
        data_loader['train'] = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
            collate_fn=train_collate_fn
        )

    if cfg.TEST.PARTIAL_REID == 'off':
        eval_set = ImageTestDataset(dataset.query + dataset.gallery, cfg.INPUT.DIRECTION, cfg.INPUT.TYPE, cfg.INPUT.RADIUS, transforms['eval'], cfg.INPUT.IMG_SIZE)
        data_loader['eval'] = DataLoader(
            eval_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
            collate_fn=val_collate_fn 
        )
    else:
        dataset_reid = init_dataset('partial_reid', root=cfg.DATASETS.ROOT_DIR)
        dataset_ilids = init_dataset('partial_ilids', root=cfg.DATASETS.ROOT_DIR)
        eval_set_reid = ImageTestDataset(dataset_reid.query + dataset_reid.gallery, cfg.INPUT.DIRECTION, cfg.INPUT.TYPE, cfg.INPUT.RADIUS, transforms['eval'], cfg.INPUT.IMG_SIZE)
        data_loader['eval_reid'] = DataLoader(
            eval_set_reid, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
            collate_fn=val_collate_fn 
        )
        eval_set_ilids = ImageTestDataset(dataset_ilids.query + dataset_ilids.gallery, cfg.INPUT.DIRECTION, cfg.INPUT.TYPE, cfg.INPUT.RADIUS, transforms['eval'], cfg.INPUT.IMG_SIZE)
        data_loader['eval_ilids'] = DataLoader(
            eval_set_ilids, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
            collate_fn=val_collate_fn 
        )

    if cfg.DATASETS.NAMES.split('_')[-1] == 'val':
        val_set = ImageTestDataset(dataset.val_query + dataset.val_gallery, cfg.INPUT.DIRECTION, cfg.INPUT.TYPE, cfg.INPUT.RADIUS, transforms['eval'], cfg.INPUT.IMG_SIZE)
        data_loader['val'] = DataLoader(
            val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
            collate_fn=val_collate_fn
        )
        return data_loader, len(dataset.query), num_classes, len(dataset.val_query)

    return data_loader, len(dataset.query), num_classes
