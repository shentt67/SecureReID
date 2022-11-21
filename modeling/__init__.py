# encoding: utf-8

from .baseline import Baseline
from .privacy_model import PrivacyModel

def build_model(cfg, num_classes):
    # if cfg.TEST.EVALUATE_ONLY == 'on':
    #     return build_test_query_model(cfg, num_classes)
    # else:
    return build_train_model(cfg, num_classes)

def build_test_query_model(cfg, num_classes):
    model = Baseline(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NAME,
                     cfg.MODEL.GENERALIZED_MEAN_POOL, cfg.MODEL.PRETRAIN_CHOICE)
    return model

def build_train_model(cfg, num_classes):
    model = PrivacyModel(cfg, num_classes)
    return model

