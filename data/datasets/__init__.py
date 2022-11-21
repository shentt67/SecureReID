# encoding: utf-8

from .cuhk03 import CUHK03
from .dukemtmcreid import DukeMTMCreID
from .market1501 import Market1501
from .market1501_val import Market1501Val
from .msmt17 import MSMT17
from .msmt17_val import MSMT17VAL
from .cuhk03_val import CUHK03VAL
from .veri import VeRi
from .partial_ilids import PartialILIDS
from .partial_reid import PartialREID
from .dataset_loader import ImageTrainDataset, ImageTestDataset

__factory = {
    'market1501': Market1501,
    'market1501_val': Market1501Val,
    'cuhk03': CUHK03,
    'cuhk03_val': CUHK03VAL,
    'dukemtmc': DukeMTMCreID,
    'msmt17': MSMT17,
    'msmt17_val': MSMT17VAL,
    'veri': VeRi,
    'partial_reid' : PartialREID,
    'partial_ilids' : PartialILIDS,
}


def get_names():
    return __factory.keys()


def init_dataset(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown datasets: {}".format(name))
    return __factory[name](*args, **kwargs)
