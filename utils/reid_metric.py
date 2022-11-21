# encoding: utf-8

import numpy as np
import torch
from ignite.metrics import Metric

from data.datasets.eval_reid import eval_func
from .re_ranking import re_ranking
import copy


class r1_mAP_mINP(Metric):
    def __init__(self, num_query, max_rank=50, feat_norm='on', gallery_mode='origin', query_mode='origin'):
        super(r1_mAP_mINP, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.gallery_mode = gallery_mode
        self.query_mode = query_mode

    def reset(self):
        self.feats_O = []
        self.feats_P = []
        self.pids = []
        self.camids = []

    def update(self, output):
        feat_O, feat_P, pid, camid = output
        self.feats_O.append(feat_O)
        self.feats_P.append(feat_P)
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))

    def compute(self):
        feats_o = torch.cat(self.feats_O, dim=0)
        feats_p = torch.cat(self.feats_P, dim=0)
        if self.gallery_mode == 'origin':
            feats_g_list = [feats_o]
        elif self.gallery_mode == 'protected':
            feats_g_list = [feats_p]
        elif self.gallery_mode == 'both':
            feats_g_list = [feats_o, feats_p]
        
        if self.query_mode == 'origin':
            feats_q_list = [feats_o]
        elif self.query_mode == 'protected':
            feats_q_list = [feats_p]
        # elif self.query_mode == 'both':
        #     feats_q_list = [feats_o, feats_p]

        if self.feat_norm == 'on':
            print("The test feature is normalized")
            for i in range(len(feats_q_list)):
                feats_q_list[i] = torch.nn.functional.normalize(feats_q_list[i], dim=1, p=2)

        for i in range(len(feats_q_list)):
            feats_q = feats_q_list[i]
            for j in range(len(feats_g_list)):
                feats_g = feats_g_list[j]
                if self.feat_norm == 'on':
                    feats_g = torch.nn.functional.normalize(feats_g, dim=1, p=2)
                # query
                qf = feats_q[:self.num_query].clone()
                q_pids = np.asarray(copy.deepcopy(self.pids[:self.num_query]))
                q_camids = np.asarray(copy.deepcopy(self.camids[:self.num_query]))
                # gallery
                gf = feats_g[self.num_query:]
                g_pids = np.asarray(copy.deepcopy(self.pids[self.num_query:]))
                g_camids = np.asarray(copy.deepcopy(self.camids[self.num_query:]))
                m, n = qf.shape[0], gf.shape[0]
                distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                        torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
                distmat.addmm_(1, -2, qf, gf.t())
                distmat = distmat.cpu().numpy()
                if len(feats_g_list) == 1 and len(feats_q_list) == 1:
                    cmc1, mAP1, mINP1 = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)
                    cmc2 = mAP2 = mINP2 = 0.0
                else:
                    if i == 0 and j == 0:
                        cmc1, mAP1, mINP1 = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)
                    elif i == 1 or j == 1:
                        cmc2, mAP2, mINP2 = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)

        return cmc1, mAP1, mINP1, cmc2, mAP2, mINP2


class r1_mAP_mINP_reranking(Metric):
    def __init__(self, num_query, max_rank=50, feat_norm='on', gallery_mode='origin'):
        super(r1_mAP_mINP_reranking, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.gallery_mode = gallery_mode

    def reset(self):
        self.feats_O = []
        self.feats_P = []
        self.pids = []
        self.camids = []

    def update(self, output):
        feat_O, feat_P, pid, camid = output
        self.feats_O.append(feat_O)
        self.feats_P.append(feat_P)
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))

    def compute(self):
        feats_q = torch.cat(self.feats_O, dim=0)
        if self.gallery_mode == 'origin':
            feats_g = [feats_q]
        elif self.gallery_mode == 'protected':
            feats_g = [torch.cat(self.feats_P, dim=0)]
        elif self.gallery_mode == 'both':
            feats_g_list = [feats_q.clone(), torch.cat(self.feats_P, dim=0)]
        
        if self.feat_norm == 'on':
            print("The test feature is normalized")
            feats_q = torch.nn.functional.normalize(feats_q, dim=1, p=2)
        for i in range(len(feats_g_list)):
            feats_g = feats_g_list[i]
            if self.feat_norm == 'on':
                feats_g = torch.nn.functional.normalize(feats_g, dim=1, p=2)
            # query
            qf = feats_q[:self.num_query].clone()
            q_pids = np.asarray(copy.deepcopy(self.pids[:self.num_query]))
            q_camids = np.asarray(copy.deepcopy(self.camids[:self.num_query]))
            # gallery
            gf = feats_g[self.num_query:]
            g_pids = np.asarray(copy.deepcopy(self.pids[self.num_query:]))
            g_camids = np.asarray(copy.deepcopy(self.camids[self.num_query:]))
            # m, n = qf.shape[0], gf.shape[0]
            # distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
            #           torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
            # distmat.addmm_(1, -2, qf, gf.t())
            # distmat = distmat.cpu().numpy()
            print("Enter reranking")
            distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
            if len(feats_g_list) == 1:
                cmc1, mAP1, mINP1 = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)
                cmc2 = mAP2 = mINP2 = 0
            else:
                if i == 0:
                    cmc1, mAP1, mINP1 = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)
                if i == 1:
                    cmc2, mAP2, mINP2 = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)

        return cmc1, mAP1, mINP1, cmc2, mAP2, mINP2