# Copyright (c) Facebook, Inc. and its affiliates.
import functools
import logging
import math
import torch
import numpy
import random
import torch.nn.functional as F
import numpy as np
# import clip
from mmf.common.registry import registry
from mmf.models.base_model import BaseModel
from mmf.modules.layers import ClassifierLayer
from mmf.utils.build import build_image_encoder
from omegaconf import OmegaConf
from torch import nn
from mmf.models.gcn import PyGCN
from transformers.modeling_bert import (
    BertConfig,
    BertLayerNorm
)
import json
from sklearn.cluster import DBSCAN
from mmf.models.mr_gcn import RoleRGCNEncoder, RGCNEncoderConfig
from mmf.modules.henG import Graph_reasoning

logger = logging.getLogger(__name__)


# @registry.register_model("m4c")
class M4C(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.mmt_config = BertConfig(**self.config.mmt)
        self._datasets = registry.get("config").datasets.split(",")
        self._use_clip = True
        self._use_myclip = True
        self._use_clip_features = True
        self._use_large_clip = False
        self.use_gcn = True
        self.use_graph_reasoning = False
        self.use_cluster = False

    @classmethod
    def config_path(cls):
        return "configs/models/m4c/defaults.yaml"

    @classmethod
    def format_state_key(cls, key):
        key = key.replace("obj_faster_rcnn_fc7.module.lc", "obj_faster_rcnn_fc7.lc")
        key = key.replace("ocr_faster_rcnn_fc7.module.lc", "ocr_faster_rcnn_fc7.lc")
        return key

    def build(self):
        self.mmt_config.hidden_size = 1000
        # modules requiring custom learning rates (usually for finetuning)
        self.finetune_modules = []

        # split model building into several components
        self._build_obj_encoding()
        if self._use_clip_features:
            self._build_ocr_encoding_clip_features()
        elif self._use_clip:
            self._build_ocr_encoding_clip()
        else:
            self._build_ocr_encoding()
        if self.use_graph_reasoning:
            self._build_graph_reasoning_encoding()

        self._build_mma_sr()
        self._build_output()

    def _build_encoder_config(self):
        return OmegaConf.create(
            {
                "type": "finetune_faster_rcnn_fpn_fc7",
                "params": {
                    "in_dim": 2048,
                    "weights_file": "models/detectron.defaults/fc7_w.pkl",
                    "bias_file": "models/detectron.defaults/fc7_b.pkl",
                    "model_data_dir": self.config.model_data_dir,
                },
            }
        )

    def _build_obj_encoding(self):
        # object appearance feature: Faster R-CNN
        self.obj_faster_rcnn_fc7 = build_image_encoder(
            self._build_encoder_config(), direct_features=True
        )
        # apply smaller lr to pretrained Faster R-CNN fc7
        self.finetune_modules.append(
            {"module": self.obj_faster_rcnn_fc7, "lr_scale": self.config.lr_scale_frcn}
        )
        # self.linear_obj_feat_to_mmt_in = nn.Linear(
        #     self.config.obj.mmt_in_dim, self.mmt_config.hidden_size
        # )
        self.linear_obj_feat_to_mmt_in = nn.Linear(
            512, self.mmt_config.hidden_size
        )
        self.obj_feat_layer_norm = BertLayerNorm(self.mmt_config.hidden_size)
        self.obj_drop = nn.Dropout(self.config.obj.dropout_prob)
        self.relu = nn.ReLU()

    def _build_ocr_encoding(self):

        self.remove_ocr_fasttext = getattr(self.config.ocr, "remove_ocr_fasttext", False)
        self.remove_ocr_phoc = getattr(self.config.ocr, "remove_ocr_phoc", False)
        self.remove_ocr_frcn = getattr(self.config.ocr, "remove_ocr_frcn", False)
        self.remove_ocr_semantics = getattr(
            self.config.ocr, "remove_ocr_semantics", False
        )
        self.remove_ocr_bbox = getattr(self.config.ocr, "remove_ocr_bbox", False)

        # OCR appearance feature: Faster R-CNN
        self.ocr_faster_rcnn_fc7 = build_image_encoder(
            self._build_encoder_config(), direct_features=True
        )
        self.finetune_modules.append(
            {"module": self.ocr_faster_rcnn_fc7, "lr_scale": self.config.lr_scale_frcn}
        )

        self.linear_ocr_feat_to_mmt_in = nn.Linear(
            self.config.ocr.mmt_in_dim - 50, self.mmt_config.hidden_size
        )
        # OCR location feature: relative bounding box coordinates (4-dim)
        self.linear_ocr_bbox_to_mmt_in = nn.Linear(4, self.mmt_config.hidden_size)

        self.ocr_feat_layer_norm = BertLayerNorm(self.mmt_config.hidden_size)
        self.ocr_bbox_layer_norm = BertLayerNorm(self.mmt_config.hidden_size)
        if self.use_cluster:
            self.ocr_pos_to_mmt_in = nn.Linear(60, self.mmt_config.hidden_size)
            self.ocr_pos_layer_norm = BertLayerNorm(self.mmt_config.hidden_size)

        self.ocr_drop = nn.Dropout(self.config.ocr.dropout_prob)


    def _build_ocr_encoding_clip(self):

        self.remove_ocr_fasttext = getattr(self.config.ocr, "remove_ocr_fasttext", False)
        self.remove_ocr_phoc = getattr(self.config.ocr, "remove_ocr_phoc", False)
        self.remove_ocr_frcn = getattr(self.config.ocr, "remove_ocr_frcn", False)
        self.remove_ocr_semantics = getattr(
            self.config.ocr, "remove_ocr_semantics", False
        )
        self.remove_ocr_bbox = getattr(self.config.ocr, "remove_ocr_bbox", False)

        # OCR appearance feature: Faster R-CNN
        self.ocr_faster_rcnn_fc7 = build_image_encoder(
            self._build_encoder_config(), direct_features=True
        )
        self.finetune_modules.append(
            {"module": self.ocr_faster_rcnn_fc7, "lr_scale": self.config.lr_scale_frcn}
        )

        self.linear_ocr_feat_to_mmt_in = nn.Linear(
            300+604, self.mmt_config.hidden_size
        )
        # OCR location feature: relative bounding box coordinates (4-dim)
        self.linear_ocr_bbox_to_mmt_in = nn.Linear(2048+4, self.mmt_config.hidden_size)

        self.ocr_feat_layer_norm = BertLayerNorm(self.mmt_config.hidden_size)
        self.ocr_bbox_layer_norm = BertLayerNorm(self.mmt_config.hidden_size)
        if self.use_cluster:
            self.ocr_pos_to_mmt_in = nn.Linear(60, self.mmt_config.hidden_size)
            self.ocr_pos_layer_norm = BertLayerNorm(self.mmt_config.hidden_size)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.ocr_drop = nn.Dropout(self.config.ocr.dropout_prob)

    def _build_ocr_encoding_clip_features(self):

        self.remove_ocr_fasttext = getattr(self.config.ocr, "remove_ocr_fasttext", False)
        self.remove_ocr_phoc = getattr(self.config.ocr, "remove_ocr_phoc", False)
        self.remove_ocr_frcn = getattr(self.config.ocr, "remove_ocr_frcn", False)
        self.remove_ocr_semantics = getattr(
            self.config.ocr, "remove_ocr_semantics", False
        )
        self.remove_ocr_bbox = getattr(self.config.ocr, "remove_ocr_bbox", False)

        # OCR appearance feature: Faster R-CNN
        self.ocr_faster_rcnn_fc7 = build_image_encoder(
            self._build_encoder_config(), direct_features=True
        )
        self.finetune_modules.append(
            {"module": self.ocr_faster_rcnn_fc7, "lr_scale": self.config.lr_scale_frcn}
        )

        self.linear_ocr_feat_to_mmt_in = nn.Linear(
            512, self.mmt_config.hidden_size
        )
        # OCR location feature: relative bounding box coordinates (4-dim)
        self.linear_ocr_bbox_to_mmt_in = nn.Linear(512, self.mmt_config.hidden_size)

        self.ocr_feat_layer_norm = BertLayerNorm(self.mmt_config.hidden_size)
        self.ocr_bbox_layer_norm = BertLayerNorm(self.mmt_config.hidden_size)
        if self.use_cluster:
            self.ocr_pos_to_mmt_in = nn.Linear(60, self.mmt_config.hidden_size)
            self.ocr_pos_layer_norm = BertLayerNorm(self.mmt_config.hidden_size)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.ocr_drop = nn.Dropout(self.config.ocr.dropout_prob)


    def _build_graph_encoding(self):
        # self.gcn = PyGCN(1000, 2048, 1000)
        self.gcn = RoleRGCNEncoder(RGCNEncoderConfig())
        self.obj_graph = BertLayerNorm(self.mmt_config.hidden_size)
        self.ocr_graph = BertLayerNorm(self.mmt_config.hidden_size)

  
    def build_ocr_next(self, sample_list, fwd_results):
        # Get the prepared features
        ocr_fasttext = sample_list.context_feature_0
        # OCR appearance feature: Faster R-CNN fc7
        ocr_fc6 = sample_list.image_feature_1[:, : ocr_fasttext.size(1), :]
        ocr_fc7 = self.ocr_faster_rcnn_fc7(ocr_fc6)
        ocr_fc7 = F.normalize(ocr_fc7, dim=-1)

        if self.remove_ocr_fasttext:
            ocr_fasttext = torch.zeros_like(ocr_fasttext)

        if self.remove_ocr_frcn:
            ocr_fc7 = torch.zeros_like(ocr_fc7)

        ocr_box = sample_list.ocr_bbox_coordinates.float()
        device = ocr_box.device
        batch_size = ocr_box.size(0)
        ocr_is_substr_tensor = torch.tensor(sample_list.ocr_is_substr, device=device)

        if self.remove_ocr_bbox:
            ocr_bbox = torch.zeros_like(ocr_box).float32()
        remove_substr = True

        # Calculate the next box for each ocr box
        max_iou = 0.3
        max_dist = 0.1
        min_sim = 0.5
        # Initialize the array of next ocr with -1. eg.[[-1,-1,...,-1]]
        next_ocr = torch.zeros([batch_size, ocr_bbox.size(1)], device=device)
        next_ocr = next_ocr-1
        ocr_nums = sample_list.context_info_0.max_features

        # Serval functions to calculate the IOU, distance between two boxes.
        def iou(bbox1, bbox2):
            xmin1, ymin1, xmax1, ymax1 = bbox1
            xmin2, ymin2, xmax2, ymax2 = bbox2

            # get the coordination of rectangle 获取矩形框交加对应的顶点坐标(intersection)
            xx1 = max([xmin1, xmin2])
            yy1 = max([ymin1, ymin2])
            xx2 = min([xmax1, xmax2])
            yy2 = min([ymax1, ymax2])
            # 计算交加面积
            inter_area = max([0, xx2 - xx1]) * max([0, yy2 - yy1])

            # # 计算两个矩形框面积
            area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
            area2 = (xmax2 - xmin2) * (ymax2 - ymin2)
            # iou = inter_area / area2

            # 计算交并比（交加/并集）
            iou = inter_area / (area1 + area2 - inter_area)  # 留神：这里inter_area不能乘以2，乘以2就相当于把交加局部挖空了
            return iou

        def dist(box1, box2):
            ax1, ay1, ax2, ay2 = box1
            bx1, by1, bx2, by2 = box2

            mid_ax = (ax2 + ax1) / 2
            mid_ay = (ay2 + ay1) / 2

            mid_bx = (bx2 + bx1) / 2
            mid_by = (by2 + by1) / 2

            sum_w = (ax2 - ax1 + bx2 - bx1) / 2
            sum_h = (ay2 - ay1 + by2 - by1) / 2

            Dx = math.fabs(mid_ax - mid_bx)
            Dy = math.fabs(mid_ay - mid_by)

            if Dx < sum_w and Dy >= sum_h:
                min_dist = Dy - sum_h
            elif Dx >= sum_w and Dy < sum_h:
                min_dist = Dx - sum_w
            elif Dx >= sum_w and Dy >= sum_h:
                delta_x = Dx - sum_w
                delta_y = Dy - sum_h
                min_dist = math.sqrt(delta_x * delta_x + delta_y * delta_y)
            else:
                min_dist = 0

            return min_dist

        def judge_next(ocr_box1, ocr_box2):

            return (iou(ocr_box1, ocr_box2) > max_iou
                    and dist(ocr_box1, ocr_box2) < max_dist)

        # Calculate the next box for each ocr box step by step
        for i in range(batch_size):
            ocr_box_per_batch = ocr_box[i]
            for j in range(ocr_nums[i]):
                for k in range(j + 1, ocr_nums[i]):
                    if judge_next(ocr_box_per_batch[j], ocr_box_per_batch[k]):
                        next_ocr[i, j] = k

        fwd_results["ocr_next"] = next_ocr


    def _build_mma_sr(self):
        hidden_size = 1000
        if self.use_gcn:
            self._build_graph_encoding()
        self.lstm_r = LSTM_R(decoder_dim=hidden_size,
                             obj_dim=hidden_size,
                             ocr_dim=hidden_size,
                             emb_dim=hidden_size,
                             attention_dim=hidden_size,
                             mmt_config=self.mmt_config)

    def _build_output(self):
        self.answer_processor = registry.get(self._datasets[0] + "_answer_processor")


    def forward(self, sample_list):
        # fwd_results holds intermediate forward pass results
        # TODO possibly replace it with another sample list

        fwd_results = {}
        if self._use_clip:
            self._forward_obj_clip_features_encoding_clip(sample_list, fwd_results)
        else:
            self._forward_obj_encoding(sample_list, fwd_results)
        if self._use_clip:
            self._forward_ocr_encoding_clip_feature(sample_list, fwd_results)
        else:
            self._forward_ocr_encoding(sample_list, fwd_results)

        if self.use_gcn:
            self._forward_mr_gcn_encoding(sample_list, fwd_results)
        self._foward_lstm_bsl(sample_list, fwd_results)
        results = {"scores": fwd_results["scores"]}
        results["prev_inds"] = fwd_results["prev_inds"]
        results["ocr_num"] = torch.sum(fwd_results["ocr_mask"], dim=-1)
        if self._use_clip:
            results["clip_mask"] = fwd_results["ocr_mask"]
            results["clip_scores"] = fwd_results["clip_scores"]
            if self._use_large_clip:
                results["clip_mask"] = fwd_results["ocr_mask"].\
                    view(fwd_results["ocr_mask"].size(0)*fwd_results["ocr_mask"].size(1), -1)

        return results

    def _forward_obj_encoding(self, sample_list, fwd_results):
        # object appearance feature: Faster R-CNN fc7
        obj_fc6 = sample_list.image_feature_0
        obj_fc7 = self.obj_faster_rcnn_fc7(obj_fc6)
        obj_fc7 = F.normalize(obj_fc7, dim=-1)
        obj_feat = obj_fc7

        obj_mmt_in = self.relu(self.obj_feat_layer_norm(self.linear_obj_feat_to_mmt_in(obj_feat)))
        obj_mmt_in = self.obj_drop(obj_mmt_in)
        fwd_results["obj_mmt_in"] = obj_mmt_in

        # binary mask of valid object vs padding
        obj_nums = sample_list.image_info_0.max_features
        # obj_nums = 36
        fwd_results["obj_mask"] = _get_mask(obj_nums, obj_mmt_in.size(1))  # * master_obj_mask

    def _forward_obj_encoding_clip(self, sample_list, fwd_results):
        # object appearance feature: Faster R-CNN fc7
        obj_fc6 = sample_list.image_feature_0
        obj_fc7 = self.obj_faster_rcnn_fc7(obj_fc6)
        obj_fc7 = F.normalize(obj_fc7, dim=-1)

        obj_feat = obj_fc7
        # obj_bbox = sample_list.obj_bbox_coordinates
        obj_unormalized = self.linear_obj_feat_to_mmt_in(obj_feat)
        if self._use_myclip:
            obj_mmt_in = self.relu(self.obj_feat_layer_norm(obj_unormalized))
            obj_mmt_in = self.obj_drop(obj_mmt_in)
        elif self._use_large_clip:
            obj_mmt_in = self.relu(self.obj_feat_layer_norm(obj_unormalized))
            obj_mmt_in = self.obj_drop(obj_mmt_in)
        else:
            obj_mmt_in_undrop = obj_unormalized / obj_unormalized.norm(dim=1, keepdim=True)
            obj_mmt_in = self.obj_drop(obj_mmt_in_undrop)

        if self.use_gcn:
            master_obj_mask1 = torch.tensor(sample_list.master_obj_mask, device=obj_fc7.device)
            master_obj_mask = (master_obj_mask1 != -1).long()
            # master_obj_mask = torch.tensor(master_obj_mask1, device=obj_fc7.device)

        # binary mask of valid object vs padding
        obj_nums = sample_list.image_info_0.max_features

        fwd_results["obj_mmt_in"] = obj_mmt_in
        fwd_results["obj_mask"] = _get_mask(obj_nums, obj_mmt_in.size(1))  # * master_obj_mask

    def _forward_obj_clip_features_encoding_clip(self, sample_list, fwd_results):
        # object appearance feature: Faster R-CNN fc7
        obj_fc6 = sample_list.image_feature_0
        obj_feat = F.normalize(obj_fc6, dim=-1)
        obj_unormalized = self.linear_obj_feat_to_mmt_in(obj_feat)
        if self._use_myclip:
            obj_mmt_in = self.relu(self.obj_feat_layer_norm(obj_unormalized))
            obj_mmt_in = self.obj_drop(obj_mmt_in)
        elif self._use_large_clip:
            obj_mmt_in = self.relu(self.obj_feat_layer_norm(obj_unormalized))
            obj_mmt_in = self.obj_drop(obj_mmt_in)
        else:
            obj_mmt_in_undrop = obj_unormalized / obj_unormalized.norm(dim=1, keepdim=True)
            obj_mmt_in = self.obj_drop(obj_mmt_in_undrop)

        if self.use_gcn:
            master_obj_mask1 = torch.tensor(sample_list.master_obj_mask, device=obj_fc6.device)
            master_obj_mask = (master_obj_mask1 != -1).long()
            # master_obj_mask = torch.tensor(master_obj_mask1, device=obj_fc7.device)

        # binary mask of valid object vs padding
        obj_nums = sample_list.image_info_0.max_features

        fwd_results["obj_mmt_in"] = obj_mmt_in
        fwd_results["obj_mask"] = _get_mask(obj_nums, obj_mmt_in.size(1))  # * master_obj_mask

    def _forward_ocr_encoding_clip(self, sample_list, fwd_results):
        # OCR FastText feature (300-dim)
        ocr_fasttext = sample_list.context_feature_0
        ocr_fasttext = F.normalize(ocr_fasttext, dim=-1)
        assert ocr_fasttext.size(-1) == 300

        # OCR PHOC feature (604-dim)
        ocr_phoc = sample_list.context_feature_1
        ocr_phoc = F.normalize(ocr_phoc, dim=-1)
        assert ocr_phoc.size(-1) == 604

        # OCR appearance feature: Faster R-CNN fc7
        ocr_fc6 = sample_list.image_feature_1[:, : ocr_fasttext.size(1), :]
        ocr_fc7 = self.ocr_faster_rcnn_fc7(ocr_fc6)
        ocr_fc7 = F.normalize(ocr_fc7, dim=-1)

        ocr_feat_unormalized = self.linear_ocr_feat_to_mmt_in(torch.cat([ocr_fasttext, ocr_phoc], dim=-1))
        ocr_text_unormalized = self.linear_ocr_bbox_to_mmt_in(torch.cat([ocr_fc7, sample_list.ocr_bbox_coordinates.float()],dim=-1))
        if self._use_clip_features:
            logit_scale = self.logit_scale.exp()
            # ocr_feat_unormalized = self.clip_model.encode_image(ocr_feat_unormalized)
            ocr_feat = self.relu(self.ocr_feat_layer_norm(ocr_feat_unormalized))
            ocr_text = self.relu(self.ocr_bbox_layer_norm(ocr_text_unormalized))
            ocr_mmt_in = ocr_feat + ocr_text
        elif self._use_myclip:
            logit_scale = self.logit_scale.exp()
            ocr_feat = self.relu(self.ocr_feat_layer_norm(ocr_feat_unormalized))
            ocr_text = self.relu(self.ocr_bbox_layer_norm(ocr_text_unormalized))
            ocr_mmt_in = ocr_feat + ocr_text
        elif self._use_large_clip:
            ocr_feat = self.relu(self.ocr_feat_layer_norm(ocr_feat_unormalized))
            ocr_text = self.relu(self.ocr_bbox_layer_norm(ocr_text_unormalized))
            ocr_mmt_in = ocr_feat + ocr_text
        else:
            logit_scale = self.logit_scale.exp()
            ocr_feat_norm = ocr_feat_unormalized / ocr_feat_unormalized.norm(dim=1, keepdim=True)
            ocr_text_norm = ocr_text_unormalized / ocr_text_unormalized.norm(dim=1, keepdim=True)
            ocr_mmt_in = ocr_feat_norm + ocr_text_norm

        if self.use_cluster:
            ocr_pos = self._get_ocr_clusterEmbedding(sample_list)
            # ocr_mmt_in += self.relu(self.ocr_pos_layer_norm(self.ocr_pos_to_mmt_in(ocr_pos)))
            ocr_mmt_in = self.relu(self.ocr_pos_layer_norm(ocr_mmt_in + ocr_pos))

        if self._use_myclip:
            sim_score = ocr_feat.matmul(ocr_text.transpose(-1, -2))
            # sim_score = torch.cdist(ocr_feat, ocr_text, p=2)
            # sim_score = permute_score
            fwd_results["clip_scores"] = sim_score
        elif self._use_large_clip:
                permute_ocr_feat = ocr_feat.view(ocr_feat.size(0)*ocr_feat.size(1), -1)
                permute_ocr_text = ocr_text.view(ocr_text.size(0)*ocr_text.size(1), -1)
                permute_score = permute_ocr_feat.matmul(permute_ocr_text.transpose(0, 1))
                sim_score = permute_score
                fwd_results["clip_scores"] = sim_score
        else:
            sim_score = logit_scale * ocr_feat_norm.matmul(ocr_text_norm.transpose(-1,-2))
            fwd_results["clip_scores"] = sim_score

        # binary mask of valid OCR vs padding
        ocr_nums = sample_list.context_info_0.max_features
        fwd_results["ocr_mask"] = _get_mask(ocr_nums, ocr_mmt_in.size(1))  # ocr_is_substr_mask

        remove_substr = False
        if remove_substr:
            ocr_is_substr_tensor = torch.tensor(sample_list.ocr_is_substr, device=ocr_fc7.device)
            ocr_is_substr_mask = (ocr_is_substr_tensor == 0).long()
            fwd_results["ocr_mask"] *= ocr_is_substr_mask

        ocr_mmt_in = self.ocr_drop(ocr_mmt_in)
        fwd_results["ocr_mmt_in"] = ocr_mmt_in


    def _forward_ocr_encoding_clip_feature(self, sample_list, fwd_results):
        normalize = False
        # OCR appearance feature: RN101
        ocr_feature = sample_list.image_feature_1
        ocr_feat_unormalized = ocr_feature[:, :50, :512]
        ocr_text_unormalized = ocr_feature[:, :50, 512:]
        ocr_feat_unormalized = self.linear_ocr_feat_to_mmt_in(ocr_feat_unormalized)
        ocr_text_unormalized = self.linear_ocr_bbox_to_mmt_in(ocr_text_unormalized)

        if normalize:
            ocr_feat_unormalized = F.normalize(ocr_feat_unormalized, dim=-1)
            ocr_text_unormalized = F.normalize(ocr_text_unormalized, dim=-1)

        # logit_scale = self.logit_scale.exp()

        ocr_feat = self.relu(self.ocr_feat_layer_norm(ocr_feat_unormalized))
        ocr_text = self.relu(self.ocr_bbox_layer_norm(ocr_text_unormalized))
        ocr_mmt_in = ocr_feat + ocr_text

        if self.use_cluster:
            ocr_pos = self._get_ocr_clusterEmbedding(sample_list)
            # ocr_mmt_in += self.relu(self.ocr_pos_layer_norm(self.ocr_pos_to_mmt_in(ocr_pos)))
            ocr_mmt_in = self.relu(self.ocr_pos_layer_norm(ocr_mmt_in + ocr_pos))

        sim_score = ocr_feat.matmul(ocr_text.transpose(-1, -2))
        # sim_score = torch.cdist(ocr_feat, ocr_text, p=2)
        # sim_score = permute_score
        fwd_results["clip_scores"] = sim_score

        # binary mask of valid OCR vs padding
        ocr_nums = sample_list.context_info_0.max_features
        fwd_results["ocr_mask"] = _get_mask(ocr_nums, ocr_mmt_in.size(1))  # ocr_is_substr_mask

        remove_substr = False
        if remove_substr:
            ocr_is_substr_tensor = torch.tensor(sample_list.ocr_is_substr, device=ocr_feature.device)
            ocr_is_substr_mask = (ocr_is_substr_tensor == 0).long()
            fwd_results["ocr_mask"] *= ocr_is_substr_mask

        ocr_mmt_in = self.ocr_drop(ocr_mmt_in)
        fwd_results["ocr_mmt_in"] = ocr_mmt_in



    def _forward_ocr_encoding(self, sample_list, fwd_results):
        # OCR FastText feature (300-dim)
        ocr_fasttext = sample_list.context_feature_0
        ocr_fasttext = F.normalize(ocr_fasttext, dim=-1)
        assert ocr_fasttext.size(-1) == 300

        # OCR PHOC feature (604-dim)
        ocr_phoc = sample_list.context_feature_1
        ocr_phoc = F.normalize(ocr_phoc, dim=-1)
        assert ocr_phoc.size(-1) == 604

        # OCR appearance feature: Faster R-CNN fc7
        ocr_fc6 = sample_list.image_feature_1[:, : ocr_fasttext.size(1), :]
        ocr_fc7 = self.ocr_faster_rcnn_fc7(ocr_fc6)
        ocr_fc7 = F.normalize(ocr_fc7, dim=-1)

        if self.remove_ocr_fasttext:
            ocr_fasttext = torch.zeros_like(ocr_fasttext)
        if self.remove_ocr_phoc:
            ocr_phoc = torch.zeros_like(ocr_phoc)
        if self.remove_ocr_frcn:
            ocr_fc7 = torch.zeros_like(ocr_fc7)
        ocr_feat = torch.cat(
            [ocr_fasttext, ocr_phoc, ocr_fc7], dim=-1
        )
        ocr_bbox = sample_list.ocr_bbox_coordinates.float()

        if self.remove_ocr_semantics:
            ocr_feat = torch.zeros_like(ocr_feat)
        if self.remove_ocr_bbox:
            ocr_bbox = torch.zeros_like(ocr_bbox).float32()

        ocr_mmt_in = self.relu(self.ocr_feat_layer_norm(self.linear_ocr_feat_to_mmt_in(ocr_feat))) \
                     + self.relu(self.ocr_bbox_layer_norm(self.linear_ocr_bbox_to_mmt_in(ocr_bbox)))

        if self.use_cluster:
            ocr_pos = self._get_ocr_clusterEmbedding(sample_list)
            #ocr_mmt_in += self.relu(self.ocr_pos_layer_norm(self.ocr_pos_to_mmt_in(ocr_pos)))
            ocr_mmt_in = self.relu(self.ocr_pos_layer_norm(ocr_mmt_in+ocr_pos))

        ocr_mmt_in = self.ocr_drop(ocr_mmt_in)
        fwd_results["ocr_mmt_in"] = ocr_mmt_in

        # binary mask of valid OCR vs padding
        ocr_nums = sample_list.context_info_0.max_features
        fwd_results["ocr_mask"] = _get_mask(ocr_nums, ocr_mmt_in.size(1))  # ocr_is_substr_mask

    def _get_ocr_clusterEmbedding(self, sample_list):
        device = sample_list.context_feature_0.device
        batch_size = sample_list.context_feature_0.size(0)

        ocr_pos_np = sample_list.ocr_pos
        ocr_pos_tensor = torch.tensor(ocr_pos_np, device=device)
        row_pos = ocr_pos_tensor[:, :, 0]
        col_pos = ocr_pos_tensor[:, :, 1]
        clu_pos = ocr_pos_tensor[:, :, 2]
        ocr_pos = self.get_sinusoid_pos_embedding(clu_pos, 1000)\
                  + self.get_sinusoid_pos_embedding(row_pos, 1000)\
                  + self.get_sinusoid_pos_embedding(col_pos, 1000)
        '''
            two ways to construct the cluster embedding:
            [clus_pos, row_pos, col_pos, zeros_for_padding]
        Or [clus_pos, row_pos, col_pos, ocr_pos, zeros_for_padding]
        '''
        return ocr_pos

    def _forward_mr_gcn_encoding(self, sample_list, fwd_results):

        gcn_input = torch.cat([fwd_results["obj_mmt_in"], fwd_results["ocr_mmt_in"]], dim=1)
        batch_size, device = gcn_input.size(0), gcn_input.device
        master_obj_matrix_numpy = sample_list.master_ocr_obj_matrix
        master_obj_matrix = torch.tensor(master_obj_matrix_numpy, dtype=torch.float).to(device)

        rel_edges = torch.zeros([batch_size, 2, 150, 150]).to(device)
        rel_edges[:, 0, :100, -50:] = master_obj_matrix.transpose(1, 2)
        rel_edges[:, 1, -50:, :100] = master_obj_matrix

        nodes_types = torch.zeros([batch_size, 150]).to(device)
        nodes_types[:, -50:] = 1
        nodes_types = nodes_types.long()

        encoded_v = self.gcn(gcn_input, nodes_types, rel_edges)

        fwd_results["obj_mmt_in"] = self.obj_graph(encoded_v[:, :100])
        fwd_results["ocr_mmt_in"] = self.ocr_graph(encoded_v[:, -50:])

    def _forward_ocr_cluster(self, sample_list, fwd_results):
        ocr_order_vectors = torch.zeros_like(sample_list.order_vectors)

        # Master object adjacent matrix: [16,50,100]
        ocr_bbox = sample_list.ocr_bbox_coordinates.cpu().numpy()
        ocr_nums = sample_list.context_info_0.max_features
        ocr_bbox_label = torch.zeros([fwd_results["ocr_mask"].size(0), fwd_results["ocr_mask"].size(1) + 1],
                                     dtype=torch.long).to(ocr_order_vectors.device)
        for i in range(len(ocr_bbox)):
            if ocr_nums[i] > 0:
                tmp = DBSCAN(eps=0.15, min_samples=1, metric=cal_distance).fit(ocr_bbox[i, : ocr_nums[i]]).labels_
                ocr_bbox_label[i, 1:ocr_nums[i] + 1] = torch.tensor(tmp).to(ocr_order_vectors.device)

        fwd_results["ocr_cluster"] = ocr_bbox_label

    def _foward_lstm_bsl(self, sample_list, fwd_results):

        if self.training:

            fwd_results["prev_inds"] = sample_list.train_prev_inds.clone()
            target_caption = sample_list.train_prev_inds.clone()
            target_cap_len = sample_list["train_loss_mask"].sum(dim=-1).unsqueeze(1)
            fwd_results["scores"] = self.lstm_r(fwd_results["obj_mmt_in"], fwd_results["ocr_mmt_in"],
                                                      fwd_results["obj_mask"], fwd_results["ocr_mask"], target_caption,
                                                      target_cap_len)
        else:

            fwd_results["prev_inds"] = torch.zeros_like(sample_list.train_prev_inds)
            fwd_results["prev_inds"][:, 0] = 1  # self.answer_processor.BOS_IDX = 1
            fwd_results["scores"] = self.lstm_r(fwd_results["obj_mmt_in"], fwd_results["ocr_mmt_in"],
                                                      fwd_results["obj_mask"], fwd_results["ocr_mask"], training=False,
                                                      )

    def get_sinusoid_pos_embedding(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        self.dropout1 = nn.Dropout(p=0.1)
        # Compute the positional encodings once in log space.
        pe = torch.zeros(n_position.size(0), n_position.size(1), d_hid).to(n_position.device)
        position = n_position.unsqueeze(2)
        div_term = torch.exp(torch.arange(0, d_hid, 2).to(n_position.device) *
                             -(math.log(10000.0) / d_hid))
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return self.dropout1(pe)

    def get_optimizer_parameters(self, config):
        optimizer_param_groups = []

        base_lr = config.optimizer.params.lr
        # collect all the parameters that need different/scaled lr
        finetune_params_set = set()
        for m in self.finetune_modules:
            optimizer_param_groups.append(
                {
                    "params": list(m["module"].parameters()),
                    "lr": base_lr * m["lr_scale"],
                }
            )
            finetune_params_set.update(list(m["module"].parameters()))
        # remaining_params are those parameters w/ default lr
        remaining_params = [
            p for p in self.parameters() if p not in finetune_params_set
        ]
        # put the default lr parameters at the beginning
        # so that the printed lr (of group 0) matches the default lr
        optimizer_param_groups.insert(0, {"params": remaining_params})

        return optimizer_param_groups

    @classmethod
    def update_registry_for_pretrained(cls, config, checkpoint, full_output):
        from omegaconf import OmegaConf

        # Hack datasets using OmegaConf
        datasets = full_output["full_config"].datasets
        dataset = datasets.split(",")[0]
        config_mock = OmegaConf.create({"datasets": datasets})
        registry.register("config", config_mock)
        registry.register(
            f"{dataset}_num_final_outputs",
            # Need to add as it is subtracted
            checkpoint["classifier.module.weight"].size(0)
            + config.classifier.ocr_max_num,
        )
        # Fix this later, when processor pipeline is available
        answer_processor = OmegaConf.create({"BOS_IDX": 1})
        registry.register(f"{dataset}_answer_processor", answer_processor)


class OcrPtrNet(nn.Module):
    def __init__(self, hidden_size, query_key_size=None):
        super().__init__()

        if query_key_size is None:
            query_key_size = hidden_size
        self.hidden_size = hidden_size
        self.query_key_size = query_key_size

        self.query = nn.Linear(hidden_size, query_key_size)
        self.key = nn.Linear(hidden_size, query_key_size)

    def forward(self, query_inputs, key_inputs, attention_mask):
        extended_attention_mask = (1.0 - attention_mask) * -10000.0
        if extended_attention_mask.dim() == 2:
            extended_attention_mask = extended_attention_mask.unsqueeze(1)

        query_layer = self.query(query_inputs)
        if query_layer.dim() == 2:
            query_layer = query_layer.unsqueeze(1)
            squeeze_result = True
        else:
            squeeze_result = False
        key_layer = self.key(key_inputs)

        scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) / math.sqrt(self.query_key_size)
        scores = scores + extended_attention_mask

        if squeeze_result:
            scores = scores.squeeze(1)

        return scores


class PrevPredEmbeddings1(nn.Module):

    def __init__(self, config):
        super().__init__()
        MAX_DEC_LENGTH = 100
        MAX_TYPE_NUM = 5
        hidden_size = config.hidden_size
        ln_eps = config.layer_norm_eps

        self.token_type_embeddings = nn.Embedding(MAX_TYPE_NUM, hidden_size)

        self.ans_layer_norm = BertLayerNorm(hidden_size, eps=ln_eps)
        self.ocr_layer_norm = BertLayerNorm(hidden_size, eps=ln_eps)
        self.emb_layer_norm = BertLayerNorm(hidden_size, eps=ln_eps)
        self.emb_dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, ans_emb, ocr_emb, prev_inds):
        assert prev_inds.dim() == 2 and prev_inds.dtype == torch.long
        assert ans_emb.dim() == 2
        batch_size = prev_inds.size(0)
        ans_num = ans_emb.size(0)

        # apply layer normalization to both answer embedding and OCR embedding
        # before concatenation, so that they have the same scale
        assert ans_emb.size(-1) == ocr_emb.size(-1)

        # apply layer normalization to both answer embedding and OCR embedding
        # before concatenation, so that they have the same scale
        ans_emb = self.ans_layer_norm(ans_emb)
        ocr_emb = self.ocr_layer_norm(ocr_emb)

        ans_emb = ans_emb.unsqueeze(0).expand(batch_size, -1, -1)
        ans_ocr_emb_cat = torch.cat([ans_emb, ocr_emb], dim=1)
        raw_dec_emb = _batch_gather(ans_ocr_emb_cat, prev_inds)

        # Token type ids: 0 -- vocab; 1 -- OCR
        token_type_ids = prev_inds.ge(ans_num).long()
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = token_type_embeddings
        embeddings = self.emb_layer_norm(embeddings)
        embeddings = self.emb_dropout(embeddings)
        dec_emb = raw_dec_emb + embeddings

        return dec_emb


class PrevPredEmbeddings(nn.Module):

    def __init__(self, config):
        super().__init__()

        hidden_size = config.hidden_size
        self.emb_dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, ans_emb, ocr_emb, prev_inds):
        batch_size = prev_inds.size(0)

        assert ans_emb.size(-1) == ocr_emb.size(-1)

        ans_emb = ans_emb.unsqueeze(0).expand(batch_size, -1, -1)
        ans_ocr_emb_cat = torch.cat([ans_emb, ocr_emb], dim=1)
        # ans_num = 6736
        # token_type_ids = prev_inds.ge(ans_num).long()
        # token_type_embeddings = self.token_type_embeddings(token_type_ids)

        dec_emb = _batch_gather(ans_ocr_emb_cat, prev_inds)

        return dec_emb


def _get_mask(nums, max_num):
    # non_pad_mask: b x lq, torch.float32, 0. on PAD
    batch_size = nums.size(0)
    arange = torch.arange(0, max_num).unsqueeze(0).expand(batch_size, -1)
    non_pad_mask = arange.to(nums.device).lt(nums.unsqueeze(-1))
    non_pad_mask = non_pad_mask.type(torch.float32)
    return non_pad_mask


@functools.lru_cache(maxsize=32)
def _get_causal_mask(seq_length, device):
    # generate a lower triangular mask
    mask = torch.zeros(seq_length, seq_length, device=device)
    for i in range(seq_length):
        for j in range(i + 1):
            mask[i, j] = 1.0
    return mask


def _batch_gather(x, inds):
    assert x.dim() == 3
    batch_size = x.size(0)
    length = x.size(1)
    dim = x.size(2)
    x_flat = x.view(batch_size * length, dim)

    batch_offsets = torch.arange(batch_size, device=inds.device) * length
    if inds.dim() == 2:
        batch_offsets = batch_offsets.unsqueeze(-1)
    assert batch_offsets.dim() == inds.dim()
    inds_flat = batch_offsets + inds
    results = F.embedding(inds_flat, x_flat)
    return results


class AttentionC(nn.Module):

    def __init__(self, image_features_dim, decoder_dim, attention_dim):
        super(AttentionC, self).__init__()

        self.features_att = nn.Linear(image_features_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, attended_features, decoder_hidden, attention_mask=None):

        att1 = self.features_att(attended_features)  # (batch_size, attend_features_dim, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, decoder_features_dim, attention_dim)
        att = self.full_att(self.tanh(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, m, n)
        if attention_mask is not None:
            extended_attention_mask = (1.0 - attention_mask) * -10000.0
            alpha = self.softmax(att + extended_attention_mask)  # (batch_size, 36)
        else:
            alpha = self.softmax(att)
        context = (attended_features * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, 2048)
        return context


class LSTM_R(nn.Module):

    def __init__(self,
                 decoder_dim=1000,
                 obj_dim=1000,
                 ocr_dim=1000,
                 emb_dim=1000,
                 attention_dim=1000,
                 mmt_config=None,
                 use_graph_reasoning=False):

        super(LSTM_R, self).__init__()
        self.dropout = nn.Dropout(0.5)
        self.decoder_dim = decoder_dim
        self.embed_config = mmt_config
        self.vocab_size = 6736
        self.ocr_nums = 50
        self.ss_prob = 0.0

        self.voc_emb = nn.Embedding(self.vocab_size, emb_dim)
        self.embed = PrevPredEmbeddings(self.embed_config)

        self.visual_attention = AttentionC(obj_dim, decoder_dim, attention_dim)

        self.fusion_lstm = nn.LSTMCell(decoder_dim + emb_dim + obj_dim, decoder_dim)
        if use_graph_reasoning:
            self.graph_reason_net = Graph_reasoning(obj_dim)

        self.ocr_prt = OcrPtrNet(hidden_size=attention_dim)

        self.fc = nn.Linear(decoder_dim, self.vocab_size)

    def init_hidden_state(self, batch_size, device):
        h = torch.zeros(batch_size, self.decoder_dim).to(device)  # (batch_size, decoder_dim)
        c = torch.zeros(batch_size, self.decoder_dim).to(device)
        return h, c

    def _modify_score_distribution(self, cur_step, argmax_inds, scores, label):

        scores = torch.softmax(scores, dim=-1)

        r = torch.ones_like(scores)

        ocr_inds = argmax_inds - 6736

        idx = (ocr_inds >= 0).long() * (ocr_inds + 1)

        for i in range(len(r)):
            if idx[i] > 0:
                r[i, -50:] += (label[i, 1:] == label[i, idx[i]]).long()
        scores = scores * r
        scores = scores / torch.sum(scores, dim=-1).unsqueeze(dim=-1)

        return scores

    def forward(self, obj_features, ocr_features, obj_mask, ocr_mask, target_caption=None, target_cap_len=None,
                training=True,
                label=None):
        max_len = 30
        batch_size = obj_features.size(0)
        device = obj_features.device
        # repeat_mask = torch.zeros([batch_size, self.vocab_size + self.ocr_nums]).to(device)

        if training:
            caption_lengths, sort_ind = target_cap_len.squeeze(1).sort(dim=0, descending=True)
            target_caption = target_caption[:, :max_len]

        else:
            target_caption = torch.zeros([batch_size, max_len], dtype=torch.long).to(device)
            target_caption[:, 0] = 1
            caption_lengths = torch.tensor([max_len for _ in range(batch_size)])

        h_v, c_v = self.init_hidden_state(batch_size, device)  # (batch_size, decoder_dim)
        # Remove <end> from lengths since we've finished generating words when we predict <end>
        decode_lengths = caption_lengths.tolist()
        predictions = torch.zeros(batch_size, max_len, self.vocab_size + self.ocr_nums).to(device)

        ocr_num = ocr_mask.sum(dim=-1)
        x_main_nums = (ocr_num + (ocr_num == 0).long()) + obj_mask.sum(dim=-1)

        x_main = torch.cat([obj_features, ocr_features], dim=1)
        x_main_mean = x_main.sum(dim=1) / x_main_nums.unsqueeze(1)
        x_mask = torch.cat([obj_mask, ocr_mask], dim=-1)
        state = torch.cat([h_v, c_v], dim=-1)

        dec_num = int(max(decode_lengths))
        if dec_num > max_len:
            dec_num = max_len
        # dec_num = 30
        beam_size = 5
        if not training and beam_size > 1:

            for batch_idx in range(batch_size):
                beam_repeat_mask = torch.zeros(beam_size, self.vocab_size + self.ocr_nums).to(device)
                state_idx = repeat_tensors(beam_size, state[batch_idx].unsqueeze(0))

                x_main_idx = repeat_tensors(beam_size, x_main[batch_idx].unsqueeze(0))
                x_main_mean_idx = repeat_tensors(beam_size, x_main_mean[batch_idx].unsqueeze(0))
                ocr_features_idx = repeat_tensors(beam_size, ocr_features[batch_idx].unsqueeze(0))
                x_mask_idx = repeat_tensors(beam_size, x_mask[batch_idx].unsqueeze(0))
                ocr_mask_idx = repeat_tensors(beam_size, ocr_mask[batch_idx].unsqueeze(0))

                tmp_pred_idx = torch.zeros(beam_size, max_len).to(device)
                sum_log_prob = torch.zeros(beam_size).to(device)
                acc_len = torch.zeros_like(sum_log_prob)
                done_beams_table = [[] for _ in range(beam_size)]
                tmp_pred_idx[:, 0] = 1
                it_idx = (tmp_pred_idx[:, 0]).long()
                tmp_pred_idx[:, 0] = 0
                for t in range(30):
                    beam_scores, beam_state = self.get_logprobs_state(it_idx, x_main_idx, x_main_mean_idx,
                                                                      ocr_features_idx, x_mask_idx, ocr_mask_idx
                                                                      , state_idx)
                    beam_scores[:, 3] = -1000000
                    beam_scores[:, -self.ocr_nums:] += beam_repeat_mask[:, -self.ocr_nums:]

                    beam_scores = F.log_softmax(beam_scores, dim=-1)

                    if t == 0:
                        sum_log_prob, topk_pred = beam_scores[0].topk(beam_size, dim=-1)
                        it_idx = topk_pred
                        tmp_pred_idx[:, t] = it_idx
                        state_idx = beam_state
                        acc_len += 1
                    else:
                        candidate_socres, candidate_idx = beam_scores.topk(beam_size, dim=-1)
                        candidate_socres, candidate_idx = repeat_tensors(beam_size,
                                                                         sum_log_prob) + candidate_socres.view(
                            beam_size * beam_size), \
                                                          candidate_idx.view(beam_size * beam_size)
                        # _, correspond_idx = (candidate_socres/(repeat_tensors(beam_size, acc_len)+1)
                        #                      ).topk(beam_size, dim=-1)
                        _, correspond_idx = candidate_socres.topk(beam_size, dim=-1)
                        corespond_beam_idx = torch.floor(correspond_idx / beam_size).long()

                        it_idx = torch.index_select(candidate_idx, 0, correspond_idx)
                        sum_log_prob = torch.index_select(candidate_socres, 0, correspond_idx)

                        state_idx = torch.index_select(beam_state, 0, corespond_beam_idx)
                        beam_repeat_mask = torch.index_select(beam_repeat_mask, 0, corespond_beam_idx)
                        acc_len = torch.index_select(acc_len, 0, corespond_beam_idx)
                        tmp_pred_idx[:, :t] = torch.index_select(tmp_pred_idx[:, :t], 0, corespond_beam_idx)
                        tmp_pred_idx[:, t] = it_idx
                        # if ((tmp_pred_idx[:, t]==2).long()).sum(dim=-1)==beam_size:
                        #     break
                        is_end = (tmp_pred_idx[:, t] == 2).long()
                        # if is_end[0] > 0 and (random.choice([0,1,2,3,4,5,6,7,8,9]) >= 1):
                        #     break
                        # if is_end[0] > 0 or t==29:
                        #     done_beams_table[0].append({
                        #         'seq': tmp_pred_idx[0].long(),
                        #         'p': sum_log_prob[0] / (acc_len[0] + 1)
                        #     })
                        #     break

                        for beam_index in range(beam_size):
                            if is_end[beam_index] > 0 or t == 29:
                                done_beams_table[beam_index].append({
                                    'seq': tmp_pred_idx[beam_index].long(),
                                    'p': sum_log_prob[beam_index] / (acc_len[beam_index] + 1)
                                })
                                sum_log_prob[beam_index] -= 1000
                            else:
                                acc_len[beam_index] += 1
                                # sum_log_prob[beam_index] -=5
                    #
                    beam_repeat_mask = beam_repeat_mask.scatter(1, it_idx.unsqueeze(1), -1000000)
                sorted(done_beams_table[0], key=lambda x: -x['p'])
                # print(done_beams_table[0])
                # print()
                # print()
                beam_res = done_beams_table[0][0]['seq']
                # beam_res = tmp_pred_idx[0].long()
                predictions[batch_idx] = predictions[batch_idx].scatter(1, beam_res.unsqueeze(1), 10)

        else:
            for t in range(dec_num):

                if training and t >= 1 and self.ss_prob > 0.0:
                    sample_prob = x_main_mean.new(batch_size).uniform_(0, 1)
                    sample_mask = sample_prob < self.ss_prob
                    if sample_mask.sum() == 0:
                        it = target_caption[:, t].clone()
                    else:
                        sample_ind = sample_mask.nonzero().view(-1)
                        it = target_caption[:, t].data.clone()
                        prob_prev = torch.exp(predictions[:, t - 1].detach())  # fetch prev distribution: shape Nx(M+1)
                        it.index_copy_(0, sample_ind,
                                       torch.multinomial(prob_prev, 1).view(-1).index_select(0, sample_ind))
                else:
                    it = target_caption[:, t].clone()

                scores, state = self.get_logprobs_state(it, x_main, x_main_mean,
                                                        ocr_features, x_mask, ocr_mask, state)
                # if not training and t < dec_num - 1:
                #     # if t>0:
                #     #     prev_inds = target_caption[:, t]
                #     #     scores = self._modify_score_distribution(t, prev_inds, scores, label)
                #     #     scores[:, -50:] = scores[:, -50:] * ocr_mask
                #     scores[:, 3] = -1e6
                #     scores = scores + repeat_mask
                #     pre_idx = (scores.argmax(dim=-1)).long()
                #     target_caption[:, t + 1] = pre_idx
                #     for j in range(batch_size):
                #         used_idx = pre_idx[j]
                #         if used_idx >= self.vocab_size:
                #             repeat_mask[j, used_idx] = -1e6

                predictions[:, t] = scores

        return predictions

    def get_logprobs_state(self, it, x_features, x_mean_features, ocr_features, x_mask, ocr_mask, state):

        h, c = state[:, :1000], state[:, 1000:2000]

        x_weighted = self.visual_attention(x_features, h, x_mask)

        y = self.embed(self.voc_emb.weight, ocr_features, it)
        x_fu = torch.cat([x_mean_features, x_weighted, y], dim=-1)
        h, c = self.fusion_lstm(x_fu, (h, c))
        #-------------------------
        # dec_mask = torch.ones([y.size(0), 1], dtype=torch.float).to(y.device)
        # h_out, x_ocr, x_obj = self.graph_reason_net(h.unsqueeze(1), ocr_features, x_features[:, :100], None, ocr_mask, x_mask[:,:100])
        # h_drop = self.dropout(h_out.squeeze(1))
        #-------------------------
        h_drop = self.dropout(h)
        s_v = self.fc(h_drop)
        s_o = self.ocr_prt(h_drop, ocr_features, ocr_mask)
        scores = torch.cat([s_v, s_o], dim=-1)
        state = torch.cat([h, c], dim=-1)

        return scores, state


def repeat_tensors(n, x):
    """
    For a tensor of size Bx..., we repeat it n times, and make it Bnx...
    For collections, do nested repeat
    """
    if torch.is_tensor(x):
        x = x.unsqueeze(1)  # Bx1x...
        x = x.expand(-1, n, *([-1] * len(x.shape[2:])))  # Bxnx...
        x = x.reshape(x.shape[0] * n, *x.shape[2:])  # Bnx...
    elif type(x) is list or type(x) is tuple:
        x = [repeat_tensors(n, _) for _ in x]
    return x


def _get_json_file(json_filename):
    with open(json_filename, 'r') as f:
        raw_file = json.load(f)
    return raw_file


def cal_distance(box1, box2):
    ax1, ay1, ax2, ay2 = box1
    bx1, by1, bx2, by2 = box2

    mid_ax = (ax2 + ax1) / 2
    mid_ay = (ay2 + ay1) / 2

    mid_bx = (bx2 + bx1) / 2
    mid_by = (by2 + by1) / 2

    sum_w = (ax2 - ax1 + bx2 - bx1) / 2
    sum_h = (ay2 - ay1 + by2 - by1) / 2

    Dx = math.fabs(mid_ax - mid_bx)
    Dy = math.fabs(mid_ay - mid_by)

    if Dx < sum_w and Dy >= sum_h:
        min_dist = Dy - sum_h
    elif Dx >= sum_w and Dy < sum_h:
        min_dist = Dx - sum_w
    elif Dx >= sum_w and Dy >= sum_h:
        delta_x = Dx - sum_w
        delta_y = Dy - sum_h
        min_dist = delta_x + delta_y
    else:
        min_dist = 0

    return min_dist
