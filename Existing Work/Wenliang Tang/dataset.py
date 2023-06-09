# Copyright (c) Facebook, Inc. and its affiliates.
import math

import numpy as np

from mmf.datasets.builders.textvqa.dataset import TextVQADataset
from mmf.utils.distributed import object_to_byte_tensor
import torch


class TextCapsDataset(TextVQADataset):
    def __init__(self, config, dataset_type, imdb_file_index, *args, **kwargs):
        super().__init__(config, dataset_type, imdb_file_index, *args, **kwargs)
        self.dataset_name = "textcaps"
        self.calculate_masterobj = True
        self.build_master_obj_option = ["max_box", "min_box", "max_iou", "all_box"]
        self.stragedy_build_masterobj = self.build_master_obj_option[1]

    def preprocess_sample_info(self, sample_info):
        sample_info = super().preprocess_sample_info(sample_info)
        # add dummy questions to train with M4C (for TextVQA)
        sample_info["question_str"] = ""  # empty question
        sample_info["question_id"] = sample_info["caption_id"]
        return sample_info

    def postprocess_evalai_entry(self, entry):
        new_entry = {
            "caption_id": entry["question_id"],
            "image_id": entry["image_id"],
            "caption": entry["answer"],
            "pred_source": entry["pred_source"]
            # "ocr_nums":entry["ocr_nums"]
        }
        return new_entry

    def add_answer_info(self, sample_info, sample):
        sample_has_caption = "caption_str" in sample_info
        if sample_has_caption:
            sample_info["answers"] = [sample_info["caption_str"]]

        sample = super().add_answer_info(sample_info, sample)

        sample = self.add_master_obj_info(sample_info, sample)

        sample = self.add_ocr_pos_info(sample_info, sample)

        sample = self.add_ocr_is_remove(sample_info, sample)

        if sample_has_caption:
            sample.caption_str = object_to_byte_tensor(sample_info["caption_str"])
            sample.ref_strs = object_to_byte_tensor(sample_info["reference_strs"])
            sample.pop("answers")

        return sample

    def add_master_obj_info(self, sample_info, sample):
        ocr_boxes = sample_info["ocr_normalized_boxes"]
        obj_boxes = sample_info["obj_normalized_boxes"]
        # for ii in range(len(nounce_idx)):
        #     nounce_idx[ii] = nounce_idx[ii] - 1
        record_dict_array = []
        ocr_max_len = 50
        record_array = [-1 for _ in range(len(ocr_boxes))]
        record_array1 = [-1 for _ in range(len(obj_boxes))]
        record_array_ocr = [-1 for _ in range(ocr_max_len)]
        record_ocr_obj_matrix = [[0 for _ in range(100)] for _ in range(ocr_max_len)]
        obj_cls = sample['image_info_0']['objects']

        if self.calculate_masterobj:
            if len(ocr_boxes) == 0:
                sample.master_obj = record_array
                sample.master_obj2ocr = record_dict_array
                record_array1 = [-1 for _ in range(len(obj_boxes))]
                sample.master_obj_mask = record_array1
                sample.master_ocr2obj_label = record_array_ocr
                sample.master_ocr_obj_matrix = record_ocr_obj_matrix
                return sample

            if self.stragedy_build_masterobj == "min_box":
                master_object_output = self.build_masterobj_min_box_version(ocr_boxes, obj_boxes, sample)
            elif self.stragedy_build_masterobj == "max_box":
                pass
            elif self.stragedy_build_masterobj == "max_iou":
                pass
            elif self.stragedy_build_masterobj == "others":
                pass
            sample.master_obj = master_object_output["master_obj"]  # obj its corresponding ocr tokens idx
            sample.master_obj2ocr = master_object_output["master_obj2ocr"]
            sample.master_obj_mask = master_object_output["master_obj_mask"]
            sample.master_ocr2obj_label = master_object_output[
                "master_ocr2obj_label"]  # ocr belongs to its containing object label
            sample.master_ocr_obj_matrix = master_object_output[
                "master_ocr_obj_matrix"]  # obj_box that containing the ocr_box

        return sample

    def add_ocr_pos_info(self, sample_info, sample):
        record_ocr_pos = [[0, 0, 0] for _ in range(50)]
        if "ocr_pos" in sample_info:
            ocr_len = len(sample_info["ocr_pos"])
            if ocr_len > 50:
                ocr_len = 50
            if ocr_len > 0:
                record_ocr_pos[:ocr_len] = sample_info["ocr_pos"][:ocr_len]

        sample.ocr_pos = record_ocr_pos  # obj_box that containing the ocr_box

        return sample

    

   

    def build_masterobj_min_box_version(self, ocr_boxes, obj_boxes, sample):
        ocr_max_len = 50
        record_array = [-1 for _ in range(len(ocr_boxes))]  # for each ocr_box, which object boxes is its master object
        record_array1 = [-1 for _ in
                         range(len(obj_boxes))]  # for each obj_box, which ocr boxes is its containing master object
        record_ocr_obj_matrix = [[0 for _ in range(100)] for _ in range(ocr_max_len)]
        # a matrix that whether ocr i has master object j
        obj_cls = sample['image_info_0']['objects']
        nounce_idx = [1472, 924, 716, 149, 1597, 900, 992, 1180, 835, 1039, 915, 364, 1089, 909,
                      844, 73, 64, 602, 1000, 180, 373, 783, 328, 684, 1469, 278, 1252, 1028,
                      1172, 120, 733, 732, 1100]  # 物体类别检测结果中，属于不可能有依附文字的类别标签号码，例如天空，树叶等
        ocr_len = len(ocr_boxes) if len(ocr_boxes) <= ocr_max_len else ocr_max_len

        for i in range(ocr_len):
            ocr_box = ocr_boxes[i]
            tmp_i, tmp_j = -1, -1
            [x1_m, y1_m, x2_m, y2_m] = ocr_box
            box_min_area = math.fabs((obj_boxes[0][2] - obj_boxes[0][0])
                                     * (obj_boxes[0][3] - obj_boxes[0][1]))
            for j in range(len(obj_boxes)):
                obj_box = obj_boxes[j]
                [x1, y1, x2, y2] = obj_box

                # The next section is used for calculating the min bounding box
                if x1_m > x1 and y1_m > y1 and x2_m < x2 and y2_m < y2:
                    cur_box_area = math.fabs((x1 - x2) * (y1 - y2))
                    if cur_box_area < box_min_area and ((obj_cls[j] + 1) not in nounce_idx):
                        box_min_area = cur_box_area
                        tmp_i, tmp_j = i, j

            # Find the min obj_box containing the ocr_box
            if tmp_i >= 0 and tmp_j >= 0:
                record_ocr_obj_matrix[tmp_i][tmp_j] = 1
                record_array[tmp_i], record_array1[tmp_j] = tmp_j, tmp_i
                
        for i in range(len(record_array)):
            for j in range(len(record_array)):
                if record_array[i]>=0 and record_array[j]>=0:
                    record_ocr_obj_matrix[i][j] = 1
                    
        master_object_output = {}          
        master_object_output["master_obj"] = record_array                # 若矩阵中某元素值为v，其表示对应的主要目标物体是所有物体中的第v个物体
        master_object_output["master_obj_mask"] = record_array1          # 若矩阵中某元素值为v，其表示该物体包围着第v个OCR文字
        master_object_output["master_ocr_obj_matrix"] = record_ocr_obj_matrix  # 若矩阵a_ij = 1，其表示对应的OCR i的主要目标物体是物体j
        return master_object_output

    def add_ocr_cluster_info(self, sample_info, sample):
        # 为OCR字符增加聚类后的信息，如果没有聚类信息，就不调用这个函数
        array_ocr_obj_matrix = sample_info["ocr_obj_matrix"]
        obj_boxes = sample_info["obj_normalized_boxes"]
        # ocr_clusterd = sample[""]
        sample.master_obj = record_array  # obj its corresponding ocr tokens idx
        # sample.master_obj2ocr = record_dict_array
        # sample.master_obj_mask = record_array1
        sample.master_ocr2obj_label = record_array_ocr  # ocr belongs to its containing object label
        sample.master_ocr_obj_matrix = array_ocr_obj_matrix  # obj_box that containing the ocr_box

        return sample
    
    def add_ocr_is_remove(self, sample_info, sample):
        # 整理OCR数据时候用到的，如果这个OCR标记为和别的OCR重叠(检测效果不准确，
        # 容易出现一个文字检测出两个重叠的结果)，就标记为0，之后不会再用到这个OCR
        record_ocr_is_substr = [0 for _ in range(50)]
        if "ocr_is_substr" in sample_info:
            record_ocr_is_substr = sample_info["ocr_is_substr"]

        sample.ocr_is_substr = record_ocr_is_substr  # obj_box that containing the ocr_box

        return sample
    
    def iou(self, obj_bbox1, ocr_bbox2):
        xmin1, ymin1, xmax1, ymax1 = obj_bbox1
        xmin2, ymin2, xmax2, ymax2 = ocr_bbox2

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
