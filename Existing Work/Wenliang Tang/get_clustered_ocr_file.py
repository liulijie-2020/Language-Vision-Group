import numpy as np
import json
import math
import pickle
import lmdb
import torch
from sklearn.cluster import DBSCAN

ocr_filename = "E:/DownloadDLdata/TextVQA_Rosetta_OCR_v0.2_val.json"

#image_filename = "E:/DownloadDLdata/TextCaps_0.1_val.json"

ocr_lmdb_path = "G:/VQA_DATA/textvqa/ocr_en/features/ocr_en_frcn_features.lmdb"
# 在这个程序里ocr_lmdb没有使用到
ocr_val_npy = "G:/VQA_DATA/modified_vqa_npy/imdb_val_filtered_by_image_id.npy"


def get_json_file(json_filename):
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
    # return math.fabs(mid_ax - mid_bx) + 10 * math.fabs(mid_ay - mid_by)


ocr_val_file = (np.load(ocr_val_npy, allow_pickle=True))[1:]
new_ocr_val_file = ocr_val_file
ocr_raw_file = get_json_file(ocr_filename)


def merge_bbox(box1, box2):
    ax1, ay1, ax2, ay2 = box1
    bx1, by1, bx2, by2 = box2
    lx = ax1 if (ax1 < bx1) else bx1
    ly = ay1 if (ay1 < by1) else by1
    rx = ax2 if (ax2 > bx2) else bx2
    ry = ay2 if (ay2 > by2) else by2

    return lx, ly, rx, ry


def trans(nums, limit):
    flag = max(nums)+1
    idx, count = [i for i in range(50)], [0 for i in range(50)]
    for j in range(len(nums)):
        i = nums[j]
        count[idx[i]] += 1
        nums[j] = idx[i]
        if count[idx[i]] == limit:
            idx[i] = flag
            flag += 1
    return nums
val_set_len = len(ocr_val_file)

# pred_val_file.sort(key=lambda x: x["caption_id"])

eps_param = 0.2

# val_set_len = 10

# Read the feature of each OCR region from the ocr_lmdb_path
# 从ocrfile中读取有用的信息，然后根据ocr中的image id找出lmdb中的ocr的特征，
# 对每一幅图像里的所有的ocr特征进行聚类
env_ocr = lmdb.open(
            ocr_lmdb_path,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
with env_ocr.begin(write=False) as txn:
    for i in range(val_set_len):
        #print(i)
        item = ocr_val_file[i]
        item_lmdb = pickle.loads(txn.get(('train/' + ocr_val_file[i]["image_id"]).encode()))
        ocr_features = item_lmdb['features']
        ocr_tokens = ocr_val_file[i]['ocr_tokens']
        ocr_bbox = ocr_val_file[i]['ocr_normalized_boxes']

        len_ocr_tokens = len(ocr_tokens)
        if len_ocr_tokens > 50:
            len_ocr_tokens = 50
            ocr_tokens = ocr_tokens[:50]
            ocr_bbox = ocr_bbox[:50]
        if len_ocr_tokens > 0:

            sorted_idx = np.argsort(ocr_bbox[:, 1])
            sorted_ocr_box = ocr_bbox[sorted_idx]
            sorted_ocr_tokens = (np.array(ocr_tokens)[sorted_idx]).tolist()
            # sort the ocr tokens by the x corridinates
            # if i>=9:
            #     print(sorted_ocr_tokens)
            # if len_ocr_tokens > 15:
            #     eps_param = 0.01
            # else:
            eps_param = 0.02
            labels = DBSCAN(eps=eps_param, min_samples=0, metric=cal_distance).fit(sorted_ocr_box).labels_
            labels = trans(labels, 4)

            max_kind = max(labels)
            cluster_array = ['' for _ in range(max_kind + 2)]
            cluster_bbox = [[0 for _ in range(4)] for _ in range(max_kind + 2)]
            cluster_ocr_info = [{} for _ in range(max_kind + 1)]
            ocr_nums_per_cluster = [0 for _ in range(max_kind+2)]
            for k in range(len_ocr_tokens):
                cls_idx = labels[k] + 1
                cluster_bbox[cls_idx] = ocr_bbox[k]

            for k in range(len_ocr_tokens):
                ocr_nums_per_cluster[labels[k]+1] += 1

                cluster_array[labels[k] + 1] += (ocr_tokens[k] + ' ')
                cls_idx = labels[k] + 1
                cluster_bbox[cls_idx] = merge_bbox(cluster_bbox[labels[k] + 1], ocr_bbox[k])

            for ocr_comb in cluster_array:
                ocr_comb = ocr_comb[:-1]
            ocr_val_file[i]['ocr_tokens'] = cluster_array[1:]
            ocr_val_file[i]['ocr_normalized_boxes'] = np.array(cluster_bbox[1:])
            for k in range(1,len(cluster_array)):
                cluster_ocr_info[k-1]['word'] = cluster_array[k]

                cluster_ocr_info[k-1]['bounding_box'] = {'top_left_x': cluster_bbox[k][0],
                                                       'top_left_y': cluster_bbox[k][1],
                                                       'width': cluster_bbox[k][2] - cluster_bbox[k][0],
                                                       'height': cluster_bbox[k][3] - cluster_bbox[k][1]
                                                       }
            new_ocr_val_file[i]['ocr_info'] = cluster_ocr_info

            # print("origianl:")
            # print(ocr_tokens)
            # print("clustered:")
            # print(cluster_array)
            #
            # print()


final_file = [ {} for _ in range(len(new_ocr_val_file)+1 )]
final_file[0]={'data create':'25-JAN-2020' }
final_file[1:] = new_ocr_val_file

np.save(ocr_val_npy[:-4]+"_modified.npy", final_file)
print("done")



print("done")
