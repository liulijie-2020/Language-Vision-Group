import json
from urllib import request
from PIL import Image, ImageDraw, ImageFont
import pickle
import lmdb
import torch
from sklearn.cluster import DBSCAN
import numpy as np
import math


obj_cls_label_map = "F:/DownloadModel/mmf-master/objects_vocab.json"

lmdb_path = "G:/VQA_DATA/textvqa/defaults/features/open_images/detectron.lmdb"
ocr_lmdb_path = "G:/VQA_DATA/textvqa/ocr_en/features/ocr_en_frcn_features.lmdb"

original_caption_path ="E:/DownloadDLdata/comparision_on_textcaps_val/textcaps_run_val_original.json"


ocr_train_npy = "G:/VQA_DATA/imdb_val_filtered_by_image_id_ocr_pos_modified_0119.npy"


noun_idx = [1472, 924, 716, 149, 1597, 900, 992, 1180, 835, 1039, 915, 364, 1089, 909,
                      844, 73, 64, 602, 1000, 180, 373, 783, 328, 684, 1469, 278, 1252, 1028,
                      1172, 120, 733, 732, 1100]
# 73-sky 1000-birds  181-bird  373-hand  783-hands 328-face  684-shadow  1469-shadows 278-scissors
noun_str = ['letter', 'letters', 'logo', 'sign', 'signs', 'date', 'text', 'word', 'words', 'number', 'numbers', 'writing', 'paper', 'papers', 'button', 'buttons', 'label']

for i in range(len(noun_idx)):
    noun_idx[i] = noun_idx[i] - 1


def get_json_file(json_filename):
    with open(json_filename, 'r') as f:
        raw_file = json.load(f)
    return raw_file


def draw_rect(img, left_x, left_y, width, height):
    draw = ImageDraw.Draw(img)
    draw.line((left_x, left_y, left_x, left_y + height), fill=128, width=3)
    draw.line((left_x, left_y, left_x + width, left_y), fill=128, width=3)
    draw.line(
        (left_x, left_y + height, left_x + width, left_y + height),
        fill=128, width=3)
    draw.line(
        (left_x + width, left_y, left_x + width, left_y + height),
        fill=128,
        width=3)
    return img


def cal_distance(box1, box2):
    ax1, ay1, ax2, ay2 = box1
    bx1, by1, bx2, by2 = box2

    mid_ax = (ax2 + ax1) / 2
    mid_ay = (ay2 + ay1) / 2

    mid_bx = (bx2 + bx1) / 2
    mid_by = (by2 + by1) / 2

    return math.sqrt((mid_ax - mid_bx) * (mid_ax - mid_bx) + (mid_ay - mid_by) * (mid_ay - mid_by))

ocr_train_npy_file = np.load(ocr_train_npy, allow_pickle=True)

cls_label_map = get_json_file(obj_cls_label_map)

original_caption = get_json_file(original_caption_path)
original_caption.sort(key = lambda x:x['caption_id'])



res = [ ]
i = 0
record = [0 for _ in range(100)]
font = ImageFont.truetype("arialuni.ttf", 25)
font1 = ImageFont.truetype("arialuni.ttf", 20)

env = lmdb.open(
            lmdb_path,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
with env.begin(write=False) as txn:

    ocr_file = ocr_train_npy_file[1:]
    n = len(ocr_file)

    count = 0
    total = 0
    for i in range(635, n):

        image_root_path = "F:/迅雷下载/train/"
        image_name = ocr_file[i]["image_id"]+".jpg"
        img = Image.open(image_root_path + image_name)
        [width_fact, height_fact] = img.size


        item = pickle.loads( txn.get(('train/'+ocr_file[i]["image_id"]).encode()  )  )
        obj_bounding_box = item["bbox"]
        m_obj_box_nums = item["num_boxes"]
        obj_cls = item["objects"]

        ocr_bounding_box = ocr_train_npy_file[1:][i]['ocr_normalized_boxes']
        m = len(ocr_bounding_box)
        if m > 50:
            m = 50

        record = [0 for _ in range(100)]

        total += m

        #original_caption_instance = original_caption[i]
        ocr_bbox_string = ocr_train_npy_file[1:][i]['ocr_info']


        draw = ImageDraw.Draw(img)


        for j in range(m):
            ocr_bounding_box_j_th = ocr_bounding_box[j]
            ocr_string = ocr_bbox_string[j]["word"]#+"_"+str(j)


            olx, oly, orx, ory = ocr_bounding_box_j_th
            olx *= width_fact
            oly *= height_fact
            orx *= width_fact
            ory *= height_fact

            color = 128
            #draw.rectangle((olx, oly, orx, ory), outline='red', width=8)

            #tmp_color1 = (150,60,20)


        for k in range(m_obj_box_nums):
            obj_bounding_box_k_th = obj_bounding_box[k]

            top_left_x_obj = obj_bounding_box_k_th[0] #* width_fact / item["image_width"]
            top_left_y_obj = obj_bounding_box_k_th[1] #* height_fact / item["image_height"]
            right_x_obj = obj_bounding_box_k_th[2] #* width_fact / item["image_width"]
            right_y_obj = obj_bounding_box_k_th[3] #* height_fact / item["image_height"]
            draw.rectangle((top_left_x_obj, top_left_y_obj, right_x_obj, right_y_obj), outline='green', width=6)
        

        
        path = "F:/test_predict_file/master_object_img/"+image_name
        img.save(path)
        img.show()
    print("------------------------------------------------------------")
    print("Done")
