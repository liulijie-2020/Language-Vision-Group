import json
from PIL import Image
import pickle
import lmdb
import os
import torch
import numpy as np
import math
import clip
import glob as gb
import tqdm


device = "cuda" if torch.cuda.is_available() else "cpu"
#model, preprocess = clip.load("ViT-B/32", device=device)
model, preprocess = clip.load("RN101", device=device)

image_folder = "/home2/tangwenliang/train/*.jpg"
image_path = gb.glob(image_folder)
new_image_save_path = "/home2/tangwenliang/train_save_image/*.jpg"
imdb_file_train_path = "/home2/tangwenliang/mmf/datasets/textcaps/defaults" \
                       "/annotations/imdb_train.npy"
imdb_file_val_path = "/home2/tangwenliang/mmf/datasets/textcaps/defaults/" \
                     "annotations/imdb_val.npy"

image_dir = "/home2/tangwenliang/train/"
save_dir = "/home2/tangwenliang/obj_CLIP_RN101_features/train/"

obj_cls_label_map = "F:/DownloadModel/mmf-master/objects_vocab.json"

obj_lmdb_path = "/home2/tangwenliang/mmf/datasets/textvqa/defaults/features/open_images/detectron.lmdb"



def extract_CLIP_features(IMAGE_DIR, IMDB_FILE, LMDB_FILE, SAVE_DIR, device):
    imdb = np.load(IMDB_FILE, allow_pickle=True)[1:]
    # keep only one entry per image_id
    image_id2info = {info["image_id"]: info for info in imdb}
    imdb = list(image_id2info[k] for k in sorted(image_id2info))

    print("\textracting from box file:", LMDB_FILE)
    print("\textracting from IMDB file", IMDB_FILE)
    print("\tsaving to ", SAVE_DIR)
    # Get object boxes from the LMDB_FILE
    env = lmdb.open(
        LMDB_FILE,
        max_readers=1,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
    )
    with env.begin(write=False) as txn:

        for index, info in enumerate(tqdm.tqdm(imdb)):
            if index > 3000:
                image_path = os.path.join(IMAGE_DIR, info["image_id"]+".jpg")
                save_feat_path = os.path.join(SAVE_DIR, info["feature_path"])
                save_info_path = save_feat_path.replace(".npy", "_info.npy")
                os.makedirs(os.path.dirname(save_feat_path), exist_ok=True)

                w = info["image_width"]
                h = info["image_height"]
                item = pickle.loads( txn.get(('train/'+info["image_id"]).encode() ) )
                obj_boxes = item["bbox"]
                obj_cls = item["objects"]

                if len(obj_boxes) > 0:

                    extracted_ocr_feat = torch.zeros(len(obj_boxes), 512).to(device)
                    image = Image.open(image_path)

                    for idx, obj_box_item in enumerate(obj_boxes):
                        obj_img = preprocess(image.crop(obj_box_item)).unsqueeze(0).to(device)
                        # Is each OCR or all OCR?
                        extracted_ocr_feat[idx] = model.encode_image(obj_img)
                    extracted_feat = extracted_ocr_feat.detach().cpu().numpy()
                else:
                    extracted_feat = np.zeros((0, 512), np.float32)
                del item['features']

                np.save(save_info_path, item)
                np.save(save_feat_path, extracted_feat)


if __name__== '__main__':
    extract_CLIP_features(IMAGE_DIR=image_dir, IMDB_FILE=imdb_file_val_path, LMDB_FILE=obj_lmdb_path,
                          SAVE_DIR=save_dir, device=device)
    print("Done")
