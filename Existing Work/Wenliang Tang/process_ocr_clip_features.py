'''
1. import CLIP, image
2. import OCR region information
    1) find the corresponding region of each OCR in the images
    2) get the image in each region
    3) use clip extract the features from the image in each region
3. build the format of .npy file
'''
import torch
import clip
import os
import glob as gb
import numpy as np
import tqdm
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

image_folder = "/home2/tangwenliang/train/*.jpg"
image_path = gb.glob(image_folder)

imdb_file_train_path = "/home2/tangwenliang/mmf/datasets/textcaps/defaults" \
                       "/annotations/imdb_train.npy"
imdb_file_val_path = "/home2/tangwenliang/mmf/datasets/textcaps/defaults/" \
                     "annotations/imdb_val.npy"

image_dir = "/home2/tangwenliang/train/"
save_dir = "/home2/tangwenliang/OCR_CLIP_ViT32_features/train/"


def extract_CLIP_features(IMAGE_DIR, IMDB_FILE, SAVE_DIR, device):
    imdb = np.load(IMDB_FILE, allow_pickle=True)[1:]
    # keep only one entry per image_id
    image_id2info = {info["image_id"]: info for info in imdb}
    imdb = list(image_id2info[k] for k in sorted(image_id2info))

    print("Faster R-CNN OCR features")
    print("\textracting from ", IMDB_FILE)
    print("\tsaving to ", SAVE_DIR)

    for index, info in enumerate(tqdm.tqdm(imdb)):
        image_file_path = os.path.join(IMAGE_DIR, info["image_id"] + ".jpg")
        save_feat_path = os.path.join(SAVE_DIR, info["feature_path"])
        save_info_path = save_feat_path.replace(".npy", "_info.npy")
        os.makedirs(os.path.dirname(save_feat_path), exist_ok=True)

        w = info["image_width"]
        h = info["image_height"]
        ocr_normalized_boxes = np.array(info["ocr_normalized_boxes"])
        ocr_boxes = ocr_normalized_boxes.reshape(-1, 4) * [w, h, w, h]
        ocr_tokens = info["ocr_tokens"]
        if len(ocr_boxes) > 0:

            extracted_ocr_feat = torch.zeros(len(ocr_boxes), 1024).to(device)
            image = Image.open(image_file_path)
            for idx, ocr_box_item in enumerate(ocr_boxes):
                ocr_img = preprocess(image.crop(ocr_box_item)).unsqueeze(0).to(device)
                ocr_text = clip.tokenize(ocr_tokens[idx]).to(device)

                extracted_ocr_feat[idx, :512] = model.encode_image(ocr_img)
                extracted_ocr_feat[idx, 512:] = model.encode_text(ocr_text)

            extracted_feat = extracted_ocr_feat.detach().cpu().numpy()
            pass
        else:
            extracted_feat = np.zeros((0, 1024), np.float32)

        np.save(save_info_path, {"ocr_boxes": ocr_boxes, "ocr_tokens": ocr_tokens})
        np.save(save_feat_path, extracted_feat)


if __name__ == '__main__':
    extract_CLIP_features(IMAGE_DIR=image_dir, IMDB_FILE=imdb_file_train_path,
                          SAVE_DIR=save_dir, device=device)
