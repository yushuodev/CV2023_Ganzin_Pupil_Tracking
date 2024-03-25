import os 
import natsort
import re
import glob
import json 
import cv2
import numpy as np
import random
from tqdm import tqdm
import json

# 檔案擺放格式
# dataset/
# • private/
# • S5 
# • S6 
# • S7 
# • S8 
# • public/
# • S1 
# • S2 
# • S3 
# • S4 

# 檔案生成完格式
# dataset/
# • private
# • public
# • train/
# • {... .jpg}
# • train.json
# • val/
# • {... .jpg}
# • val.json

dataset_path = './dataset'

subjects = ['S1', 'S2', 'S3', 'S4']
val_ratio = 0.1


def preprocessing(img_path):
    img = cv2.imread(img_path, 0)
    clahe = cv2.createCLAHE(clipLimit=1, tileGridSize=(24, 24))
    med = clahe.apply(img).astype(np.float32)/255
    gamma = (1/2.2)
    med = med ** gamma
    med = np.clip(med * 255, 0, 255).astype(np.uint8)
    med = cv2.medianBlur(med, 17)
    return med

def binarization(img_path):
    img = cv2.imread(img_path, 0)
    _, thresh = cv2.threshold(img, thresh=1, maxval=255, type=cv2.THRESH_BINARY)
    return thresh


open_eyes_label_path = []
close_eyes_label_path = []

for subject in subjects:
    subject_path = os.path.join(os.path.join(dataset_path, 'public'), subject)
    nums = os.listdir(subject_path)
    if '.DS_Store' in nums:
        nums.remove('.DS_Store')
    nums = natsort.natsorted(nums, reverse=False)
    for num in nums:
        file_path = os.path.join(subject_path, num)
        # 讀取所有的 .png 檔案
        labels_path = glob.glob(file_path + '/*.png')
        labels_path = natsort.natsorted(labels_path, reverse=False)
        # 將資料分為睜閉眼
        for label_path in labels_path:
            label = cv2.imread(label_path, 0)
            if np.count_nonzero(label!=0) == 0:
                close_eyes_label_path.append(label_path)
            else:
                open_eyes_label_path.append(label_path)
    
# print(len(close_eyes_label_path))
# print(len(open_eyes_label_path))
train_dir = os.path.join(dataset_path, 'train_')
val_dir = os.path.join(dataset_path, 'val_')
random.seed(9527)
open_eyes_val_indicies = random.sample(range(len(open_eyes_label_path)), int(len(open_eyes_label_path)*0.1))
close_eyes_val_indicies = random.sample(range(len(close_eyes_label_path)), int(len(close_eyes_label_path)*0.1))
#print(close_eyes_val_indicies)


# eyes open or close
# 先寫入閉眼的圖片，再寫入睜眼的的圖片。
train_data = {'filenames': [], 'labels': []}
val_data = {'filenames': [], 'labels': []}
for i, index in tqdm(enumerate(range(len(close_eyes_label_path)))):
    # train 
    if index not in close_eyes_val_indicies:
        #print(index)
        src = re.sub('png', 'jpg', close_eyes_label_path[index])
        dst = os.path.join(train_dir + 'eyes', str(i) + '.jpg')
        if not os.path.exists('dataset/train_eyes'):
            os.mkdir('dataset/train_eyes')
        #shutil.copyfile(src, dst)
        img = preprocessing(src)
        cv2.imwrite(dst, img)
        train_data['filenames'].append(dst)
        train_data['labels'].append(0)
    # validation
    else:
        src = re.sub('png', 'jpg', close_eyes_label_path[index])
        dst = os.path.join(val_dir + 'eyes', str(i) + '.jpg')
        if not os.path.exists('dataset/val_eyes'):
            os.mkdir('dataset/val_eyes')
        img = preprocessing(src)
        cv2.imwrite(dst, img)
        val_data['filenames'].append(dst)
        val_data['labels'].append(0)


for i, index in tqdm(enumerate(range(len(open_eyes_label_path)))):
    i = i + len(close_eyes_label_path)
    # train
    if index not in open_eyes_val_indicies:
        #print(index)
        src = re.sub('png', 'jpg', open_eyes_label_path[index])
        dst = os.path.join(train_dir + 'eyes', str(i) + '.jpg')
        if not os.path.exists('dataset/train_eyes'):
            os.mkdir('dataset/train_eyes')
        img = preprocessing(src)
        cv2.imwrite(dst, img)
        train_data['filenames'].append(dst)
        train_data['labels'].append(1)
    # validation
    else:
        src = re.sub('png', 'jpg', open_eyes_label_path[index])
        dst = os.path.join(val_dir + 'eyes', str(i) + '.jpg')
        if not os.path.exists('dataset/val_eyes'):
            os.mkdir('dataset/val_eyes')
        img = preprocessing(src)
        cv2.imwrite(dst, img)
        val_data['filenames'].append(dst)
        val_data['labels'].append(1)
    
with open('dataset/val_eyes/val.json', 'w') as file:
    json.dump(val_data, file, indent=4)
    
with open('dataset/train_eyes/train.json', 'w') as file:
    json.dump(train_data, file, indent=4)




# # location 
# train_data = {'filenames': [], 'labels': []}
# val_data = {'filenames': [], 'labels': []}
# for i, index in tqdm(enumerate(range(len(open_eyes_label_path)))):
#     # train
#     if index not in open_eyes_val_indicies:
#         #print(index)
#         src = re.sub('png', 'jpg', open_eyes_label_path[index])
#         img_dst = os.path.join(train_dir + 'location', str(i) + '.jpg')
#         label_dst = os.path.join(train_dir + 'location', str(i) + '.png')
#         if not os.path.exists('dataset/train_location'):
#             os.mkdir('dataset/train_location')
#         img = preprocessing(src)
#         label = binarization(open_eyes_label_path[index])
#         cv2.imwrite(img_dst, img)
#         cv2.imwrite(label_dst, label)
#         train_data['filenames'].append(img_dst)
#         train_data['labels'].append(label_dst)
#     # validation
#     else:
#         src = re.sub('png', 'jpg', open_eyes_label_path[index])
#         img_dst = os.path.join(val_dir + 'location', str(i) + '.jpg')
#         label_dst = os.path.join(val_dir + 'location', str(i) + '.png')
#         if not os.path.exists('dataset/val_location'):
#             os.mkdir('dataset/val_location')
#         img = preprocessing(src)
#         label = binarization(open_eyes_label_path[index])
#         cv2.imwrite(img_dst, img)
#         cv2.imwrite(label_dst, label)
#         val_data['filenames'].append(img_dst)
#         val_data['labels'].append(label_dst)

# with open('dataset/val_location/val.json', 'w') as file:
#     json.dump(val_data, file, indent=4)
    
# with open('dataset/train_location/train.json', 'w') as file:
#     json.dump(train_data, file, indent=4)