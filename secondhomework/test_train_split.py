import os
import os.path as osp
import random
import shutil


root = r'F:\DataSource\fruit30_train'
train_path = osp.join(root, 'train')
val_path = osp.join(root, 'val')
for i, label in enumerate(os.listdir(root)):
    img_list = os.listdir(osp.join(root, label))
    random.shuffle(img_list)
    train_list = img_list[:int(len(img_list)*0.7)]
    val_list = img_list[int(len(img_list)*0.7):]
    os.makedirs(osp.join(train_path, label), exist_ok=True)
    os.makedirs(osp.join(val_path, label), exist_ok=True)
    for img in train_list:
        shutil.copyfile(osp.join(root, label, img), osp.join(train_path, label, img))
    for img in val_list:
        shutil.copyfile(osp.join(root, label, img), osp.join(val_path, label, img))
    # with open(osp.join(root, 'meta/trian.txt'), 'w') as f:
    #     f.writelines(label+'\\'+filename + '' + i for filename in train_list)
    # with open(osp.join(root, 'meta/val.txt'), 'w') as f:
    #     f.writelines(label+'\\'+filename + '' + i for filename in val_list)