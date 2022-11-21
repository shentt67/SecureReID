import random
import os
import shutil
import argparse


# split bounding_box_train
# sequentially select p1 of training set as validation set
# randomly select p2 of validation set as validation query set, others as validation gallery set
def split_train(dataset_root, dataset_name, p1=0.2, p2=0.2):
    src_root = os.path.join(dataset_root, dataset_name)
    dst_root = os.path.join(dataset_root, dataset_name + '_val')
    shutil.copytree(src_root, dst_root)

    train_root = os.path.join(dst_root, 'bounding_box_train')
    val_query_root = os.path.join(dst_root, 'bounding_box_val_query')
    val_gallery_root = os.path.join(dst_root, 'bounding_box_val_gallery')

    os.makedirs(train_root)
    os.makedirs(val_query_root)
    os.makedirs(val_gallery_root)

    train_file_list = os.listdir(train_root)
    train_file_list.sort()
    train_file_list = train_file_list[:int(len(train_file_list) * p1)]

    for file in train_file_list:
        src = os.path.join(train_root, file)
        if (random.random() <= p2):
            dst = os.path.join(val_query_root, file)
        else:
            dst = os.path.join(val_gallery_root, file)
        shutil.move(src, dst)


parser = argparse.ArgumentParser(description="Split Training Set")
parser.add_argument("--dataset_root_dir", dest='path', type=str)
parser.add_argument("--dataset_name", dest='name', help="market1501 / msmt17", type=str)
args = parser.parse_args()

split_train(args.path, args.name)
