import os
import argparse
import pandas as pd

from glob import glob
from tqdm import tqdm

def parse():

    parser = argparse.ArgumentParser()

    parser.add_argument("--image-root-path", type=str)
    parser.add_argument("--dataframe-output-path", type=str)
    
    args = parser.parse_args()

    return args

def class_mapping(image_root_path):

    folders = glob(f"{image_root_path}/*")
    mapping_dict = {os.path.basename(folder):idx for idx, folder in enumerate(folders)}

    return mapping_dict

def get_dataframe(image_root_path, mapping_dict):

    df = pd.DataFrame()

    classes = list()
    paths = list()
    class_indices = list()

    folders = glob(f"{image_root_path}/*")
    for folder in tqdm(folders,total=len(folders),desc="Parsing Dataset"):
        image_paths = glob(f"{folder}/*")
        clss = os.path.basename(folder)
        idx = mapping_dict[os.path.basename(folder)]

        paths += image_paths
        classes += [clss]*len(image_paths)
        class_indices += [idx]*len(image_paths)

    df['path'] = paths
    df['class'] = classes
    df['target'] = class_indices
    
    return df


def main(args):

    mapping_dict = class_mapping(f"{args.image_root_path}/train")

    os.makedirs(args.dataframe_output_path,exist_ok=True)
    train_image_root = f"{args.image_root_path}/train"
    train_df = get_dataframe(train_image_root,mapping_dict)
    train_df.to_csv(f"{args.dataframe_output_path}/train.csv",index=0)

    valid_image_root = f"{args.image_root_path}/test"
    valid_df = get_dataframe(valid_image_root,mapping_dict)
    valid_df.to_csv(f"{args.dataframe_output_path}/valid.csv",index=0)

if __name__=='__main__':

    args = parse()
    main(args)