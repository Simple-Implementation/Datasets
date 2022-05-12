import os
import argparse
import pandas as pd

from glob import glob
from tqdm import tqdm

def parse():

    parser = argparse.ArgumentParser()

    parser.add_argument("--image-root-path", type=str)
    parser.add_argument("--mapping-text-path", type=str)
    parser.add_argument("--dataframe-output-path", type=str)
    parser.add_argument("--valid-solution-path", type=str)
    
    args = parser.parse_args()


    return args

def get_mapping_dict(mapping_text_path):

    mapping_dict = dict()

    class_idx = 0
    with open(mapping_text_path,'r') as f:
        while True:
            line = f.readline()
            if not line: break
            items = line.split(" ")
            idx = items[0]
            clss = items[1].split(',')[0].strip('\n')
            mapping_dict[idx] = [clss, class_idx]
            class_idx += 1

    return mapping_dict

def get_train_dataframe(image_root_path, mapping_dict):

    df = pd.DataFrame()

    classes = list()
    paths = list()
    class_indices = list()

    folders = glob(f"{image_root_path}/*")
    for folder in tqdm(folders,total=len(folders),desc="Parsing Train Dataset"):
        image_paths = glob(f"{folder}/*")
        class_info = mapping_dict[os.path.basename(folder)]
        clss = [class_info[0]]*len(image_paths)
        indices = [class_info[1]]*len(image_paths)

        paths += image_paths
        classes += clss
        class_indices += indices

    df['path'] = paths
    df['class'] = classes
    df['target'] = class_indices
    
    return df

def get_valid_dataframe(image_root_path, valid_solution_path, mapping_dict):

    solution_df = pd.read_csv(valid_solution_path)
    df = pd.DataFrame()

    classes = list()
    class_indices = list()

    image_paths = glob(f"{image_root_path}/*")
    for path in tqdm(image_paths,total=len(image_paths),desc="Parsing Validation Dataset"):
        image_id = os.path.basename(path).split('.')[0]
        class_info = mapping_dict[
            solution_df[solution_df['ImageId']==image_id]['PredictionString'].values[0].split(" ")[0]
        ]
        clss = class_info[0]
        idx = class_info[1]

        classes.append(clss)
        class_indices.append(idx)

    df['path'] = image_paths
    df['class'] = classes
    df['target'] = class_indices

    return df

def main(args):

    mapping_dict = get_mapping_dict(args.mapping_text_path)

    os.makedirs(args.dataframe_output_path,exist_ok=True)
    train_image_root = f"{args.image_root_path}/train"
    train_df = get_train_dataframe(train_image_root,mapping_dict)
    train_df.to_csv(f"{args.dataframe_output_path}/train.csv",index=0)

    valid_image_root = f"{args.image_root_path}/val"
    valid_df = get_valid_dataframe(valid_image_root,args.valid_solution_path,mapping_dict)
    valid_df.to_csv(f"{args.dataframe_output_path}/valid.csv",index=0)

if __name__=='__main__':

    args = parse()
    main(args)