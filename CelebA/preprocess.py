import os
import argparse
import pandas as pd
from colorama import Fore, Style

FONT = {
    "r": Fore.RED,
    "g": Fore.GREEN,
    "b": Fore.BLUE,
    "y": Fore.YELLOW,
    "m": Fore.MAGENTA,
    "c": Fore.CYAN,
    "start": '\033[1m',
    "end": '\033[0m',
    "reset": Style.RESET_ALL,
}

def parse():

    parser = argparse.ArgumentParser()

    parser.add_argument("--image-root-path", type=str)
    parser.add_argument("--landmark-data-path", type=str)
    parser.add_argument("--dataframe-output-path", type=str)
    parser.add_argument("--partition-data-path", type=str)

    args = parser.parse_args()

    return args

def parse_landmark_data(image_root,landmark_file,partition_file):

    filenames = list()
    categories = list()

    with open(partition_file, 'r') as f:
        while True:
            line = f.readline()
            if not line: break
            filename, category = line.strip("\n").split(" ")
            filenames.append(filename)
            categories.append(int(category))

    category_df = pd.DataFrame()
    category_df['path'] = filenames
    category_df['category'] = categories

    with open(landmark_file, 'r') as f:
        num_data = int(f.readline())
        cols = f.readline().strip("\n").split(" ")
        cols = ['path'] + cols
        data = [[] for _ in range(len(cols))]
        while True:
            line = f.readline()
            if not line: break
            line = line.strip("\n").split(" ")
            line = list(filter(None,line))
            for idx, d in enumerate(line):
                data[idx].append(d)

    df = pd.DataFrame()
    for col, d in zip(cols,data):
        df[col] = d

    df = pd.merge(left=df,right=category_df,on='path',how='inner')
    df['path'] = df['path'].apply(lambda x: os.path.join(image_root,x))

    train_df = df[df['category']==0].drop('category', axis=1).reset_index(drop=True)
    valid_df = df[df['category']==1].drop('category', axis=1).reset_index(drop=True)
    test_df = df[df['category']==2].drop('category', axis=1).reset_index(drop=True)

    return train_df, valid_df, test_df

def main(args):

    os.makedirs(args.dataframe_output_path,exist_ok=True)
    train_df, valid_df, test_df = parse_landmark_data(args.image_root_path,args.landmark_data_path,args.partition_data_path)
    train_df.to_csv(f"{args.dataframe_output_path}/train.csv",index=0)
    valid_df.to_csv(f"{args.dataframe_output_path}/valid.csv",index=0)
    test_df.to_csv(f"{args.dataframe_output_path}/test.csv",index=0)

    print(f"\n{FONT['r']}{FONT['start']}Done.\n{FONT['end']}")

if __name__=='__main__':

    args = parse()
    main(args)