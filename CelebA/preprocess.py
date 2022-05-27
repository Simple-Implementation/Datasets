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

    args = parser.parse_args()

    return args

def parse_landmark_data(image_root,landmark_file):

    with open(landmark_file, 'r') as f:
        num_data = int(f.readline())
        cols = f.readline().strip("\n").split(" ")
        cols = ['filename'] + cols
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

    df['filename'] = df['filename'].apply(lambda x: os.path.join(image_root,x))

    return df

def main(args):

    os.makedirs(args.dataframe_output_path,exist_ok=True)
    df = parse_landmark_data(args.image_root_path,args.landmark_data_path)
    df.to_csv(f"{args.dataframe_output_path}/train.csv",index=0)

    print(f"\n{FONT['r']}{FONT['start']}Done.\n{FONT['end']}")

if __name__=='__main__':

    args = parse()
    main(args)