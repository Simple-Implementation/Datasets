import os
import cv2
import json
import argparse
import numpy as np
import pandas as pd

from tqdm import tqdm
from utils import *

def parse():

    parser = argparse.ArgumentParser()

    parser.add_argument("--category-info", action='store_true')
    parser.add_argument("--object-detection", action='store_true')
    parser.add_argument("--semantic-segmentation", action='store_true')
    parser.add_argument("--keypoint-detection", action='store_true')
    parser.add_argument("--caption", action='store_true')
    parser.add_argument("--dataframe-output-path", type=str)
    
    args = parser.parse_args()

    return args

def object_detection(df, roots, mode='valid'):

    '''
        이미지 경로, 카테고리, 상위 카테고리, 바운딩 박스 좌표를 포함하는 데이터 프레임들을 반환합니다.
    '''

    json_name = 'train' if mode=='train' else 'val'
    mode_name = 'Train' if mode=='train' else 'Valid'
    json_path = f"{roots['annotations']}/instances_{json_name}2017.json"

    print(f"\n{FONT['g']}{mode_name} COCO Instance For {FONT['start']}Object Detection{FONT['reset']}")
    coco = get_coco_instance(json_path)
    image_ids = sorted(list(coco.imgs.keys()))

    new_df = pd.DataFrame()
    df_image_paths = list()
    df_categories = list()
    df_super_categories = list()
    df_top_left_xs = list()
    df_top_left_ys = list()
    df_widths = list()
    df_heights = list()
    
    for image_id in tqdm(image_ids,total=len(image_ids),desc=f'{FONT["y"]}Looping {mode_name} Set{FONT["reset"]}'):

        image_path = get_image_path(image_id,roots[f'{mode}_images'])

        annotation_ids = get_annotation_ids(coco,image_id=image_id)
        # 어노테이션이 없는 데이터는 건너뜀
        if len(annotation_ids) == 0: continue
        annotations = get_annotations(coco, annotation_ids)
        class_list = [
            get_category_name(
                get_category_info(
                    coco,annot['category_id']
                ) 
            )
            for annot in annotations
        ]
        # 2차원 리스트를 Transpose
        class_list = list(zip(*class_list))
        categories = " ".join([cat.replace(" ","_") for cat in class_list[0]])
        super_categories = " ".join([sup_cat.replace(" ","_") for sup_cat in class_list[1]])

        all_top_left_x = " ".join([str(annot['bbox'][0]) for annot in annotations])
        all_top_left_y = " ".join([str(annot['bbox'][1]) for annot in annotations])
        all_width = " ".join([str(annot['bbox'][2]) for annot in annotations])
        all_height = " ".join([str(annot['bbox'][3]) for annot in annotations])

        df_image_paths.append(image_path)
        df_categories.append(categories)
        df_super_categories.append(super_categories)
        df_top_left_xs.append(all_top_left_x)
        df_top_left_ys.append(all_top_left_y)
        df_widths.append(all_width)
        df_heights.append(all_height)

    new_df['image_path'] = df_image_paths
    new_df['categories'] = df_categories
    new_df['super_categories'] = df_super_categories
    new_df['top_left_xs'] = df_top_left_xs
    new_df['top_left_ys'] = df_top_left_ys
    new_df['box_widths'] = df_widths
    new_df['box_heights'] = df_heights

    if len(df) == 0:
        inner_joined_df = new_df
    else:
        inner_joined_df = pd.merge(df, new_df, how='inner', on='image_path')

    return inner_joined_df

def semantic_segmentation(df, roots, mode='valid'):

    '''
        이미지 경로, 카테고리, 상위 카테고리, 마스크 경로 등을 포함하는 데이터 프레임들을 반환합니다.
    '''

    json_name = 'train' if mode=='train' else 'val'
    mode_name = 'Train' if mode=='train' else 'Valid'

    mask_root = f"{roots[f'{mode}_masks']}"
    os.makedirs(mask_root,exist_ok=True)

    json_path = f"{roots['annotations']}/instances_{json_name}2017.json"

    print(f"\n{FONT['g']}{mode_name} COCO Instance For {FONT['start']}Semantic Segmentation{FONT['reset']}")
    coco = get_coco_instance(json_path)
    image_ids = sorted(list(coco.imgs.keys()))

    new_df = pd.DataFrame()
    df_image_paths = list()
    df_categories = list()
    df_super_categories = list()
    df_image_widths = list()
    df_image_heights = list()
    df_mask_paths = list()

    for image_id in tqdm(image_ids,total=len(image_ids),desc=f'{FONT["y"]}Looping {mode_name} Set{FONT["reset"]}'):

        image_path = get_image_path(image_id,roots[f'{mode}_images'])
        mask_path = os.path.join(mask_root, os.path.basename(image_path))
        annotation_ids = get_annotation_ids(coco,image_id=image_id)
        # 어노테이션이 없는 데이터는 건너뜀
        if len(annotation_ids) == 0: continue
        annotations = get_annotations(coco, annotation_ids)
        class_list = [
            get_category_name(
                get_category_info(
                    coco,annot['category_id']
                ) 
            )
            for annot in annotations
        ]
        # 2차원 리스트를 Transpose
        class_list = list(zip(*class_list))
        category_ids = [annot['category_id'] for annot in annotations]
        categories = " ".join([cat.replace(" ","_") for cat in class_list[0]])
        super_categories = " ".join([sup_cat.replace(" ","_") for sup_cat in class_list[1]])

        image_info = get_image_info(coco, image_id)[0]
        image_width = image_info['width']
        image_height = image_info['height']
        new_image = np.zeros((image_height,image_width),dtype=np.uint8) # 검은색 이미지 

        is_crowds = [annot['iscrowd'] for annot in annotations]
        segmentations = [annot['segmentation'] for annot in annotations]

        for is_crowd, segmentation, category_id in zip(is_crowds, segmentations, category_ids):
            # (x,y) 좌표로 구성된 리스트
            if is_crowd==0:
                segmentation = segmentation[0]
                segmentation = np.array([(round(segmentation[i]),round(segmentation[i+1])) for i in range(0,len(segmentation),2)])
                cv2.fillPoly(new_image,[segmentation],category_id)
            # Run-Length-Encoding 스타일 
            else:
                mask = get_mask_from_rle(segmentation)*category_id # 2차원 배열
                new_image += mask
        cv2.imwrite(mask_path,new_image)

        df_image_paths.append(image_path)
        df_categories.append(categories)
        df_super_categories.append(super_categories)
        df_image_widths.append(image_width)
        df_image_heights.append(image_height)
        df_mask_paths.append(mask_path)

    new_df['image_path'] = df_image_paths
    new_df['categories'] = df_categories
    new_df['super_categories'] = df_super_categories
    new_df['mask_path'] = df_mask_paths
    new_df['image_widths'] = df_image_widths
    new_df['image_heights'] = df_image_heights

    if len(df) == 0:
        inner_joined_df = new_df
    else:
        inner_joined_df = pd.merge(df, new_df, how='inner', on='image_path')

    return inner_joined_df

def keypoint_detection(train_df, valid_df, roots):

    pass

def caption(train_df, valid_df, roots):

    pass

def save_category_information(args, json_path):

    '''
        카테고리 정보들을 csv 파일로 저장합니다.
    '''
    
    category_path = os.path.join(args.dataframe_output_path,"categories.csv")
    print(f"\n{FONT['r']}{FONT['start']}Save Category Information{FONT['reset']}: {category_path}")
    with open(json_path, 'r') as j:
        data = json.load(j)
    
    ids = list()
    super_categories = list()
    categories = list()
    for category in data['categories']:
        ids.append(category['id'])
        super_categories.append(category['supercategory'])
        categories.append(category['name'])

    category_df = pd.DataFrame()
    category_df['id'] = ids
    category_df['supercategory'] = super_categories
    category_df['category'] = categories
    category_df.to_csv(category_path,index=0)

def main(args):

    os.makedirs(args.dataframe_output_path,exist_ok=True)

    roots = {
        'annotations' : 'COCO2017/annotations_trainval2017/annotations',
        'train_images' : 'COCO2017/train2017',
        'valid_images' : 'COCO2017/val2017',
        'train_masks' : 'COCO2017/train_masks',
        'valid_masks' : 'COCO2017/val_masks',
    }
    
    train_df = pd.DataFrame()
    valid_df = pd.DataFrame()
    train_csv_name = 'train.csv'
    valid_csv_name = 'valid.csv'

    if args.category_info:
        save_category_information(args, 'COCO2017/annotations_trainval2017/annotations/instances_val2017.json')

    if args.object_detection:
        train_df = object_detection(train_df, roots, 'train')
        valid_df = object_detection(valid_df, roots, 'valid')
        train_csv_name = 'object_detection_' + train_csv_name
        valid_csv_name = 'object_detection_' + valid_csv_name
    if args.semantic_segmentation:
        train_df= semantic_segmentation(train_df, roots, 'train')
        valid_df= semantic_segmentation(valid_df, roots, 'valid')
        train_csv_name = 'semantic_segmentation_' + train_csv_name
        valid_csv_name = 'semantic_segmentation_' + valid_csv_name
    if args.keypoint_detection:
        train_df= keypoint_detection(train_df, roots, 'train')
        valid_df= keypoint_detection(valid_df, roots, 'valid')
        train_csv_name = 'keypoint_detection_' + train_csv_name
        valid_csv_name = 'keypoint_detection_' + valid_csv_name
    if args.caption:
        train_df= caption(train_df, roots, 'train')
        valid_df= caption(valid_df, roots, 'valid')
        train_csv_name = 'caption_' + train_csv_name
        valid_csv_name = 'caption_' + valid_csv_name
    
    train_df.to_csv(f"{args.dataframe_output_path}/{train_csv_name}",index=0)
    valid_df.to_csv(f"{args.dataframe_output_path}/{valid_csv_name}",index=0)

if __name__=='__main__':

    args = parse()
    main(args)