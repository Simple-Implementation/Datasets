import os
import argparse
import pandas as pd

from tqdm import tqdm
from utils import *

def parse():

    parser = argparse.ArgumentParser()

    parser.add_argument("--object-detection", action='store_true')
    parser.add_argument("--semantic-segmentation", action='store_true')
    parser.add_argument("--keypoint-detection", action='store_true')
    parser.add_argument("--captions", action='store_true')
    parser.add_argument("--dataframe-output-path", type=str)
    
    args = parser.parse_args()

    return args

def object_detection(train_df, valid_df, roots):

    '''
        이미지 경로, 카테고리, 상위 카테고리, 바운딩 박스 좌표를 포함하는 데이터 프레임들을 반환합니다.
    '''

    train_json = f"{roots['annotations']}/instances_train2017.json"
    valid_json = f"{roots['annotations']}/instances_val2017.json"

    print(f"\n{FONT['g']}Load Train COCO Instance{FONT['reset']}")
    train_coco = get_coco_instance(train_json)
    train_image_ids = sorted(list(train_coco.imgs.keys()))

    new_train_df = pd.DataFrame()
    train_image_paths = list()
    train_categories = list()
    train_super_categories = list()
    train_top_left_xs = list()
    train_top_left_ys = list()
    train_widths = list()
    train_heights = list()
    
    for image_id in tqdm(train_image_ids,total=len(train_image_ids),desc='Looping Train Set'):

        image_path = get_image_path(image_id,roots['train_images'])

        annotation_ids = get_annotation_ids(train_coco,image_id=image_id)
        if len(annotation_ids) == 0: continue
        annotations = get_annotations(train_coco, annotation_ids)
        class_list = [
            get_category_name(
                get_category_info(
                    train_coco,annot['category_id']
                ) 
            )
            for annot in annotations
        ]
        # 2차원 리스트를 Transpose
        class_list = list(zip(*class_list))
        categories = "^".join(class_list[0])
        super_categories = "^".join(class_list[1])

        all_top_left_x = " ".join([str(annot['bbox'][0]) for annot in annotations])
        all_top_left_y = " ".join([str(annot['bbox'][1]) for annot in annotations])
        all_width = " ".join([str(annot['bbox'][2]) for annot in annotations])
        all_height = " ".join([str(annot['bbox'][3]) for annot in annotations])

        train_image_paths.append(image_path)
        train_categories.append(categories)
        train_super_categories.append(super_categories)
        train_top_left_xs.append(all_top_left_x)
        train_top_left_ys.append(all_top_left_y)
        train_widths.append(all_width)
        train_heights.append(all_height)

    new_train_df['path'] = train_image_paths
    new_train_df['categories'] = train_categories
    new_train_df['super_categories'] = train_super_categories
    new_train_df['top_left_xs'] = train_top_left_xs
    new_train_df['top_left_ys'] = train_top_left_ys
    new_train_df['widths'] = train_widths
    new_train_df['heights'] = train_heights

    if len(train_df) == 0:
        inner_joined_train_df = new_train_df
    else:
        inner_joined_train_df = pd.merge(train_df, new_train_df, how='inner', on='path')

    print(f"\n{FONT['g']}Load Valid COCO Instance{FONT['reset']}")
    valid_coco = get_coco_instance(valid_json)
    valid_image_ids = sorted(list(valid_coco.imgs.keys()))

    new_valid_df = pd.DataFrame()
    valid_image_paths = list()
    valid_categories = list()
    valid_super_categories = list()
    valid_top_left_xs = list()
    valid_top_left_ys = list()
    valid_widths = list()
    valid_heights = list()
    
    for image_id in tqdm(valid_image_ids,total=len(valid_image_ids),desc='Looping Valid Set'):

        image_path = get_image_path(image_id,roots['valid_images'])

        annotation_ids = get_annotation_ids(valid_coco,image_id=image_id)
        if len(annotation_ids) == 0: continue
        annotations = get_annotations(valid_coco, annotation_ids)
        class_list = [
            get_category_name(
                get_category_info(
                    valid_coco,annot['category_id']
                ) 
            )
            for annot in annotations
        ]
        # 2차원 리스트를 Transpose
        class_list = list(zip(*class_list))
        categories = "^".join(class_list[0])
        super_categories = "^".join(class_list[1])

        all_top_left_x = " ".join([str(annot['bbox'][0]) for annot in annotations])
        all_top_left_y = " ".join([str(annot['bbox'][1]) for annot in annotations])
        all_width = " ".join([str(annot['bbox'][2]) for annot in annotations])
        all_height = " ".join([str(annot['bbox'][3]) for annot in annotations])

        valid_image_paths.append(image_path)
        valid_categories.append(categories)
        valid_super_categories.append(super_categories)
        valid_top_left_xs.append(all_top_left_x)
        valid_top_left_ys.append(all_top_left_y)
        valid_widths.append(all_width)
        valid_heights.append(all_height)

    new_valid_df['path'] = valid_image_paths
    new_valid_df['categories'] = valid_categories
    new_valid_df['super_categories'] = valid_super_categories
    new_valid_df['top_left_xs'] = valid_top_left_xs
    new_valid_df['top_left_ys'] = valid_top_left_ys
    new_valid_df['widths'] = valid_widths
    new_valid_df['heights'] = valid_heights

    if len(valid_df) == 0:
        inner_joined_valid_df = new_valid_df
    else:
        inner_joined_valid_df = pd.merge(valid_df, new_valid_df, how='inner', on='path')

    return inner_joined_train_df, inner_joined_valid_df

def semantic_segmentation(train_df, valid_df, roots):

    pass

def keypoint_detection(train_df, valid_df, roots):

    pass

def captions(train_df, valid_df, roots):

    pass

def main(args):

    os.makedirs(args.dataframe_output_path,exist_ok=True)

    roots = {
        'annotations' : 'COCO2017/annotations_trainval2017/annotations',
        'train_images' : 'COCO2017/train2017',
        'valid_images' : 'COCO2017/val2017',
    }
    
    train_df = pd.DataFrame()
    valid_df = pd.DataFrame()

    if args.object_detection:
        train_df, valid_df = object_detection(train_df, valid_df, roots)
    if args.semantic_segmentation:
        train_df, valid_df = semantic_segmentation(train_df, valid_df, roots)
    if args.keypoint_detection:
        train_df, valid_df = keypoint_detection(train_df, valid_df, roots)
    if args.captions:
        train_df, valid_df = captions(train_df, valid_df, roots)
    
    train_df.to_csv(f"{args.dataframe_output_path}/train.csv",index=0)
    valid_df.to_csv(f"{args.dataframe_output_path}/valid.csv",index=0)

if __name__=='__main__':

    args = parse()
    main(args)