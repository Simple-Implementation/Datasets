import os
from colorama import Fore, Style

from pycocotools.coco import COCO
from pycocotools.mask import decode, frPyObjects

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

def get_coco_instance(json_path):

    '''
        COCO 클래스의 인스턴스를 반환합니다.
    '''

    coco = COCO(json_path)

    return coco

def get_annotation_ids(coco,image_id=-1,category_id=-1):

    '''
        이미지 아이디 혹은 카테고리 아이디로 어노테이션 아이디 리스트를 반환합니다.
    '''

    assert image_id!=-1 or category_id!=-1, f"Only One Of The Two,Image Id Or Category Id, Should Be Input"

    if image_id != -1:
        annotation_ids = coco.getAnnIds(imgIds=image_id)
    else:
        annotation_ids = coco.getAnnIds(catIds=category_id)

    return annotation_ids

def get_category_ids(coco, category_name='', super_category_name='', category_id=-1):

    '''
        카테고리 이름 혹은 상위 카테고리 이름 혹은 카테고리 아이디를 입력받아 해당 카테고리의 아이디를 반환합니다.
    '''
    
    assert category_name !='' or super_category_name != '' or category_id != -1 , f"Only One Of The Three,Category Name Or Super Category Name Or Category Id, Should Be Input"

    if category_name != '':
        category_ids = coco.getCatIds(catNms=category_name)
    elif super_category_name != '':
        category_ids = coco.getCatIds(supNms=super_category_name)
    else:
        category_ids = coco.getCatIds(catIds=category_id)

    return category_ids

def get_image_ids(coco, image_id=-1, category_id=-1):

    '''
        이미지 아이디 혹은 카테고리 아이디로 이미지 아이디 리스트를 반환합니다.
    '''

    assert image_id!=-1 or category_id!=-1, f"Only One Of The Two,Image Id Or Category Id, Should Be Input"

    if category_id != -1:
        image_ids = coco.getImgIds(catIds=category_id)
    else:
        image_ids = coco.getImgIds(imgIds=image_id)

    return image_ids
    
def get_annotations(coco, annotation_ids):

    '''
        어노테이션 아이디를 입력 받아 어노테이션 딕셔너리 전체를 반환합니다.
    '''

    annotation_list = coco.loadAnns(annotation_ids)

    return annotation_list

def get_category_info(coco, category_ids):

    '''
        카테고리 아이디를 입력 받아 해당 카테고리의 정보가 담긴 딕셔너리를 반환합니다.
    '''

    category_info = coco.loadCats(category_ids)

    return category_info

def get_category_name(category_info):

    '''
        카테고리 이름을 반환합니다.
    '''

    category_name = category_info[0]['name']
    super_category_name = category_info[0]['supercategory']

    return [category_name, super_category_name]

def get_image_info(coco, image_ids):

    '''
        이미지 아이디를 입력 받아 해당 이미지의 정보가 담긴 딕셔너리를 반환합니다.
    '''

    image_info = coco.loadImgs(image_ids)

    return image_info

def get_mask_from_rle(segmentation):

    '''
        어노테이션의 RLE를 이진 마스크로 변환합니다.
    '''

    height = segmentation['size'][0]
    width = segmentation['size'][1]
    rle = frPyObjects(segmentation, height, width)
    binary_mask = decode(rle)

    return binary_mask

def get_image_path(image_id, root_path):

    '''
        이미지 아이디와 이미지 루트 경로를 입력 받아 이미지의 경로를 반환합니다.
    '''

    image_name = f"{image_id:0>12}.jpg"

    return os.path.join(root_path,image_name)