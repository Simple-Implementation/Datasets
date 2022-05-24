# ImageNet 1K

- Download Site
> - https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data
- Script

```
$ python ImageNet-1K/preprocess.py \
    --image-root-path ImageNet-1K/imagenet-object-localization-challenge/imagenet_object_localization_patched2019/ILSVRC/Data/CLS-LOC \
    --mapping-text-path ImageNet-1K/imagenet-object-localization-challenge/LOC_synset_mapping.txt \
    --valid-solution-path ImageNet-1K/imagenet-object-localization-challenge/LOC_val_solution.csv\
    --dataframe-output-path ImageNet-1K/dataframes
```

# ImageNet 21K

- Download Site
> - https://image-net.org/download-images.php

# Pascal VOC 2012

- Download Site
> - https://pjreddie.com/projects/pascal-voc-dataset-mirror/

# COCO

- Download Site
> - https://cocodataset.org/#download
- Data Format
> - https://cocodataset.org/#format-data
- Reference
> - https://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch
> - https://comlini8-8.tistory.com/67
- Script

```
$ python COCO2017/preprocess.py \
    --category-info \
    --object-detection \
    --semantic-segmentation \
    --keypoint-detection \
    --captions \
    --dataframe-output-path COCO2017/dataframes
```

# CIFAR-10

- Download Site
> - https://github.com/YoongiKim/CIFAR-10-images

# DIV2K

- Download Site
> - https://drive.google.com/file/d/1xAqZUsD2QdIofSOZP2cpmCgssyFXvMyj/view?usp=sharing

# Korean Wikipedia

- Download Site
> - https://ko.wikipedia.org/wiki/%EC%9C%84%ED%82%A4%EB%B0%B1%EA%B3%BC:%EB%8D%B0%EC%9D%B4%ED%84%B0%EB%B2%A0%EC%9D%B4%EC%8A%A4_%EB%8B%A4%EC%9A%B4%EB%A1%9C%EB%93%9C