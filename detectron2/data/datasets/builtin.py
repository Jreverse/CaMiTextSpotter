# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.


"""
This file registers pre-defined datasets at hard-coded paths, and their metadata.

We hard-code metadata for common datasets. This will enable:
1. Consistency check when loading the datasets
2. Use models on these standard datasets directly and run demos,
   without having to download the dataset annotations

We hard-code some paths to the dataset that's assumed to
exist in "./datasets/".

Users SHOULD NOT use this file to create new dataset / metadata for new dataset.
To add new dataset, refer to the tutorial "docs/DATASETS.md".
"""

import os

from detectron2.data import DatasetCatalog, MetadataCatalog

from .builtin_meta import ADE20K_SEM_SEG_CATEGORIES, _get_builtin_metadata
from .cityscapes import load_cityscapes_instances, load_cityscapes_semantic
from .cityscapes_panoptic import register_all_cityscapes_panoptic
from .coco import load_sem_seg, register_coco_instances
from .coco_panoptic import register_coco_panoptic, register_coco_panoptic_separated
from .lvis import get_lvis_instances_meta, register_lvis_instances
from .pascal_voc import register_pascal_voc

# ==== Predefined datasets and splits for COCO ==========

_PREDEFINED_SPLITS_COCO = {}
_PREDEFINED_SPLITS_COCO["coco"] = {
    # debug
    'bug_rec_gt_MLT19synth': ("debug/bug_rec_gt_MLT19synth", "debug/bug_rec_gt_MLT19synth.json"),
    'bug_det_loss_MLT19': ("debug/bug_det_loss_MLT19", "debug/bug_det_loss_MLT19.json"),
    'multilang_debug': ("MLT19/train/imgs", "custom_data/annotations/train.json"),
    'ocronly_debug': ("debug/bug_for_ocronly", "debug/bug_for_ocronly.json"),

    # ICDAR 2015
    "icdar_2015_train": ("icdar2015/train/imgs", "icdar2015/annotations/multilingual_train2.json"),
    "icdar_2015_test": ("icdar2015/test/imgs", "icdar2015/annotations/multilingual_test2.json"),
    # ICDAR 2013
    "icdar_2013_train": ("icdar2013/train/imgs", "icdar2013/annotations/multilingual_train2.json"),
    # RCTW17
    "RCTW17_train": ("RCTW/train_images", "RCTW/annotations/multilingual_train2.json"),
    # LSVT
    "LSVT19_train": ("LSVT/train_images", "LSVT/annotations/multilingual_train2.json"),
    # ArT19
    "ArT19_train": ("ArT19/train_images", "ArT19/annotations/multilingual_train2.json"),
    # totaltext
    "totaltext_train": ("totaltext/train_images", "totaltext/annotations/multilingual_train2.json"),
    "totaltext_test": ("totaltext/test_images", "totaltext/annotations/multilingual_test2.json"),
    # # ctw1500
    # "ctw1500_train": ("CTW1500/ctwtrain_text_image", "CTW1500/traint_ctw1500_maxlen100_v2.json"),
    # "ctw1500_test": ("CTW1500/ctwtest_text_image", "CTW1500/test_ctw1500_maxlen100.json"),
    # vintext
    "vintext_train": ("vintext/train/imgs", "vintext/annotations/multilingual_train2.json"),
    "vintext_test": ("vintext/test/imgs", "vintext/annotations/multilingual_test2.json"),
    "vintext_val": ("vintext/val/imgs", "vintext/annotations/multilingual_val2.json"),
    # ICDAR 2019 MLT
    'MLT19_train': ("MLT19/train/imgs", "MLT19/annotations/multilingual_train.json"),
    'MLT19_val': ("MLT19/val/imgs", "MLT19/annotations/multilingual_val.json"),
    'MLT19_test': ("MLT19/test/imgs", "MLT19/annotations/multilingual_test.json"),
    # ICDAR 2019 MLT Synthetic
    'MLT19_Synthetic': ("MLT19_Synthetic", "MLT19_Synthetic/annotations/multilingual_train.json"),
    # ICDAR 2017 MLT
    "MLT17_train": ("MLT17/train/imgs", "MLT17/annotations/multilingual_train2.json"),
    "MLT17_val": ("MLT17/val/imgs", "MLT17/annotations/multilingual_val2.json"),
}

_PREDEFINED_SPLITS_COCO["coco_coeus"] = {
    # ICDAR 2015
    "icdar_2015_train": ("/dataset/s3/cv_platform/JeremyFeng/datasets/multilingual/icdar2015/train/imgs", "/dataset/s3/cv_platform/JeremyFeng/datasets/multilingual/icdar2015/annotations/multilingual_train2.json"),
    "icdar_2015_test": ("/dataset/s3/cv_platform/JeremyFeng/datasets/multilingual/icdar2015/test/imgs", "/dataset/s3/cv_platform/JeremyFeng/datasets/multilingual/icdar2015/annotations/multilingual_test2.json"),
    # ICDAR 2013
    "icdar_2013_train": ("/dataset/s3/cv_platform/JeremyFeng/datasets/multilingual/icdar2013/train/imgs", "/dataset/s3/cv_platform/JeremyFeng/datasets/multilingual/icdar2013/annotations/multilingual_train2.json"),
    # RCTW17
    "RCTW17_train": ("/dataset/s3/cv_platform/JeremyFeng/datasets/multilingual/RCTW/train_images", "/dataset/s3/cv_platform/JeremyFeng/datasets/multilingual/RCTW/annotations/multilingual_train2.json"),
    # LSVT
    "LSVT19_train": ("/dataset/s3/cv_platform/JeremyFeng/datasets/multilingual/LSVT/train_images", "/dataset/s3/cv_platform/JeremyFeng/datasets/multilingual/LSVT/annotations/multilingual_train2.json"),
    # ArT19
    "ArT19_train": ("/dataset/s3/cv_platform/JeremyFeng/datasets/multilingual/ArT19/train_images", "/dataset/s3/cv_platform/JeremyFeng/datasets/multilingual/ArT19/annotations/multilingual_train2.json"),
    # totaltext
    "totaltext_train": ("/dataset/s3/cv_platform/JeremyFeng/datasets/multilingual/totaltext/train_images", "/dataset/s3/cv_platform/JeremyFeng/datasets/multilingual/totaltext/annotations/multilingual_train2.json"),
    "totaltext_test": ("/dataset/s3/cv_platform/JeremyFeng/datasets/multilingual/totaltext/test_images", "/dataset/s3/cv_platform/JeremyFeng/datasets/multilingual/totaltext/annotations/multilingual_test2.json"),
    # # ctw1500
    # "ctw1500_train": ("CTW1500/ctwtrain_text_image", "CTW1500/traint_ctw1500_maxlen100_v2.json"),
    # "ctw1500_test": ("CTW1500/ctwtest_text_image", "CTW1500/test_ctw1500_maxlen100.json"),
    # vintext
    "vintext_train": ("/dataset/s3/cv_platform/JeremyFeng/datasets/multilingual/vintext/train/imgs", "/dataset/s3/cv_platform/JeremyFeng/datasets/multilingual/vintext/annotations/multilingual_train2.json"),
    "vintext_test": ("/dataset/s3/cv_platform/JeremyFeng/datasets/multilingual/vintext/test/imgs", "/dataset/s3/cv_platform/JeremyFeng/datasets/multilingual/vintext/annotations/multilingual_test2.json"),
    "vintext_val": ("/dataset/s3/cv_platform/JeremyFeng/datasets/multilingual/vintext/val/imgs", "/dataset/s3/cv_platform/JeremyFeng/datasets/multilingual/vintext/annotations/multilingual_val2.json"),
    # ICDAR 2017 MLT
    "MLT17_train": ("/dataset/s3/cv_platform/JeremyFeng/datasets/multilingual/MLT17/train/imgs", "/dataset/s3/cv_platform/JeremyFeng/datasets/multilingual/MLT17/annotations/multilingual_train2.json"),
    "MLT17_val": ("/dataset/s3/cv_platform/JeremyFeng/datasets/multilingual/MLT17/val/imgs", "/dataset/s3/cv_platform/JeremyFeng/datasets/multilingual/MLT17/annotations/multilingual_val2.json"),
    # ICDAR 2019 MLT
    'MLT19_train': ("/dataset/s3/cv_platform/JeremyFeng/datasets/multilingual/MLT19/train/imgs", "/dataset/s3/cv_platform/JeremyFeng/datasets/multilingual/MLT19/annotations/multilingual_train.json"),
    'MLT19_val': ("/dataset/s3/cv_platform/JeremyFeng/datasets/multilingual/MLT19/val/imgs", "/dataset/s3/cv_platform/JeremyFeng/datasets/multilingual/MLT19/annotations/multilingual_val.json"),
    'MLT19_test': ("/dataset/s3/cv_platform/JeremyFeng/datasets/multilingual/MLT19/test/imgs", "/dataset/s3/cv_platform/JeremyFeng/datasets/multilingual/MLT19/annotations/multilingual_test.json"),
    'MLT19_Synthetic': ("/dataset/s3/cv_platform/JeremyFeng/datasets/multilingual/MLT19_Synthetic", "/dataset/s3/cv_platform/JeremyFeng/datasets/multilingual/MLT19_Synthetic/annotations/multilingual_train.json"),
}


# dataset anno keys
# ANNO_KEYS = ["iscrowd", "bbox", "keypoints", "category_id", "rec"] 
ANNO_KEYS = ["iscrowd", "bbox", "keypoints", "category_id",  "text", "rec", "language"]
USE_COEUS_PATH = os.path.exists('/dataset/s3/cv_platform/JeremyFeng/')

def register_all_coco(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_COCO.items():
        if USE_COEUS_PATH and not dataset_name.endswith('_coeus'):
            continue
        if not USE_COEUS_PATH and dataset_name.endswith('_coeus'):
            continue
        for key, (image_root, json_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_coco_instances(
                key,
                _get_builtin_metadata(dataset_name),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
                ANNO_KEYS,
            )


# ==== Predefined datasets and splits for LVIS ==========


_PREDEFINED_SPLITS_LVIS = {
    "lvis_v1": {
        "lvis_v1_train": ("coco/", "lvis/lvis_v1_train.json"),
        "lvis_v1_val": ("coco/", "lvis/lvis_v1_val.json"),
        "lvis_v1_test_dev": ("coco/", "lvis/lvis_v1_image_info_test_dev.json"),
        "lvis_v1_test_challenge": ("coco/", "lvis/lvis_v1_image_info_test_challenge.json"),
    },
    "lvis_v0.5": {
        "lvis_v0.5_train": ("coco/", "lvis/lvis_v0.5_train.json"),
        "lvis_v0.5_val": ("coco/", "lvis/lvis_v0.5_val.json"),
        "lvis_v0.5_val_rand_100": ("coco/", "lvis/lvis_v0.5_val_rand_100.json"),
        "lvis_v0.5_test": ("coco/", "lvis/lvis_v0.5_image_info_test.json"),
    },
    "lvis_v0.5_cocofied": {
        "lvis_v0.5_train_cocofied": ("coco/", "lvis/lvis_v0.5_train_cocofied.json"),
        "lvis_v0.5_val_cocofied": ("coco/", "lvis/lvis_v0.5_val_cocofied.json"),
    },
}


def register_all_lvis(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_LVIS.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            register_lvis_instances(
                key,
                get_lvis_instances_meta(dataset_name),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )


# ==== Predefined splits for raw cityscapes images ===========
_RAW_CITYSCAPES_SPLITS = {
    "cityscapes_fine_{task}_train": ("cityscapes/leftImg8bit/train/", "cityscapes/gtFine/train/"),
    "cityscapes_fine_{task}_val": ("cityscapes/leftImg8bit/val/", "cityscapes/gtFine/val/"),
    "cityscapes_fine_{task}_test": ("cityscapes/leftImg8bit/test/", "cityscapes/gtFine/test/"),
}


def register_all_cityscapes(root):
    for key, (image_dir, gt_dir) in _RAW_CITYSCAPES_SPLITS.items():
        meta = _get_builtin_metadata("cityscapes")
        image_dir = os.path.join(root, image_dir)
        gt_dir = os.path.join(root, gt_dir)

        inst_key = key.format(task="instance_seg")
        DatasetCatalog.register(
            inst_key,
            lambda x=image_dir, y=gt_dir: load_cityscapes_instances(
                x, y, from_json=True, to_polygons=True
            ),
        )
        MetadataCatalog.get(inst_key).set(
            image_dir=image_dir, gt_dir=gt_dir, evaluator_type="cityscapes_instance", **meta
        )

        sem_key = key.format(task="sem_seg")
        DatasetCatalog.register(
            sem_key, lambda x=image_dir, y=gt_dir: load_cityscapes_semantic(x, y)
        )
        MetadataCatalog.get(sem_key).set(
            image_dir=image_dir,
            gt_dir=gt_dir,
            evaluator_type="cityscapes_sem_seg",
            ignore_label=255,
            **meta,
        )


# ==== Predefined splits for PASCAL VOC ===========
def register_all_pascal_voc(root):
    SPLITS = [
        ("voc_2007_trainval", "VOC2007", "trainval"),
        ("voc_2007_train", "VOC2007", "train"),
        ("voc_2007_val", "VOC2007", "val"),
        ("voc_2007_test", "VOC2007", "test"),
        ("voc_2012_trainval", "VOC2012", "trainval"),
        ("voc_2012_train", "VOC2012", "train"),
        ("voc_2012_val", "VOC2012", "val"),
    ]
    for name, dirname, split in SPLITS:
        year = 2007 if "2007" in name else 2012
        register_pascal_voc(name, os.path.join(root, dirname), split, year)
        MetadataCatalog.get(name).evaluator_type = "pascal_voc"


def register_all_ade20k(root):
    root = os.path.join(root, "ADEChallengeData2016")
    for name, dirname in [("train", "training"), ("val", "validation")]:
        image_dir = os.path.join(root, "images", dirname)
        gt_dir = os.path.join(root, "annotations_detectron2", dirname)
        name = f"ade20k_sem_seg_{name}"
        DatasetCatalog.register(
            name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="png", image_ext="jpg")
        )
        MetadataCatalog.get(name).set(
            stuff_classes=ADE20K_SEM_SEG_CATEGORIES[:],
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=255,
        )


# True for open source;
# Internally at fb, we register them elsewhere
if __name__.endswith(".builtin"):
    # Assume pre-defined datasets live in `./datasets`.
    _root = os.getenv("DETECTRON2_DATASETS", "datasets-multi")
    # _root = os.getenv("DETECTRON2_DATASETS", "datasets")
    register_all_coco(_root)
    register_all_lvis(_root)
    register_all_cityscapes(_root)
    register_all_cityscapes_panoptic(_root)
    register_all_pascal_voc(_root)
    register_all_ade20k(_root)
