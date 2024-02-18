import contextlib
import copy
import io
import itertools
import json
import logging
import numpy as np
import os
import re
import torch
from collections import OrderedDict
from fvcore.common.file_io import PathManager
from pycocotools.coco import COCO

import pycocotools.mask as mask_utils
from detectron2.utils import comm
from detectron2.data import MetadataCatalog
from detectron2.evaluation.evaluator import DatasetEvaluator

import math
import glob
import shutil
from shapely.geometry import Polygon, LinearRing
from detectron2.evaluation import text_eval_script
from detectron2.evaluation import text_eval_script_ic15
import zipfile
import pickle
import cv2
import editdistance
from detectron2.utils.translator import MultilingualTranslator

from detectron2.evaluation.mlt19.prepare_results import prepare_results_for_evaluation
from detectron2.evaluation.mlt19.task1.script import mlt19_eval_task1
from detectron2.evaluation.mlt19.task3.script import mlt19_eval_task3
from detectron2.evaluation.mlt19.task4.script import mlt19_eval_task4

import re

class TextMultilingualEvaluator(DatasetEvaluator):
    """
    Evaluate text proposals and recognition.
    """

    def __init__(self, dataset_name, cfg, distributed, output_dir=None):
        
        self.cfg = cfg
        self._tasks = ("polygon", "recognition")
        self._distributed = distributed
        self._output_dir = output_dir

        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)
        
        self.multilangual_translator = MultilingualTranslator(cfg)

        self.dataset_name = dataset_name
        # use dataset_name to decide eval_gt_path
        self.lexicon_type = 3
        if "MLT17" in dataset_name:
            self._text_eval_gt_path = "datasets-multi/evaluation/gt_mlt17.zip"
            if os.path.exists('/dataset/s3'):
                self._text_eval_gt_path = "/dataset/s3/cv_platform/JeremyFeng/datasets/multilingual/evaluation/gt_mlt17.zip"
            self._word_spotting = True
            self.dataset_name = "MLT17"
        elif "MLT19" in dataset_name:
            self._text_eval_gt_path = "datasets-multi/evaluation/gt_mlt19.zip"
            if os.path.exists('/dataset/s3'):
                self._text_eval_gt_path = "/dataset/s3/cv_platform/JeremyFeng/datasets/multilingual/evaluation/gt_mlt19.zip"
            self._word_spotting = True
            self.dataset_name = "MLT19"
        else:
            raise NotImplementedError
        
        self.split = dataset_name.split('_')[1]
        assert self.split in ['train', 'val', 'test']
        
        assert isinstance(cfg.DATASETS.TASK, tuple) and all([t in (1,3,4) for t in cfg.DATASETS.TASK]), 'Please select more than one eval task in (1,3,4,)'
        tasks = ['task{}'.format(str(t)) for t in cfg.DATASETS.TASK]
        self.eval_tasks = {}
        
        for task in tasks:
            self.eval_tasks[task] = eval('mlt19_eval_{}'.format(task))

        self._text_det_thresh = cfg.TEST.DET_CONF_THRESH
        self._text_lang_thresh = cfg.TEST.LANG_CONF_THRESH
        self._text_seq_thresh = cfg.TEST.SEQ_CONF_THRESH
        self.nms_enable = cfg.TEST.USE_NMS_IN_TSET
        
        self.save_intermediate_dir = 'tmp/{}/{}/{}/'.format(self.cfg.MODEL.WEIGHTS.split('/')[-1].split('.')[0], self.dataset_name, self.split)
        
    def lang_format(self, lang):
        formated_dict = {'ar': 'Arabic', 'la': 'Latin', 'zh': 'Chinese', 'ja': 'Japanese', 'ko': 'Korean', 'bn': 'Bangla', 'hi': 'Hindi', 'symbol': 'Symbols'}
        if lang not in formated_dict:
            return 'any'
        return formated_dict[lang]

    def reset(self):
        self._predictions = []

    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            prediction = {"image_id": input["image_id"]}
            instances = output["instances"].to(self._cpu_device)
            instances = instances[instances.scores > self._text_det_thresh]
            
            prediction["instances"] = self.instances_to_coco_json(instances, input)
            self._predictions.append(prediction)

    def evaluate_with_official_code(self, result_path, gt_path):
        if "icdar2015" in self.dataset_name:
            return text_eval_script_ic15.text_eval_main_ic15(det_file=result_path, gt_file=gt_path, is_word_spotting=self._word_spotting)
        else:
            return text_eval_script.text_eval_main(det_file=result_path, gt_file=gt_path, is_word_spotting=self._word_spotting)

    def gen_intermediate(self, predictions):
        intermediate_dir = os.path.join(self.save_intermediate_dir, 'intermediate')
        os.makedirs(intermediate_dir, exist_ok=True)
        for each in predictions:
            img_id, instances = each['image_id'], each['instances']
            pred_filename = 'res_img_{}.txt'.format(str(img_id).zfill(5))
            lines = []
            for single in instances:
                bbox, score, lang, lang_score, rec, rec_score = single['bbox'], \
                    single['score'], single['lang'], single['lang_score'], single['rec'], single['seq_score']
                output = "{},{},{},{},{},{}\n".format(
                    ",".join([str(x) for x in bbox]),
                    str("%.4f" % score),
                    str("%.4f" % rec_score),
                    str("%.4f" % lang_score),
                    self.lang_format(lang),
                    rec
                )
                lines.append(output)

            with open(os.path.join(intermediate_dir, pred_filename),'w') as f:
                f.writelines(lines)
        return intermediate_dir
        
    def evaluate(self):
        if self._distributed:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))

            if not comm.is_main_process():
                return {}
        else:
            predictions = self._predictions

        if len(predictions) == 0:
            self._logger.warning("[TextMultilingualEvaluator] Did not receive valid predictions.")
            return {}

        # coco_results = list(itertools.chain(*[x["instances"] for x in predictions]))
        # PathManager.mkdirs(self._output_dir)
        
        assert self.save_intermediate_dir!=None, 'need call gen_intermediate() first.'
        intermediate_dir = self.gen_intermediate(predictions)
        
        # file_path = os.path.join(self.save_intermediate_dir, "text_results.json")
        # self._logger.info("Saving results to {}".format(file_path))
        # with PathManager.open(file_path, "w") as f:
        #     f.write(json.dumps(coco_results))
        #     f.flush()
        
        # eval text
        assert os.path.exists(self._text_eval_gt_path), 'need gt file.'
        
        for task, eval_func in self.eval_tasks.items():
            # prepare data format for sub-task
            pred_zip_path = prepare_results_for_evaluation(
                results_dir = intermediate_dir,
                task = task,
                cache_dir = os.path.join(self.save_intermediate_dir, task),
                score_det = self._text_det_thresh,
                score_rec_seq = self._text_seq_thresh,
                overlap = 0.2,
                use_rec_seq=True,
                confidence_type="det",
                split=self.split,
                filter_heuristic="equal",
                lexicon=None,
                weighted_ed=True,
            )
            if self.split=='val':
                # call eval function
                eval_result = eval_func(
                    pred_zip_file=pred_zip_path,
                    gt_zip_file=self._text_eval_gt_path,
                    output_dir=None,
                    languages=None,
                )
                metrics = eval_result['method']
                self._logger.info('~~ {} Summary ~~'.format(task))
                for k, v in metrics.items():
                    self._logger.info('{}: {:.4f}'.format(k, v))
                    
                # return eval_result
            else:
                self._logger.info('Please submit the results in {} to icdar website manually.'.format(pred_zip_path))
            
        return {}

    def instances_to_coco_json(self, instances, inputs):
        img_id = inputs["image_id"]
        width = inputs['width']
        height = inputs['height']
        num_instances = len(instances)
        if num_instances == 0:
            return []
        scores = instances.scores.tolist()
        masks = np.asarray(instances.pred_masks)
        masks = [GenericMask(x, height, width) for x in masks]

        results = []
        i = 0
        
        # import ipdb;ipdb.set_trace()
        words, seq_scores, langs, lang_probs = self.multilangual_translator(instances.seq_outputs_list, instances.seq_scores_list, instances.language_probs)
        
        if self.nms_enable:
            polys = []
            # rboxs = []
            for mask in masks:
                if not len(mask.polygons):
                    continue
                # rbox = mask.rbox()
                # rboxs.append(rbox)
                polys.append(np.concatenate(mask.polygons).reshape(-1,2))
            keep = self.py_cpu_pnms(polys,scores,0.5)
        
        for mask, rec_string, lang_string, score, seq_score, lang_score in zip(masks, words, langs, scores, seq_scores, lang_probs):
            if not len(mask.polygons):
                continue
            if self.nms_enable:
                if i not in keep:
                    i = i+1
                    continue
            poly = polys[i].tolist()
            # rbox = rboxs[i]
                
            rbox = polygon2rbox(poly, height, width)
            
            if not len(rec_string):
                i = i+1
                continue
            result = {
                "image_id": img_id,
                "category_id": 1,
                "polys": poly,
                "bbox": rbox,
                "lang": lang_string,
                "rec": rec_string,
                "score": score,
                "lang_score": lang_score,
                "seq_score": seq_score,
            }
            results.append(result)
            i = i+1
        
        return results

    def py_cpu_pnms(self, dets, scores, thresh):
        pts = dets
        scores = np.array(scores)
        order = scores.argsort()[::-1]
        areas = np.zeros(scores.shape)
        order = scores.argsort()[::-1]
        inter_areas = np.zeros((scores.shape[0], scores.shape[0]))
        for il in range(len(pts)):
            poly = Polygon(pts[il]).buffer(0.001)
            areas[il] = poly.area
            for jl in range(il, len(pts)):
                polyj = Polygon(pts[jl].tolist()).buffer(0.001)
                inS = poly.intersection(polyj)
                try:
                    inter_areas[il][jl] = inS.area
                except:
                    import pdb;pdb.set_trace()
                inter_areas[jl][il] = inS.area

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            ovr = inter_areas[i][order[1:]] / ((areas[i]) + areas[order[1:]] - inter_areas[i][order[1:]])
            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]

        return keep

# cv2 numpy 操作耗时
def polygon2rbox(polygon, image_height, image_width):
    poly = np.array(polygon).reshape((-1, 2)).astype(np.float32)
    rect = cv2.minAreaRect(poly)
    corners = cv2.boxPoints(rect)
    corners = np.array(corners, dtype="int")
    pts = get_tight_rect(corners, 0, 0, image_height, image_width, 1)
    pts = np.array(pts)
    pts = pts.tolist()
    return pts

def get_tight_rect(points, start_x, start_y, image_height, image_width, scale):
    points = list(points)
    ps = sorted(points, key=lambda x: x[0])

    if ps[1][1] > ps[0][1]:
        px1 = ps[0][0] * scale + start_x
        py1 = ps[0][1] * scale + start_y
        px4 = ps[1][0] * scale + start_x
        py4 = ps[1][1] * scale + start_y
    else:
        px1 = ps[1][0] * scale + start_x
        py1 = ps[1][1] * scale + start_y
        px4 = ps[0][0] * scale + start_x
        py4 = ps[0][1] * scale + start_y
    if ps[3][1] > ps[2][1]:
        px2 = ps[2][0] * scale + start_x
        py2 = ps[2][1] * scale + start_y
        px3 = ps[3][0] * scale + start_x
        py3 = ps[3][1] * scale + start_y
    else:
        px2 = ps[3][0] * scale + start_x
        py2 = ps[3][1] * scale + start_y
        px3 = ps[2][0] * scale + start_x
        py3 = ps[2][1] * scale + start_y

    px1 = min(max(px1, 1), image_width - 1)
    px2 = min(max(px2, 1), image_width - 1)
    px3 = min(max(px3, 1), image_width - 1)
    px4 = min(max(px4, 1), image_width - 1)
    py1 = min(max(py1, 1), image_height - 1)
    py2 = min(max(py2, 1), image_height - 1)
    py3 = min(max(py3, 1), image_height - 1)
    py4 = min(max(py4, 1), image_height - 1)
    return [px1, py1, px2, py2, px3, py3, px4, py4]

class GenericMask:
    """
    Attribute:
        polygons (list[ndarray]): list[ndarray]: polygons for this mask.
            Each ndarray has format [x, y, x, y, ...]
        mask (ndarray): a binary mask
    """

    def __init__(self, mask_or_polygons, height, width):
        self._mask = self._polygons = self._has_holes = None
        self.height = height
        self.width = width

        m = mask_or_polygons
        if isinstance(m, dict):
            # RLEs
            assert "counts" in m and "size" in m
            if isinstance(m["counts"], list):  # uncompressed RLEs
                h, w = m["size"]
                assert h == height and w == width
                m = mask_utils.frPyObjects(m, h, w)
            self._mask = mask_utils.decode(m)[:, :]
            return

        if isinstance(m, list):  # list[ndarray]
            self._polygons = [np.asarray(x).reshape(-1) for x in m]
            return

        if isinstance(m, np.ndarray):  # assumed to be a binary mask
            assert m.shape[1] != 2, m.shape
            assert m.shape == (height, width), m.shape
            self._mask = m.astype("uint8")
            return

        raise ValueError("GenericMask cannot handle object {} of type '{}'".format(m, type(m)))

    @property
    def mask(self):
        if self._mask is None:
            self._mask = self.polygons_to_mask(self._polygons)
        return self._mask

    @property
    def polygons(self):
        if self._polygons is None:
            self._polygons, self._has_holes = self.mask_to_polygons(self._mask)
        return self._polygons

    @property
    def has_holes(self):
        if self._has_holes is None:
            if self._mask is not None:
                self._polygons, self._has_holes = self.mask_to_polygons(self._mask)
            else:
                self._has_holes = False  # if original format is polygon, does not have holes
        return self._has_holes

    def mask_to_polygons(self, mask):
        # cv2.RETR_CCOMP flag retrieves all the contours and arranges them to a 2-level
        # hierarchy. External contours (boundary) of the object are placed in hierarchy-1.
        # Internal contours (holes) are placed in hierarchy-2.
        # cv2.CHAIN_APPROX_NONE flag gets vertices of polygons from contours.
        mask = np.ascontiguousarray(mask)  # some versions of cv2 does not support incontiguous arr
        #res = cv2.findContours(mask.astype("uint8"), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        res = cv2.findContours(mask.astype("uint8"), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        hierarchy = res[-1]
        if hierarchy is None:  # empty mask
            return [], False
        has_holes = (hierarchy.reshape(-1, 4)[:, 3] >= 0).sum() > 0
        res = res[-2]
        res = [x.flatten() for x in res]
        # These coordinates from OpenCV are integers in range [0, W-1 or H-1].
        # We add 0.5 to turn them into real-value coordinate space. A better solution
        # would be to first +0.5 and then dilate the returned polygon by 0.5.
        res = [x + 0.5 for x in res if len(x) >= 6]
        return res, has_holes

    def polygons_to_mask(self, polygons):
        rle = mask_utils.frPyObjects(polygons, self.height, self.width)
        rle = mask_utils.merge(rle)
        return mask_utils.decode(rle)[:, :]

    def area(self):
        return self.mask.sum()

    def bbox(self):
        p = mask_utils.frPyObjects(self.polygons, self.height, self.width)
        p = mask_utils.merge(p)
        bbox = mask_utils.toBbox(p)
        bbox[2] += bbox[0]
        bbox[3] += bbox[1]
        return bbox
    
    def rbox(self):
        poly = self.polygons[0].reshape((-1,2))
        poly = np.int0(poly)
        rect = cv2.minAreaRect(poly)
        rect = cv2.boxPoints(rect)
        return np.int0(rect)

dictionary = "aàáạảãâầấậẩẫăằắặẳẵAÀÁẠẢÃĂẰẮẶẲẴÂẦẤẬẨẪeèéẹẻẽêềếệểễEÈÉẸẺẼÊỀẾỆỂỄoòóọỏõôồốộổỗơờớợởỡOÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠiìíịỉĩIÌÍỊỈĨuùúụủũưừứựửữƯỪỨỰỬỮUÙÚỤỦŨyỳýỵỷỹYỲÝỴỶỸ"


def make_groups():
    groups = []
    i = 0
    while i < len(dictionary) - 5:
        group = [c for c in dictionary[i : i + 6]]
        i += 6
        groups.append(group)
    return groups


groups = make_groups()

TONES = ["", "ˋ", "ˊ", "﹒", "ˀ", "˜"]
SOURCES = ["ă", "â", "Ă", "Â", "ê", "Ê", "ô", "ơ", "Ô", "Ơ", "ư", "Ư", "Đ", "đ"]
TARGETS = ["aˇ", "aˆ", "Aˇ", "Aˆ", "eˆ", "Eˆ", "oˆ", "o˒", "Oˆ", "O˒", "u˒", "U˒", "D-", "d‑"]


def correct_tone_position(word):
    word = word[:-1]
    if len(word) < 2:
        pass
    first_ord_char = ""
    second_order_char = ""
    for char in word:
        for group in groups:
            if char in group:
                second_order_char = first_ord_char
                first_ord_char = group[0]
    if word[-1] == first_ord_char and second_order_char != "":
        pair_chars = ["qu", "Qu", "qU", "QU", "gi", "Gi", "gI", "GI"]
        for pair in pair_chars:
            if pair in word and second_order_char in ["u", "U", "i", "I"]:
                return first_ord_char
        return second_order_char
    return first_ord_char


def vintext_decoder(recognition):
    for char in TARGETS:
        recognition = recognition.replace(char, SOURCES[TARGETS.index(char)])
    if len(recognition) < 1:
        return recognition
    if recognition[-1] in TONES:
        if len(recognition) < 2:
            return recognition
        replace_char = correct_tone_position(recognition)
        tone = recognition[-1]
        recognition = recognition[:-1]
        for group in groups:
            if replace_char in group:
                recognition = recognition.replace(replace_char, group[TONES.index(tone)])
    return recognition
