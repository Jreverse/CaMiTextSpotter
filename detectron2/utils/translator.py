import math
import random
from turtle import forward

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.modeling.roi_heads.multilingual.language.languages import LANGUAGE_COMBO
from detectron2.modeling.roi_heads.multilingual.language.char_map_arabic import ArabicCharMap
from detectron2.structures.word_result import WordResult

gpu_device = torch.device("cuda")
cpu_device = torch.device("cpu")



# TODO
class MultilingualTranslator(nn.Module):
    def __init__(self,
            cfg, 
        ):
        super(MultilingualTranslator, self).__init__()
        self.cfg = cfg
        
        self.run_all_heads = cfg.TEST.RUN_ALL_HEADS
        if not self.run_all_heads:
            self.build_rec_head_map()
    
    def build_rec_head_map(self):
        if self.cfg.SEQUENCE.LANGUAGES_ENABLED == self.cfg.SEQUENCE.LANGUAGES:
            self.enabled_all_rec_heads = True
            self.rec_head_map = None
        else:
            self.enabled_all_rec_heads = False
            # rec_head_map is a 1-to-N mapping from recognition head to a subset of languages;
            # it's possible that some languages do not have a dedicated recognition head.
            self.rec_head_map = {}
            for rec_id, language_rec in enumerate(self.cfg.SEQUENCE.LANGUAGES_ENABLED):
                if language_rec in LANGUAGE_COMBO:
                    covered_language_set = set(LANGUAGE_COMBO[language_rec]) - set(
                        self.cfg.SEQUENCE.LANGUAGES_ENABLED
                    )
                    if language_rec in self.cfg.SEQUENCE.LANGUAGES:
                        # In this case, this unified rec head has
                        # its own corresponding output from LID
                        # and acts like a non-unified head
                        covered_language_set.add(language_rec)

                    assert len(covered_language_set) > 0, (
                        f"[Error] Rec-head seq_{language_rec} is unnecessary"
                        " since all sub-languages have dedicated heads"
                    )
                else:
                    covered_language_set = {language_rec}

                self.rec_head_map[rec_id] = [
                    id
                    for id, language in enumerate(self.cfg.SEQUENCE.LANGUAGES)
                    if language in covered_language_set
                ]

    def forward(
        self,
        rec,
        rec_score,
        rec_lang,
    ):
        # import ipdb;ipdb.set_trace()
        num_words = len(rec_lang)
        word_result_list = []
        assert num_words==len(rec) and num_words==len(rec_score)
        for k in range(num_words):
            word_result = WordResult()
            language_id = torch.argmax(rec_lang[k]).item()
            language_prob = rec_lang[k][language_id].item()
            seq_word = rec[k][language_id]
            
            score_list = [t for t in rec_score[k][language_id] if t >0]
            if len(score_list)>0:
                    word_result.seq_score = (sum(score_list) / float(len(score_list))).item()
            else:
                word_result.seq_score = 0.0
                
            # TODO: 考虑 enable 的问题
            
            # check if necessary
            for i in range(len(seq_word)):
                if ArabicCharMap.contain_char_exclusive(seq_word[i]):
                    seq_word = seq_word[::-1]
                    break

            word_result.seq_word = seq_word
            word_result.language_id = language_id
            word_result.language = self.cfg.SEQUENCE.LANGUAGES[language_id]
            word_result.language_prob = language_prob

            word_result_list.append(word_result)
        
        words = [t.seq_word for t in word_result_list]
        seq_scores = [t.seq_score for t in word_result_list]
        languages = [t.language for t in word_result_list]
        language_probs = [t.language_prob for t in word_result_list]
        
        return words, seq_scores, languages, language_probs
    