# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

# from .language_predictor_v0 import V0LanguagePredictor
from .language_predictor import V1LanguagePredictor, V2LanguagePredictor, V3LanguagePredictor


_LANGUAGE_PREDICTOR = {
    "V1LanguagePredictor": V1LanguagePredictor,
    "V2LanguagePredictor": V2LanguagePredictor,
    "V3LanguagePredictor": V3LanguagePredictor,
}


def make_language_predictor(cfg):
    predictor = _LANGUAGE_PREDICTOR[cfg.MODEL.LANGUAGE_HEAD.PREDICTOR]
    language_head = predictor(cfg)
    if cfg.MODEL.LANGUAGE_HEAD.FREEZE:
        print("[Info] Freezing language_head Layers.")
        for p in language_head.parameters():
            p.requires_grad = False
    return language_head
