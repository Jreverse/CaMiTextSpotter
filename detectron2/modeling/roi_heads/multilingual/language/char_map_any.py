# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from .char_map import CharMap


class AnyCharMap(CharMap):
    MAX_CHAR_NUM = 8000

    @classmethod
    def contain_char(cls, char):
        return True
