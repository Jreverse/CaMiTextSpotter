# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import torch.nn.functional as F
from torch import nn
import torch

from .language_predictor_base import BaseLanguagePredictor

class BidirectionalLSTM(nn.Module):
    def __init__(self, in_ch, hidden_ch, out_ch):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(in_ch, hidden_ch, bidirectional=True)
        if out_ch is not None:
            self.fc = nn.Linear(hidden_ch * 2, out_ch)
        else:
            self.fc = None

    def forward(self, input):
        # input size: [W; BS; in_ch]
        output, _ = self.rnn(input)
        # output size: [W; BS; hidden_ch * 2] (bi-bidirectional)
        if self.fc is not None:
            w, bs, hc = output.size()
            # view in size: [W * BS; hidden_ch * 2]
            output_view = output.view(w * bs, hc)
            # output size: [W * BS; out_ch]
            output = self.fc(output_view)
            # separate width and batch size: [W; BS; out_ch]
            output = output.view(w, bs, -1)
        return output

class V1LanguagePredictor(BaseLanguagePredictor):
    def __init__(self, cfg, do_init_weights=True):
        super(V1LanguagePredictor, self).__init__(cfg=cfg, do_init_weights=False)
        input_c = cfg.MODEL.LANGUAGE_HEAD.INPUT_C  # default: 256
        input_h = cfg.MODEL.LANGUAGE_HEAD.INPUT_H  # default: 40
        input_w = cfg.MODEL.LANGUAGE_HEAD.INPUT_W  # default: 40
        conv1_c = cfg.MODEL.LANGUAGE_HEAD.CONV1_C  # default: 64
        conv2_c = cfg.MODEL.LANGUAGE_HEAD.CONV2_C  # default: 32

        assert input_h % 8 == 0
        assert input_w % 8 == 0

        fc1_in = (input_h // 8) * (input_w // 8) * conv2_c
        
        self.input_size = (input_h, input_w)
        self.conv11 = nn.Conv2d(input_c, conv1_c, 2, 2, 0)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv22 = nn.Conv2d(conv1_c, conv2_c, 2, 2, 0)
        self.fc11 = nn.Linear(fc1_in, 64)
        self.fc22 = nn.Linear(64, self.num_classes)
        
        if do_init_weights:
            self.init_weights()

    def forward(self, x):
        x = F.interpolate(x, size=self.input_size, mode="bilinear", align_corners=False)
        # [n, 256, 32, 32] => [n, 64, 16, 16] (input=32x32)
        # [n, 256, 48, 48] => [n, 64, 24, 24] (input=48x48)
        x = F.relu(self.conv11(x))
        # [n, 64, 16, 16] => [n, 64, 8, 8] (input=32x32)
        # [n, 64, 24, 24] => [n, 64, 12, 12] (input=48x48)
        x = self.maxpool(x)
        # [n, 64, 8, 8] => [n, 32, 4, 4] (input=32x32)
        # [n, 64, 12, 12] => [n, 32, 6, 6] (input=48x48)
        x = F.relu(self.conv22(x))
        # [n, 32, 4, 4] => [n, 512] (input=32x32)
        # [n, 32, 6, 6] => [n, 1152] (input=48x48)
        x = x.view(x.size(0), -1)
        # [n, 512] => [n, 64] (input=32x32)
        # [n, 1152] => [n, 64] (input=32x32)
        x = F.relu(self.fc11(x))
        # [n, 64] => [n, num_class]
        x = self.fc22(x)

        return x
    

class V2LanguagePredictor(BaseLanguagePredictor):
    def __init__(self, cfg, do_init_weights=True):
        # Compared to V3, the main change is the support for dynamic input size.
        super(V2LanguagePredictor, self).__init__(cfg=cfg, do_init_weights=False)
        input_c = cfg.MODEL.LANGUAGE_HEAD.INPUT_C  # default: 512
        input_h = cfg.MODEL.LANGUAGE_HEAD.INPUT_H  # default: 3
        self.input_w = cfg.MODEL.LANGUAGE_HEAD.INPUT_W  # default: 40
        conv1_c = cfg.MODEL.LANGUAGE_HEAD.CONV1_C  # default: 128
        conv2_c = cfg.MODEL.LANGUAGE_HEAD.CONV2_C  # default: 64

        assert input_h % 3 == 0
        assert self.input_w % 4 == 0

        fc1_in = (input_h // 3) * (self.input_w // 4) * conv2_c

        self.conv1 = nn.Conv2d(input_c, conv1_c, kernel_size=(3, 2), stride=(3, 2), padding=0)
        self.conv2 = nn.Conv2d(conv1_c, conv2_c, kernel_size=(1, 2), stride=(1, 2), padding=0)
        self.fc1 = nn.Linear(fc1_in, 64)
        self.fc2 = nn.Linear(64, self.num_classes)

        if do_init_weights:
            self.init_weights()

    def forward(self, x):
        assert (
            x.shape[3] <= self.input_w
        ), f"Input patch length {x.shape[3]} > max supported length {self.input_w}"

        # [n, 512, 3, <=40] => [n, 512, 3, 40] with 0-padding
        if x.shape[3] < self.input_w:
            x = torch.cat(
                (
                    x,
                    torch.zeros(
                        x.shape[0],
                        x.shape[1],
                        x.shape[2],
                        self.input_w - x.shape[3],
                        device=x.device,
                    ),
                ),
                dim=3,
            )

        # [n, 512, 3, 40] => [n, 128, 1, 20]
        x = F.relu(self.conv1(x))
        # [n, 128, 1, 20] => [n, 64, 1, 10]
        x = F.relu(self.conv2(x))

        # [n, 64, 1, 10] => [n, 640]
        x = x.view(x.size(0), -1)
        # [n, 640] => [n, 64]
        x = F.relu(self.fc1(x))
        # [n, 64] => [n, num_class]

        x = self.fc2(x)

        return x

class V3LanguagePredictor(V2LanguagePredictor):
    def __init__(self, cfg, do_init_weights=True):
        super(V3LanguagePredictor, self).__init__(cfg=cfg, do_init_weights=False)

        self.bilstm_hidden_size = 192
        self.bilstm_output_size = 192
        self.lstm0_c = 256
        self.pre_lstm_kernel_height = 3

        self.pre_lstm_conv = nn.Conv2d(
            512,
            self.lstm0_c,
            kernel_size=(self.pre_lstm_kernel_height, 1),
            stride=1,
        )

        # note: original ctc-based rec-head uses self.num_classes + 1 to support dummy label
        self.lstm = nn.Sequential(
            BidirectionalLSTM(self.lstm0_c, self.bilstm_hidden_size, self.bilstm_output_size),
            BidirectionalLSTM(self.bilstm_output_size, self.bilstm_hidden_size, self.num_classes),
        )

        # self.ctc_reduction = "sum_manual"  # "sum"
        # reduction = self.ctc_reduction
        # if "manual" in self.ctc_reduction:
        #     reduction = "none"
        # self.criterion_seq_decoder = nn.CTCLoss(reduction=reduction, zero_infinity=True)

        if do_init_weights:
            self.init_weights()

    def forward(self, x):
        # n x 512 x 3 x <=40
        x = self.pre_lstm_conv(x)

        # shape before squeeze: n x ch x 1 x w(<=40)]
        x = torch.squeeze(x, 2)
        # shape after squeeze: n x ch x w

        x = x.permute(2, 0, 1).contiguous()
        # shape after permute: w x n x ch
        preds = self.lstm(x)
        # output size is w x n x cl

        # w x n x cl => n x cl
        aggregated_preds = torch.mean(F.relu(preds), dim=0)

        return aggregated_preds
    
