import torch
from torch import nn, Tensor
from .FocalTransformer import FocalTransformerBlock
from .transformer import PositionalEncoding
from .roi_seq_predictor_base import SequencePredictor
from torch.nn import functional as F

from detectron2.modeling.roi_heads.multilingual.language.languages import LANGUAGE_COMBO, get_language_config
from detectron2.modeling.roi_heads.multilingual.predictor.language_predictor_builder import make_language_predictor

import random
import numpy as np

class DynamicConv_v2(nn.Module):
    
    def __init__(self, cfg):
        super().__init__()

        self.hidden_dim = cfg.MODEL.SWINTS.HIDDEN_DIM
        self.dim_dynamic = cfg.MODEL.SWINTS.DIM_DYNAMIC
        self.num_dynamic = cfg.MODEL.SWINTS.NUM_DYNAMIC
        self.num_params = self.hidden_dim * self.dim_dynamic
        self.dynamic_layer = nn.Linear(self.hidden_dim, self.num_dynamic * self.num_params)

        self.norm1 = nn.LayerNorm(self.dim_dynamic)
        self.norm2 = nn.LayerNorm(self.hidden_dim)

        self.activation = nn.ELU(inplace=True)

    def forward(self, pro_features, roi_features):
        '''
        pro_features: (1,  N * nr_boxes, self.d_model)
        roi_features: (rec_resolution, N * nr_boxes, self.d_model)
        '''
        features = roi_features.permute(1, 0, 2)
        parameters = self.dynamic_layer(pro_features).permute(1, 0, 2)

        param1 = parameters[:, :, :self.num_params].view(-1, self.hidden_dim, self.dim_dynamic)
        param2 = parameters[:, :, self.num_params:].view(-1, self.dim_dynamic, self.hidden_dim)
        del parameters

        features = torch.bmm(features, param1)
      
        del param1
        features = self.norm1(features)
        features = self.activation(features)

        features = torch.bmm(features, param2)

        del param2

        features = self.norm2(features)
        features = self.activation(features)

        return features

class REC_STAGE(nn.Module):

    def __init__(self, cfg, d_model, num_classes, dim_feedforward=2048, nhead=8, dropout=0.2, activation="relu"):
        super().__init__()
        self.cfg = cfg
        self.d_model = d_model

        # dynamic.
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.inst_interact = DynamicConv_v2(cfg)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.ELU(inplace=True)

        self.feat_size = cfg.MODEL.REC_HEAD.POOLER_RESOLUTION
        self.rec_batch_size = cfg.MODEL.REC_HEAD.BATCH_SIZE
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=3)

        self.TLSAM =  nn.Sequential(
            FocalTransformerBlock(dim=256, input_resolution=self.feat_size, num_heads=8, window_size=7, expand_size=0, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.2,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, pool_method="fc", 
                 focal_level=2, focal_window=3, use_layerscale=False, layerscale_value=1e-4),
                FocalTransformerBlock(dim=256, input_resolution=self.feat_size, num_heads=8, window_size=7, expand_size=0, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.2,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, pool_method="fc", 
                 focal_level=2, focal_window=3, use_layerscale=False, layerscale_value=1e-4),FocalTransformerBlock(dim=256, input_resolution=self.feat_size, num_heads=8, window_size=7, expand_size=0, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.2,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, pool_method="fc", 
                 focal_level=2, focal_window=3, use_layerscale=False, layerscale_value=1e-4)
                 )

        self.pos_encoder = PositionalEncoding(self.d_model, max_len=(self.feat_size[0]//4)*(self.feat_size[1]//4))
        num_channels = d_model
        in_channels = d_model
        mode = 'nearest'
        self.k_encoder = nn.Sequential(
            encoder_layer(num_channels, num_channels, s=(2, 2)),
            encoder_layer(num_channels, num_channels, s=(2, 2))
        )
        self.k_decoder_det = nn.Sequential(
            decoder_layer_worelu(num_channels, num_channels, scale_factor=2, mode=mode),
            decoder_layer_worelu(num_channels, num_channels, scale_factor=2, mode=mode),
            decoder_layer(num_channels, in_channels, size=(self.feat_size[0], self.feat_size[1]), mode=mode)
        )
        self.k_decoder_rec = nn.Sequential(
            decoder_layer(num_channels, num_channels, scale_factor=2, mode=mode),
            decoder_layer(num_channels, num_channels, scale_factor=2, mode=mode),
        )
        
        self.run_all_heads = cfg.TEST.RUN_ALL_HEADS
        
        self.only_generic_training = (len(cfg.SEQUENCE.LANGUAGES_ENABLED)==1 and 'any' in cfg.SEQUENCE.LANGUAGES_ENABLED)
        if not self.only_generic_training:
            self.language_predictor = make_language_predictor(cfg=cfg)
        
        # TODO: remove once multi ok
        if not cfg.SEQUENCE.MULTI:
            from .roi_seq_predictors import SequencePredictor
            self.seq_decoder = SequencePredictor(cfg, 
                                                 dim_in=d_model,
                                                 )
        else:
            for language in cfg.SEQUENCE.LANGUAGES_ENABLED:
                language_cfg = get_language_config(cfg, language)
                
                frozen = (cfg.SEQUENCE.LANGUAGES_UNFREEZED is not None) and (
                    language not in cfg.SEQUENCE.LANGUAGES_UNFREEZED
                )
                
                if cfg.SEQUENCE.DECODER_LOSS == "CTCLoss" or language_cfg.ARCH.startswith("ctc_"):
                    # self.seq_predictor = None
                    raise NotImplementedError
                else:
                    from .roi_seq_predictor_base import SequencePredictor
                    self.seq_predictor = SequencePredictor
                
                setattr(
                    self,
                    "seq_{}".format(language),
                    self.seq_predictor(
                        cfg=cfg,
                        dim_in=d_model,
                        language=language,
                        num_char=language_cfg.NUM_CHAR,
                        embed_size=language_cfg.EMBED_SIZE,
                        hidden_size=language_cfg.HIDDEN_SIZE,
                        frozen=frozen,
                    ),
                )
        
        self.lang_to_id = {}
        self.id_to_lang = {}
        id = 0
        for language in self.cfg.SEQUENCE.LANGUAGES_ENABLED:
            self.lang_to_id[language] = id

            # Example: if we only have u_la1 in SEQUENCE.LANGUAGES but not 'fr',
            # we can map 'fr' in the gt to the id of 'u_la1' directly
            if language in LANGUAGE_COMBO:
                for sub_language in LANGUAGE_COMBO[language]:
                    if sub_language not in self.cfg.SEQUENCE.LANGUAGES_ENABLED:
                        self.lang_to_id[sub_language] = id

            # reverse map
            self.id_to_lang[id] = language
            id += 1

        # Assign default language
        default_languages = ["en", "la", "u_la1", cfg.SEQUENCE.LANGUAGES[0]]
        for lang in default_languages:
            if lang in cfg.SEQUENCE.LANGUAGES:
                self.default_language = lang
                break
            
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
        
        self.rescale = nn.Upsample(size=(self.feat_size[0], self.feat_size[1]), mode="bilinear", align_corners=False)
        
        if cfg.MODEL.REC_HEAD.FREEZE:
            print("[Info] Freezing rec_stage Layers. W/O language_predictor and seq_xx")
            self._freeze()

    def _freeze(self,):
        for key, param in self.named_parameters():
            if 'language_predictor' in key or 'seq_' in key:
                # freeze language_predictor by cfg.MODEL.LANGUAGE_HEAD.FREEZE
                # freeze seq_xx by cfg.SEQUENCE...
                continue
            if 'k_encoder' in key:
                continue
            param.requires_grad=False
        
    def init_weights(self,):
        # need special handling of init_weights for BatchNorm
        for name, param in self.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                # Caffe2 implementation uses MSRAFill, which in fact
                # corresponds to kaiming_normal_ in PyTorch
                try:
                    nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")
                except ValueError as e:
                    weight_names = ["bn", "downsample.1", "transformer_encoder"]
                    if any(s in name for s in weight_names):
                        continue  # skip known BatchNorms in res_layer and transformer layers
                    else:
                        raise Exception(f"Exception for weight {name}: {e}")
    
    def RecoConversion(self, k_roi_features, roi_features, features, pro_features, gt_masks, N, nr_boxes, idx, device):
        n,c,h,w = roi_features.size()
        
        # adding 
        k_roi_features = k_roi_features.view(n, c, -1).permute(2, 0, 1)
        # self_att.
        pro_features = pro_features.view(N, nr_boxes, self.d_model).permute(1, 0, 2)
        pro_features2 = self.self_attn(pro_features, pro_features, value=pro_features)[0]
        pro_features = pro_features + self.dropout1(pro_features2)

        del pro_features2

        pro_features = self.norm1(pro_features)

        # inst_interact.
        if idx:
            pro_features = pro_features.permute(1, 0, 2)[idx]
            pro_features = pro_features.repeat(2,1)[:self.rec_batch_size]
        else:
            pro_features = pro_features.permute(1, 0, 2)
        pro_features = pro_features.reshape(1, -1, self.d_model)
        pro_features2 = self.inst_interact(pro_features, k_roi_features)
        pro_features = k_roi_features.permute(1,0,2) + self.dropout2(pro_features2)

        del pro_features2

        obj_features = self.norm2(pro_features)

        # obj_feature.
        obj_features2 = self.linear2(self.dropout(self.activation(self.linear1(obj_features))))
        obj_features = obj_features + self.dropout3(obj_features2)

        del obj_features2
        obj_features = self.norm3(obj_features)
        obj_features = obj_features.permute(1,0,2)
        obj_features = self.pos_encoder(obj_features)
        obj_features = self.transformer_encoder(obj_features)
        obj_features = obj_features.permute(1,2,0)
        n,c,w = obj_features.shape
        obj_features = obj_features.view(n,c,self.feat_size[0]//4,self.feat_size[1]//4)
        obj_features = obj_features
        k_roi_features = k_roi_features.permute(1,2,0)
        k_roi_features = k_roi_features.view(n,c,self.feat_size[0]//4,self.feat_size[1]//4)
        k_rec = k_roi_features * obj_features.sigmoid()
        k_rec = self.k_decoder_rec[0](k_rec)
        k_rec = k_rec + features[0]
        # import ipdb; ipdb.set_trace()
        k_det = obj_features
        k_det = self.k_decoder_det[0](k_det)
        k_det = k_det + features[0]
        k_rec = k_rec * k_det.sigmoid()

        k_rec = self.k_decoder_rec[1](k_rec) + roi_features
        k_det = self.k_decoder_det[1](k_det) + roi_features
        k_rec = k_rec * k_det.sigmoid()

        k_rec = self.k_decoder_det[-1](k_rec)
        k_rec = k_rec.flatten(-2,-1).permute(0,2,1)
        k_rec = self.TLSAM(k_rec)
        k_rec = k_rec.permute(0,2,1).view(n,c,self.feat_size[0],self.feat_size[1])
        k_rec = k_rec * gt_masks
        
        return k_rec

    def forward(self, roi_features, pro_features, gt_masks, N, nr_boxes, idx=None, targets=None, targets_text=None, targets_lang=None):
        """
        :param bboxes: (N, nr_boxes, 4)
        :param pro_features: (N, nr_boxes, d_model)
        """
        device = roi_features.device
        gt_masks = self.rescale(gt_masks.unsqueeze(1))
        
        if not self.only_generic_training:
            language_logits = self.language_predictor(roi_features*gt_masks)
        
        features = []
        k_roi_features = roi_features
        for i in range(0, len(self.k_encoder)):
            k_roi_features = self.k_encoder[i](k_roi_features)
            features.append(k_roi_features)

        
        k_rec = self.RecoConversion(k_roi_features, roi_features, features, pro_features, gt_masks, N, nr_boxes, idx, device)
        
        # import ipdb;ipdb.set_trace()
        
        if self.training:
            if self.cfg.SEQUENCE.MULTI:
                loss_seq_dict = {}
                
                num_words = roi_features.shape[0]

                keep_indices = []

                if targets_lang is not None:
                    assert (
                        len(targets_lang) == num_words
                    ), f"Dimension mismatch: {len(targets_lang)} != {num_words}"
                    for word_id in range(0, num_words):
                        gt_language = str(targets_lang[word_id])
                        if gt_language != "none":
                            keep_indices.append(word_id)
                # 处理 None 样本
                if len(keep_indices) == 0:
                    if num_words > 0:
                        # try to bypass all none samples using zero grad
                        for language, _ in self.lang_to_id.items():
                            loss_seq_dict["loss_seq_{}".format(language)] = torch.tensor(0.0).to(device=k_rec.device)
                        if not self.cfg.MODEL.REC_HEAD.E2ETrain:
                            loss_seq_dict["loss_script_cls"] = torch.tensor(0.0).to(device=k_rec.device)
                        print(
                            (
                                "[Warning] No valid word/language in this batch,"
                                "[Warning] Bypass training on this iteration using zero grad."
                            )
                        )
                        return loss_seq_dict
                        # keep_indices.append(0)
                num_filtered = num_words - len(keep_indices)
                
                if num_filtered > 0:
                    if not self.only_generic_training:
                        language_logits = language_logits[keep_indices]
                    targets_lang = targets_lang[keep_indices]

                    if random.random() < 0.001:
                        # log every 1000
                        print(
                            "[Info] Filtered {} out of {} targets using none-language criteria".format(
                                num_filtered, num_words
                            )
                        )

                    kept_x = k_rec[keep_indices]
                    num_words = kept_x.shape[0]
                    # need to handle the case when num_words == 0 below
                else:
                    kept_x = k_rec
                
                
                # prepare lang_id for training language_predictor branch
                
                if not self.only_generic_training:
                    gt_lang_id = torch.zeros(num_words).long()
                    best_coverage = torch.zeros(num_words)
                    
                    for word_id in range(0, num_words):
                        if targets_lang is not None:
                            gt_language = targets_lang[word_id]
                            
                            if gt_language in self.lang_to_id:
                                # the gt language is supported by current model
                                gt_lang_id[word_id] = self.lang_to_id[gt_language]
                                continue

                        # either there's no gt language, or the gt language is not supported,
                        # we use the following heuristics to figure out
                        # the best potential gt language
                        
                        
                        # TODO: support 'any' or not in targets_lang language
                        max_count = 0
                        best_language = self.default_language
                        for language, lang_id in self.lang_to_id.items():
                            coverage = torch.sum((targets_text[language] > 0).int(), dim=1)
                            if coverage[word_id] > best_coverage[word_id]:
                                best_coverage[word_id] = coverage[word_id]
                                gt_lang_id[word_id] = lang_id
                                best_language = language
                                max_count = 1
                            elif coverage[word_id] == best_coverage[word_id]:
                                if best_language == self.default_language:
                                    # Prefer default language when equal
                                    pass
                                elif language == self.default_language:
                                    # Prefer default language when equal
                                    gt_lang_id[word_id] = lang_id
                                    best_language = language
                                else:
                                    # Classical "Uniformly Pick Max Integer"
                                    max_count += 1
                                    if random.random() < 1.0 / max_count:
                                        gt_lang_id[word_id] = lang_id
                                        best_language = language
                    
                # import ipdb;ipdb.set_trace()
                if self.cfg.MODEL.REC_HEAD.E2ETrain:
                    # Bypass language cls step.
                    # Using language_predictor to select one rec branch or cls gt
                    # training argmax rec branch only.
                    
                    if self.cfg.MODEL.REC_HEAD.GuideByGT:
                        language_id = gt_lang_id
                    else:
                        language_id = torch.argmax(language_logits, dim=1)
                    # tt = language_id.unsqueeze(1)
                    # lang_guide_embed = torch.zeros_like(language_logits).scatter_(1,tt,1)
                    
                    # selected rec branch to train
                    for language, target_id in self.lang_to_id.items():
                        selected_idx = torch.ones_like(language_id) * (language_id==target_id)
                        selected_idx = torch.nonzero(selected_idx).flatten().tolist()
                        
                        kept_word_target = (
                            targets_text[language][keep_indices][selected_idx]
                            if num_filtered > 0
                            else targets_text[language][selected_idx]
                        )
                        kept_decoder_target = (
                            targets_text[language][keep_indices][selected_idx]
                            if num_filtered > 0
                            else targets_text[language][selected_idx]
                        )
                        
                        loss_seq_dict["loss_seq_{}".format(language)] = (
                            getattr(self, "seq_{}".format(language))(
                                x= kept_x[selected_idx],
                                decoder_targets=kept_decoder_target,
                                word_targets=kept_word_target,
                            )
                            if len(selected_idx)>0 and (language in self.cfg.SEQUENCE.LANGUAGES_UNFREEZED)
                            else torch.tensor(0.0).to(device=kept_x.device)
                        )
                    
                    return loss_seq_dict
                else:
                    # Using both cls, rec gt to train branches respectively.
                    
                    if not self.only_generic_training:
                        # script cls loss calc
                        if num_words > 0:
                            assert language_logits.size(0) == len(gt_lang_id), 'language_logits dim not match!'
                            loss_seq_dict[
                                "loss_script_cls"
                            ] = self.cfg.MODEL.LANGUAGE_HEAD.LOSS_WEIGHT * self.cross_entropy_loss(
                                language_logits, gt_lang_id.to(device=device)
                            )
                        else:
                            # Not working
                            zero_loss = torch.tensor(0.0).to(device=device)
                            loss_seq_dict["loss_script_cls"] = zero_loss.requires_grad_()
                    
                    # rec loss calc
                    for language in self.cfg.SEQUENCE.LANGUAGES_ENABLED:
                        # fix word targets -1 to be 0 for ctc loss
                        # NOTE: 0 is blank char for ctc, but use -1 will cause NaN
                        if self.cfg.SEQUENCE.DECODER_LOSS == "CTCLoss":
                            targets_text[language][targets_text[language] == -1] = 0

                        kept_word_target = (
                            targets_text[language][keep_indices]
                            if num_filtered > 0
                            else targets_text[language]
                        )
                        kept_decoder_target = (
                            targets_text[language][keep_indices]
                            if num_filtered > 0
                            else targets_text[language]
                        )
                        loss_seq_dict["loss_seq_{}".format(language)] = (
                            getattr(self, "seq_{}".format(language))(
                                x=kept_x,
                                decoder_targets=kept_decoder_target,
                                word_targets=kept_word_target,
                            )
                            if num_words > 0
                            else torch.tensor(0.0).to(device=kept_x.device)
                        )
                    
                    return loss_seq_dict
            
            else:
                # [N, C, 28, 28]
                attn_vecs = self.seq_decoder(k_rec, targets, targets)
                return {'pred_rec': attn_vecs}
        else:
            if not self.only_generic_training:
                language_probs = F.softmax(language_logits, dim=1)
            else:
                language_probs = torch.ones((roi_features.size(0), 1), device=device)
            
            if self.cfg.SEQUENCE.MULTI:
                
                if self.run_all_heads:
                    decoded_chars_list = []
                    decoded_scores_list = []
                    detailed_decoded_scores_list = []
                    # import ipdb;ipdb.set_trace()
                    for language in self.cfg.SEQUENCE.LANGUAGES_ENABLED:
                        decoded_chars, decoded_scores, detailed_decoded_scores = getattr(
                            self, "seq_{}".format(language)
                        )(k_rec, use_beam_search=self.cfg.SEQUENCE.BEAM_SEARCH)
                        
                        decoded_chars_list.append(decoded_chars)
                        decoded_scores_list.append(decoded_scores)
                        detailed_decoded_scores_list.append(detailed_decoded_scores)
                    
                    # decoded_scores_list 不等长, 被填充为-1
                    return {
                        "seq_outputs_list": decoded_chars_list,
                        "seq_scores_list": torch.tensor(decoded_scores_list),
                        "detailed_seq_scores_list": torch.tensor(detailed_decoded_scores_list),
                        "language_probs": language_probs.clone().detach()
                    }
                else:
                    pass
            else:
                attn_vecs = self.seq_decoder(k_rec, targets, targets)
                return {'pred_rec': torch.tensor(attn_vecs)}
        

def encoder_layer(in_c, out_c, k=3, s=2, p=1):
    return nn.Sequential(nn.Conv2d(in_c, out_c, k, s, p),
                         nn.BatchNorm2d(out_c),
                         nn.ReLU(True))

def decoder_layer(in_c, out_c, k=3, s=1, p=1, mode='nearest', scale_factor=None, size=None):
    align_corners = None if mode=='nearest' else True
    return nn.Sequential(nn.Upsample(size=size, scale_factor=scale_factor,
                                     mode=mode, align_corners=align_corners),
                         nn.Conv2d(in_c, out_c, k, s, p),
                         nn.BatchNorm2d(out_c),
                         nn.ReLU(True))

def decoder_layer_worelu(in_c, out_c, k=3, s=1, p=1, mode='nearest', scale_factor=None, size=None):
    align_corners = None if mode=='nearest' else True
    return nn.Sequential(nn.Upsample(size=size, scale_factor=scale_factor,
                                   mode=mode, align_corners=align_corners),
                         nn.Conv2d(in_c, in_c, k, s, p),
                         nn.BatchNorm2d(in_c),
                         nn.ReLU(True),
                         nn.Conv2d(in_c, out_c, k, s, p))