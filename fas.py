import os
import sys
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import third_party
import logging

sys.path.append('../../')
import timm

from prompt_templates import spoof_templates, real_templates,class_templates
from clip.convpass import set_Convpass, set_Convpass_MoE
from collections import OrderedDict
import math

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from clip.qformer import CrossAttnBlock
from clip.model import LayerNorm
_tokenizer = _Tokenizer()

class moga_eta(nn.Module):

    def __init__(self):
        super(moga_eta, self).__init__()
        self.model, _ = clip.load("ViT-B/16", 'cuda:0')
        logging.info("------------self.model, _ = clip.load----ViT-B/16--------")
        clip_model = self.model

        self.get_image_feature = clip_model.visual
        self.dtype = clip_model.dtype
        self.device = 'cuda'
        self.image_encoder = self.model.visual
        self.temp = nn.Parameter(0.07 * torch.ones([]))  # 温度系统
        self.itm_head = nn.Linear(1024, 2, dtype=self.dtype)

    def forward(self, input, input_caption, norm_flag=True):
        if self.training:
            bs = input.size(0)
            bs_shape = input.shape
            prompts = []
            rank = 0
            prompts = input_caption
            tokenized_prompts = torch.cat([clip.tokenize(p).to(self.device) for p in prompts])
            with torch.no_grad():
                self.tokenized_text = self.model.token_embedding(tokenized_prompts.to(self.device))
                self.tokenized_text.to(self.device)
                self.tokenized_text = self.tokenized_text.mean(dim=1)

            text_features = self.tokenized_text  #
            image_features, _ = self.image_encoder(input)  # 取的是 混合样式的特征

            ###============== Image-text Contrastive ===================###
            # image-text similarity: aggregate across all query tokens, [batch_size, batch_size, num_query_tokens]
            a1 = image_features
            a2 = text_features
            a1 = a1 / a1.norm(dim=-1, keepdim=True)
            a2 = a2.permute(1, 0)
            a2 = a2 / a2.norm(dim=-1, keepdim=True)
            sim_i2t = a1 @ a2 / self.temp
            # text-image similarity: aggregate across all query tokens, [batch_size, batch_size, num_query_tokens]
            b1 = text_features
            b2 = image_features
            b1 = b1 / b1.norm(dim=-1, keepdim=True)
            b2 = b2 / b2.norm(dim=-1, keepdim=True)
            b2 = b2.permute(1, 0)
            sim_t2i = b1 @ b2 / self.temp

            targets = torch.arange(start=0, end=bs).to(self.device)
            loss_itc = (F.cross_entropy(sim_i2t, targets, label_smoothing=0.1) +
                        F.cross_entropy(sim_t2i, targets, label_smoothing=0.1)) / 2

            ###============== Image-text Matching ===================###
            text_input_ids_world = text_features  # [32,512]
            with torch.no_grad():
                sim_t2i[:, rank * bs: rank * bs + bs].fill_diagonal_(-10000)
                sim_i2t[:, rank * bs: rank * bs + bs].fill_diagonal_(-10000)

            weights_t2i = F.softmax(sim_t2i, dim=1)
            weights_i2t = F.softmax(sim_i2t, dim=1)
            # select a negative image for each text
            image_query_neg = []
            for b in range(bs):
                neg_idx = torch.multinomial(weights_t2i[b], 1).item()
                image_query_neg.append(image_features[neg_idx])
            image_query_neg = torch.stack(image_query_neg, dim=0)
            image_query_all = torch.cat([image_features, image_query_neg, image_features], dim=0)  # pos, neg, pos

            text_ids_neg = []
            for b in range(bs):
                neg_idx = torch.multinomial(weights_i2t[b], 1).item()
                text_ids_neg.append(text_input_ids_world[neg_idx])
            text_ids_neg = torch.stack(text_ids_neg, dim=0)
            text_ids_all = torch.cat([text_features, text_features, text_ids_neg], dim=0)  # pos, pos, neg


            vl_embeddings = torch.cat((image_query_all, text_ids_all), dim=-1)
            vl_output = self.itm_head(vl_embeddings)
            logits = vl_output
            itm_labels = torch.cat([torch.ones(bs, dtype=torch.long),
                                    torch.zeros(2 * bs, dtype=torch.long)], dim=0, ).to(self.device)
            loss_itm = F.cross_entropy(logits, itm_labels)
            # style-wise
            Loss_STY = 0
            loss_qformer = 1.0 * loss_itm  # + 0.1 * loss_itc  #5-28 0.5-->0.3
            # similarity = 0  #不传送相似值
        else:
            loss_qformer = 0

        # single text prompt per class
        # logits_per_image, logits_per_text = self.model(input, self.text_inputs)

        # Ensemble of text features
        # tokenize the spoof and real templates
        spoof_texts = clip.tokenize(spoof_templates).cuda(non_blocking=True)  # tokenize
        real_texts = clip.tokenize(real_templates).cuda(non_blocking=True)  # tokenize
        # embed with text encoder
        spoof_class_embeddings = self.model.encode_text(spoof_texts)
        spoof_class_embeddings = spoof_class_embeddings.mean(dim=0)
        real_class_embeddings = self.model.encode_text(real_texts)
        real_class_embeddings = real_class_embeddings.mean(dim=0)

        # stack the embeddings for image-text similarity
        ensemble_weights = [spoof_class_embeddings, real_class_embeddings]
        text_features = torch.stack(ensemble_weights, dim=0).cuda()

        # get the image features
        #import pdb; pdb.set_trace()
        image_features,cls = self.get_image_feature(input)
        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.model.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()

        similarity = logits_per_image
        #return similarity, None
        return similarity,loss_qformer     # zlpiao 20250409 change ########

    def forward_vis(self, input, norm_flag=True):
        _, image_features_proj = self.model.visual.forward_full(input)
        feature = image_features_proj
        return None, feature
