# Copyright (c) 2020 Microsoft Corporation. Licensed under the MIT license. 
from __future__ import absolute_import, division, print_function
import argparse
import os
import glob
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import base64
import os.path as op
import random, json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm

from oscar.utils.tsv_file import TSVFile
from oscar.utils.logger import setup_logger
from oscar.utils.misc import mkdir, set_seed, weighted_sample
from oscar.modeling.modeling_vlbert_pretrain import BiImageBertForRetrieval
from transformers.pytorch_transformers import BertTokenizer, BertConfig, WEIGHTS_NAME 
from transformers.pytorch_transformers import AdamW, WarmupLinearSchedule, WarmupConstantSchedule


class MVPTRDataset(Dataset):
    """ Image/Text Retrieval Dataset"""
    def __init__(self, tokenizer_path, img_feat_file, input_data, id2phrase='datasets/mvp/id2phrase_new.json', max_seq_length=30, max_img_seq_length=50,
                max_tag_length=20, max_phrases=5):
        """
        tokenizer_path: path to load bert tokenizer to process caption text.

        """
        super(MVPTRDataset, self).__init__()
        self.img_file = img_feat_file
        self.img_tsv = TSVFile(self.img_file)
        self.data = json.load(open(input_data, 'r'))
        
        # load image features
        self.img_feats = None
        # get the image image_id to index map
        imgid2idx_file = op.join(op.dirname(self.img_file), 'imageid2idx.json')
        self.image_id2idx = json.load(open(imgid2idx_file, 'r'))  # img_id as string
    
        self.labels = {}
        for line_no in tqdm(range(self.img_tsv.num_rows())):
            row = self.img_tsv.seek(line_no)
            image_id = row[0]
            results = json.loads(row[1])
            objects = results['predictions'] if type(
                results) == dict else results
            self.labels[image_id] = {
                "class": objects,
            }

        # self.img2theme = json.load(open(args.img2theme, 'r'))
        self.id2sg = json.load(open(id2phrase, 'r'))
        self.sg2id = {tuple(v):int(k) for k,v in self.id2sg.items()}
        self.phrase_vocab_size = len(self.sg2id)
        # self.img2theme = {k:v for k,v in self.img2theme.items() if k.startswith(self.ds_name)}

        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        self.max_seq_length = max_seq_length
        self.max_img_seq_len = max_img_seq_length
        self.max_tag_length = max_tag_length
        self.max_phrases = max_phrases

    def get_label(self, index):
        return self.data[index]['label']

    def get_od_labels(self, img_key):
        if type(self.labels[img_key]) == str:
            od_labels = self.labels[img_key]
        else:
            od_labels = ' '.join(self.labels[img_key]['class'])
        return od_labels

    def tensorize_example(self, text_a, img_feat, text_b=None, phrase_nodes=None,
            cls_token_segment_id=0, pad_token_segment_id=0,
            sequence_a_segment_id=0, sequence_b_segment_id=1):
        tokens_a = self.tokenizer.tokenize(text_a)
        num_extra_tokens = 2
        num_phrases = self.max_phrases
        if len(tokens_a) > self.max_seq_length - num_extra_tokens: # edited here to make it for sequence length == 68
            tokens_a = tokens_a[:(self.max_seq_length - num_extra_tokens)]

        if len(phrase_nodes) >= num_phrases+self.max_seq_length-2-len(tokens_a):
            phrase_nodes = phrase_nodes[:(num_phrases+self.max_seq_length-2-len(tokens_a))]

        seq_tokens_a = [self.tokenizer.cls_token] + tokens_a #  + [self.tokenizer.sep_token]
        phrase_index = [len(seq_tokens_a), len(seq_tokens_a)+len(phrase_nodes)]
        input_ids_a = self.tokenizer.convert_tokens_to_ids(seq_tokens_a) + phrase_nodes + [self.tokenizer.vocab[self.tokenizer.sep_token]]
        segment_ids_a = [cls_token_segment_id] + [sequence_a_segment_id] * (len(tokens_a) + len(phrase_nodes) + 1)
        seq_a_len = len(input_ids_a)
        input_mask_a = [1] * len(input_ids_a)

        tokens_b = self.tokenizer.tokenize(text_b)
        if len(tokens_b) > self.max_tag_length - 2:
            # num_left_tokens = max(0, self.max_seq_len - len(tokens_b) - 2) # to avoid -1
            # assert(num_left_tokens >= 0)
            tokens_b = tokens_b[: (self.max_tag_length - 2)]
        seq_tokens_b = ['[CLS]'] + tokens_b + [self.tokenizer.sep_token]
        input_ids_b = self.tokenizer.convert_tokens_to_ids(seq_tokens_b)
        segment_ids_b = [sequence_b_segment_id] * len(seq_tokens_b)
        input_mask_b = [1] * len(input_ids_b)
        seq_b_len = len(input_ids_b)

        seq_len_a = len(input_ids_a)
        tmp_max_seq_len = self.max_seq_length + self.max_phrases
        seq_padding_len_a = tmp_max_seq_len - seq_len_a
        input_ids_a += seq_padding_len_a * [0,]
        input_mask_a += seq_padding_len_a * [0,]
        segment_ids_a += seq_padding_len_a * [pad_token_segment_id,]

        seq_padding_len_b = self.max_tag_length - seq_b_len
        input_ids_b += seq_padding_len_b * [0, ]
        input_mask_b += seq_padding_len_b * [0, ]
        segment_ids_b += seq_padding_len_b * [pad_token_segment_id, ]

        # image features
        img_len = img_feat.shape[0]
        if img_len > self.max_img_seq_len:
            img_feat = img_feat[0 : self.max_img_seq_len, :]
            img_len = img_feat.shape[0]
            img_padding_len = 0
            input_mask_b += [1]*self.max_img_seq_len
        else:
            img_padding_len = self.max_img_seq_len - img_len
            padding_matrix = torch.zeros((img_padding_len, img_feat.shape[1]))
            img_feat = torch.cat((img_feat, padding_matrix), 0)
            input_mask_b += [1]*img_len + [0]*img_padding_len
        image_start_index = len(input_ids_a) # input_ids_a here for the concated sequence
        image_end_index = image_start_index + img_len
        img_index = [image_start_index, image_end_index]

        input_ids_a = torch.tensor(input_ids_a, dtype=torch.long)
        input_mask_a = torch.tensor(input_mask_a, dtype=torch.long)
        segment_ids_a = torch.tensor(segment_ids_a, dtype=torch.long)
        input_ids_b = torch.tensor(input_ids_b, dtype=torch.long)
        input_mask_b = torch.tensor(input_mask_b, dtype=torch.long)
        segment_ids_b = torch.tensor(segment_ids_b, dtype=torch.long)
        phrase_index = torch.tensor(phrase_index, dtype=torch.long)
        image_index = torch.tensor(img_index, dtype=torch.long)
        return (input_ids_a, input_mask_a, segment_ids_a, input_ids_b,
            input_mask_b, segment_ids_b, img_feat)


    def __getitem__(self, index):
        item = self.data[index]
        img_key = item['image_id']
        feature = self.get_image(img_key)
        caption = item['text']
        phrases_node = self.get_caption_phrase(item['phrases'])
        # print(phrases_node)
        od_labels = self.get_od_labels(img_key)
        example = self.tensorize_example(caption, feature, text_b=od_labels, phrase_nodes=phrases_node)
        return index, tuple(example)
    
    def get_image(self, image_id):
        image_idx = self.image_id2idx[str(image_id)]
        row = self.img_tsv.seek(image_idx)
        results = json.loads(row[1])
        features = np.frombuffer(base64.b64decode(results['feature']),
                                dtype=np.float32).reshape((-1, 2054))
        if not features.flags['WRITEABLE']:
            features = np.copy(features)
        t_features = torch.from_numpy(features)
        return t_features

    def get_caption_phrase(self, phrases):
        phrase_nodes = [tuple(t) for t in phrases]
        phrase_nodes = [self.sg2id[t] for t in phrase_nodes if t in self.sg2id]
        # if len(phrase_nodes) > self.args.max_phrases:
        #     phrase_nodes = phrase_nodes[:self.args.max_phrases]
        return phrase_nodes

    def __len__(self):
        return len(self.data)
