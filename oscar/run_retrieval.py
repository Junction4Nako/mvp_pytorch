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
from oscar.modeling.modeling_vlbert import BiImageBertForRetrieval
from transformers.pytorch_transformers import BertTokenizer, BertConfig, WEIGHTS_NAME 
from transformers.pytorch_transformers import AdamW, WarmupLinearSchedule, WarmupConstantSchedule


class RetrievalDataset(Dataset):
    """ Image/Text Retrieval Dataset"""
    def __init__(self, tokenizer, args, split='train', is_train=True, coarse_cap_index=None, coarse_img_index=None):
        """
        tokenizer: tokenizer to process caption text.
        args: configureation parameters including max_seq_length, etc.
        split: used to infer the data used for training or testing. 
             All files are in .pt format of a dictionary with image keys and 
             image features (pytorch tensors), captions (list of str, support multiple
             captions per image), labels (list of dictionary or str of all labels),

        """
        super(RetrievalDataset, self).__init__()
        self.img_file = args.img_feat_file
        caption_file = op.join(args.data_dir, '{}_captions.pt'.format(split))
        self.img_tsv = TSVFile(self.img_file)
        self.captions = torch.load(caption_file)
        self.img_keys = list(self.captions.keys())  # img_id as int
        if not type(self.captions[self.img_keys[0]]) == list:
            self.captions = {k: json.loads(self.captions[k]) for k in self.img_keys}

        self.num_of_total_captions = args.num_captions_per_img_train*len(self.img_keys)
        print('number of total captions:',self.num_of_total_captions)
        print('number of images', len(self.captions))

        # get the image image_id to index map
        # imgid2idx_file = op.join(op.dirname(self.img_file), 'imageid2idx.json')
        # self.image_id2idx = json.load(open(imgid2idx_file))  # img_id as string
        
        # get the image features and labels
        if args.dataset_name == 'flickr':
            img_feat_file = op.join(args.data_dir, '{}_img_frcnn_feats.pt'.format(split))
            self.img_feats = torch.load(img_feat_file)
            if args.add_od_labels:
                labels_file = op.join(args.data_dir, '{}_{}_labels.pt'.format(split, args.od_label_type))
                self.labels = torch.load(labels_file)
        else:
            self.img_feats = None
            # get the image image_id to index map
            imgid2idx_file = op.join(op.dirname(self.img_file), 'imageid2idx.json')
            self.image_id2idx = json.load(open(imgid2idx_file))  # img_id as string
        
            if args.add_od_labels:
                label_data_dir = op.dirname(self.img_file)
                label_file = os.path.join(label_data_dir, "predictions.tsv")
                self.label_tsv = TSVFile(label_file)
                self.labels = {}
                for line_no in tqdm(range(self.label_tsv.num_rows())):
                    row = self.label_tsv.seek(line_no)
                    image_id = row[0]
                    if int(image_id) in self.img_keys:
                        results = json.loads(row[1])
                        objects = results['objects'] if type(
                            results) == dict else results
                        self.labels[int(image_id)] = {
                            "image_h": results["image_h"] if type(
                                results) == dict else 600,
                            "image_w": results["image_w"] if type(
                                results) == dict else 800,
                            "class": [cur_d['class'] for cur_d in objects],
                            "boxes": np.array([cur_d['rect'] for cur_d in objects],
                                            dtype=np.float32)
                        }
                self.label_tsv._fp.close()
                self.label_tsv._fp = None

        # self.img2theme = json.load(open(args.img2theme, 'r'))
        self.sent_sgs = torch.load(args.sent_sg_json)
        self.id2sg = json.load(open(args.id2node, 'r'))
        self.sg2id = {tuple(v):int(k) for k,v in self.id2sg.items()}
        self.phrase_vocab_size = len(self.sg2id)
        self.ds_name = args.dataset_name
        # self.img2theme = {k:v for k,v in self.img2theme.items() if k.startswith(self.ds_name)}


        # get extra concepts
        if args.extra_concept:
            add_concept_file = op.join(args.data_dir, '{}_extra_concepts.pt'.format(split))
            self.extra_concep = torch.load(add_concept_file)
        

        if args.clip_neg_sampling and is_train:
            neg_scpres_file = op.join(args.data_dir, '{}_clip_ft_scores.pt'.format(split))
            self.neg_scores = torch.load(neg_scpres_file)

        if is_train:
            self.num_captions_per_img = args.num_captions_per_img_train
        else:
            self.num_captions_per_img = args.num_captions_per_img_val
            self.num_images_per_cap = args.num_images_per_cap_val
            if args.eval_img_keys_file:
                # select a subset of image keys for evaluation. eg. COCO 1k and 5k
                # eval_img_keys_file is a list of image keys saved in tsv file
                with open(op.join(args.data_dir, args.eval_img_keys_file), 'r') as f:
                    img_keys = f.readlines()
                self.img_keys = [int(k.strip()) for k in img_keys]
                self.num_of_total_captions = args.num_captions_per_img_train*len(self.img_keys)
                self.captions = {k: self.captions[k] for k in self.img_keys}
                if args.add_od_labels:
                    self.labels = {k: self.labels[k] for k in self.img_keys}

            if args.eval_caption_index_file:
                # hard negative image/caption indexs for retrieval re-rank setting.
                # useful for mini val set to monitor the performance during training.
                # However, it cannot be used together with cross image evaluation.
                self.has_caption_indexs = True
                assert not args.cross_image_eval 
                caption_index_file = op.join(args.data_dir, args.eval_caption_index_file)
                self.caption_indexs = torch.load(caption_index_file)
                if not type(self.caption_indexs[self.img_keys[0]]) == list:
                    self.caption_indexs = {k: json.loads(self.caption_indexs[k]) for k in self.img_keys}
            else:
                self.has_caption_indexs = False

            if coarse_cap_index:
                self.has_caption_indexs = True
                self.caption_indexs = coarse_cap_index
            else:
                self.has_caption_indexs = False

            if coarse_img_index:
                self.has_image_indexs = True
                self.image_indexs = coarse_img_index
            else:
                self.has_image_indexs = False

        self.is_train = is_train
        self.output_mode = args.output_mode
        self.tokenizer = tokenizer
        self.max_seq_len = args.max_seq_length
        self.max_img_seq_len = args.max_img_seq_length
        self.args = args

    def set_caption_index(self, caption_index):
        self.num_captions_per_img = self.args.num_captions_per_img_val
        self.has_caption_indexs = True
        self.has_image_indexs = False
        self.caption_indexs = caption_index

    def set_image_index(self, image_index):
        self.num_images_per_cap = self.args.num_images_per_cap_val
        self.has_image_indexs = True
        self.has_caption_indexs = False
        self.image_indexs = image_index

    def unset_index(self):
        self.num_captions_per_img = self.args.num_captions_per_img_train
        self.num_images_per_cap = 1
        self.has_image_indexs = False
        self.has_caption_indexs = False
    
    def get_image_caption_index(self, index):
        # return img_idx to access features and [img_key, cap_idx] to access caption
        if not self.is_train and self.args.cross_image_eval:
            img_idx = index // (self.num_captions_per_img * len(self.img_keys))
            cap_idx = index % (self.num_captions_per_img * len(self.img_keys))
            img_idx1 = cap_idx // self.num_captions_per_img
            cap_idx1 = cap_idx % self.num_captions_per_img
            return img_idx, [self.img_keys[img_idx1], cap_idx1]
        if not self.is_train and self.has_caption_indexs:
            img_idx = index // self.num_captions_per_img
            cap_idx = index % self.num_captions_per_img
            img_key1, cap_idx1 = self.caption_indexs[self.img_keys[img_idx]][cap_idx]
            return img_idx, [img_key1, cap_idx1]
        if not self.is_train and self.has_image_indexs:
            cap_idx = index // self.num_images_per_cap
            cap_img_idx = cap_idx // self.args.num_captions_per_img_train
            cap_cap_idx = cap_idx % self.args.num_captions_per_img_train
            img_idx = index % self.num_images_per_cap
            img_key1 = self.image_indexs[(self.img_keys[cap_img_idx],cap_cap_idx)][img_idx]
            return img_key1, [self.img_keys[cap_img_idx], cap_cap_idx]
        img_idx = index // self.num_captions_per_img
        cap_idx = index % self.num_captions_per_img
        return img_idx, [self.img_keys[img_idx], cap_idx]

    def get_label(self, index):
        img_idx, cap_idx = self.get_image_caption_index(index)
        return 1 if self.img_keys[img_idx] == cap_idx[0] else 0

    def get_od_labels(self, img_key, cap_index=None):
        if self.args.add_od_labels:
            if self.ds_name != 'flickr':
                if type(self.labels[img_key]) == str:
                    od_labels = self.labels[img_key]
                else:
                    od_labels = ' '.join(self.labels[img_key]['class'])
                return od_labels
            else:
                if type(self.labels[img_key]) == str:
                    od_labels = self.labels[img_key]
                else:
                    od_labels = ' '.join(list(set(self.labels[img_key]['class'])))
                    # od_labels = ' '.join(self.labels[img_key]['class'])
                
                if cap_index is not None:
                    extra_concepts = self.extra_concep[str(img_key)][cap_index]
                    if self.args.num_extra_concept < len(extra_concepts):
                        extra_concepts = random.sample(extra_concepts, self.args.num_extra_concept)
                    od_labels += ' '.join(od_labels)
                return od_labels

    def tensorize_example(self, text_a, img_feat, text_b=None, phrase_nodes=None,
            cls_token_segment_id=0, pad_token_segment_id=0,
            sequence_a_segment_id=0, sequence_b_segment_id=1):
        tokens_a = self.tokenizer.tokenize(text_a)
        num_extra_tokens = 2
        num_phrases = self.args.max_phrases
        if len(tokens_a) > self.args.max_seq_length - num_extra_tokens: # edited here to make it for sequence length == 68
            tokens_a = tokens_a[:(self.args.max_seq_length - num_extra_tokens)]

        if len(phrase_nodes) >= num_phrases+self.args.max_seq_length-2-len(tokens_a):
            phrase_nodes = phrase_nodes[:(num_phrases+self.args.max_seq_length-2-len(tokens_a))]

        seq_tokens_a = [self.tokenizer.cls_token] + tokens_a #  + [self.tokenizer.sep_token]
        phrase_index = [len(seq_tokens_a), len(seq_tokens_a)+len(phrase_nodes)]
        input_ids_a = self.tokenizer.convert_tokens_to_ids(seq_tokens_a) + phrase_nodes + [self.tokenizer.vocab[self.tokenizer.sep_token]]
        segment_ids_a = [cls_token_segment_id] + [sequence_a_segment_id] * (len(tokens_a) + len(phrase_nodes) + 1)
        seq_a_len = len(input_ids_a)
        input_mask_a = [1] * len(input_ids_a)

        tokens_b = self.tokenizer.tokenize(text_b)
        if len(tokens_b) > self.args.max_tag_length - 2:
            # num_left_tokens = max(0, self.max_seq_len - len(tokens_b) - 2) # to avoid -1
            # assert(num_left_tokens >= 0)
            tokens_b = tokens_b[: (self.args.max_tag_length - 2)]
        seq_tokens_b = ['[CLS]'] + tokens_b + [self.tokenizer.sep_token]
        input_ids_b = self.tokenizer.convert_tokens_to_ids(seq_tokens_b)
        segment_ids_b = [sequence_b_segment_id] * len(seq_tokens_b)
        input_mask_b = [1] * len(input_ids_b)
        seq_b_len = len(input_ids_b)

        seq_len_a = len(input_ids_a)
        tmp_max_seq_len = self.max_seq_len + self.args.max_phrases
        seq_padding_len_a = tmp_max_seq_len - seq_len_a
        input_ids_a += seq_padding_len_a * [0,]
        input_mask_a += seq_padding_len_a * [0,]
        segment_ids_a += seq_padding_len_a * [pad_token_segment_id,]

        seq_padding_len_b = self.args.max_tag_length - seq_b_len
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
        if self.is_train:
            return (input_ids_a, input_mask_a, segment_ids_a, input_ids_b,
            input_mask_b, segment_ids_b, img_feat)
        else:
            return (input_ids_a, input_mask_a, segment_ids_a, input_ids_b,
            input_mask_b, segment_ids_b, img_feat)


    def get_neg_txt(self, img_idx):
        img_scores = self.neg_scores['img2htxt_logit'][img_idx, :]
        sample_idx = weighted_sample(img_scores)
        neg_txt = self.neg_scores['img2htxt_index'][img_idx, sample_idx]
        img_idx_neg = neg_txt // self.num_captions_per_img
        cap_idx_neg = neg_txt % self.num_captions_per_img
        caption_neg = self.captions[self.img_keys[img_idx_neg]][cap_idx_neg]
        phrases_neg = self.get_caption_phrase(self.img_keys[img_idx_neg], cap_idx_neg)
        return caption_neg, phrases_neg


    def get_neg_img(self, img_idx, cap_idx):
        cap_scores = self.neg_scores['txt2himg_logit'][img_idx*5+cap_idx, :]
        sample_idx = weighted_sample(cap_scores)
        neg_img = self.neg_scores['txt2himg_index'][img_idx*5+cap_idx, sample_idx]
        feature_neg, v_c_neg = self.get_image(self.img_keys[neg_img])
        od_labels_neg = self.get_od_labels(self.img_keys[neg_img])
        return feature_neg, od_labels_neg, v_c_neg


    def __getitem__(self, index):
        img_idx, cap_idxs = self.get_image_caption_index(index)
        img_key = self.img_keys[img_idx]
        feature = self.get_image(img_key)
        caption = self.captions[cap_idxs[0]][cap_idxs[1]]
        phrases_node = self.get_caption_phrase(cap_idxs[0], cap_idxs[1])
        # print(phrases_node)
        od_labels = self.get_od_labels(img_key)
        example = self.tensorize_example(caption, feature, text_b=od_labels, phrase_nodes=phrases_node)
        label = 1 if img_key == cap_idxs[0] else 0
        # print([i.shape for i in example])
        return index, tuple(list(example)+[label])
        if self.is_train:
            img_idx, cap_idxs = self.get_image_caption_index(index)
            img_key = self.img_keys[img_idx]
            feature, v_c = self.get_image(img_key)
            caption = self.captions[cap_idxs[0]][cap_idxs[1]]
            phrases = self.get_caption_phrase(cap_idxs[0], cap_idxs[1])
            if self.args.extra_concept:
                od_labels = self.get_od_labels(img_key, cap_idxs[1])
            else:
                od_labels = self.get_od_labels(img_key)
            example = self.tensorize_example(caption, feature, text_b=od_labels, visual_theme=v_c, phrase_nodes=phrases)

            # select a negative pair
            if self.args.clip_neg_sampling and random.random() <= self.args.clip_neg_prob:
                if random.random() <= 0.5:
                    caption_neg, phrases_neg = self.get_neg_txt(img_idx)
                    example_neg = self.tensorize_example(caption_neg, feature, text_b=od_labels, visual_theme=v_c, phrase_nodes=phrases_neg)
                else:
                    feature_neg, od_labels_neg, v_c_neg = self.get_neg_img(img_idx, cap_idxs[1])
                    example_neg = self.tensorize_example(caption, feature_neg, text_b=od_labels_neg, visual_theme=v_c_neg, phrase_nodes=phrases)
            else:
                neg_img_indexs = list(range(0, img_idx)) + list(range(img_idx + 1, len(self.img_keys)))
                img_idx_neg = random.choice(neg_img_indexs)
                if random.random() <= 0.5:
                    # randomly select a negative caption from a different image.
                    cap_idx_neg = random.randint(0, self.num_captions_per_img - 1)
                    caption_neg = self.captions[self.img_keys[img_idx_neg]][cap_idx_neg]
                    phrases_neg = self.get_caption_phrase(self.img_keys[img_idx_neg], cap_idx_neg)
                    example_neg = self.tensorize_example(caption_neg, feature, text_b=od_labels, visual_theme=v_c, phrase_nodes=phrases_neg)
                else:
                    # randomly select a negative image 
                    feature_neg, v_c_neg = self.get_image(self.img_keys[img_idx_neg])
                    od_labels_neg = self.get_od_labels(self.img_keys[img_idx_neg])
                    example_neg = self.tensorize_example(caption, feature_neg, text_b=od_labels_neg, visual_theme=v_c_neg, phrase_nodes=phrases)

            example_pair = tuple(list(example) + [1] + list(example_neg) + [0])
            return index, example_pair
        else:
            img_idx, cap_idxs = self.get_image_caption_index(index)
            img_key = self.img_keys[img_idx]
            feature, v_c = self.get_image(img_key)
            caption = self.captions[cap_idxs[0]][cap_idxs[1]]
            phrases_node = self.get_caption_phrase(cap_idxs[0], cap_idxs[1])
            od_labels = self.get_od_labels(img_key)
            example = self.tensorize_example(caption, feature, text_b=od_labels, visual_theme=v_c, phrase_nodes=phrases_node)
            label = 1 if img_key == cap_idxs[0] else 0
            return index, tuple(list(example) + [label])

    def get_image(self, image_id):
        if self.ds_name != 'flickr':
            image_idx = self.image_id2idx[str(image_id)]
            row = self.img_tsv.seek(image_idx)
            num_boxes = int(row[1])
            features = np.frombuffer(base64.b64decode(row[-1]),
                                    dtype=np.float32).reshape((num_boxes, -1))
            if not features.flags['WRITEABLE']:
                features = np.copy(features)
            t_features = torch.from_numpy(features)
        else:
            t_features = self.img_feats[image_id]
        return t_features
        # theme_nodes = self.img2theme[self.ds_name+'_'+str(image_id)]
        # if len(theme_nodes) > self.args.max_visual_themes:
        #     theme_nodes = theme_nodes[:self.args.max_visual_themes]
        # theme_nodes = [t[0]+self.tokenizer.vocab_size+self.phrase_vocab_size for t in theme_nodes]
        # return t_features, theme_nodes

    def get_caption_phrase(self, image_id, cap_id):
        if self.ds_name == 'flickr':
            phrase_nodes = [tuple(t) for t in self.sent_sgs[image_id][cap_id]]
        else:
            phrase_nodes = [tuple(t.split('_')) for t in self.sent_sgs[image_id][cap_id]]
        phrase_nodes = [self.sg2id[t] for t in phrase_nodes if t in self.sg2id]
        # if len(phrase_nodes) > self.args.max_phrases:
        #     phrase_nodes = phrase_nodes[:self.args.max_phrases]
        return phrase_nodes

    def __len__(self):
        if not self.is_train and self.args.cross_image_eval:
            return len(self.img_keys) ** 2 * self.num_captions_per_img
        if not self.is_train and self.has_image_indexs:
            return self.num_images_per_cap * self.num_of_total_captions
        return len(self.img_keys) * self.num_captions_per_img


def compute_score_with_logits(logits, labels):
    if logits.shape[1] > 1:
        logits = torch.max(logits, 1)[1].data # argmax
        scores = logits == labels 
    else:
        scores = torch.zeros_like(labels).cuda()
        for i, (logit, label) in enumerate(zip(logits, labels)):
            logit_ = torch.sigmoid(logit)
            if (logit_ >= 0.5 and label == 1) or (logit_ < 0.5 and label == 0):
                scores[i] = 1
    return scores


def compute_ranks(dataset, results):
    labels = np.array([dataset.get_label(i) for i in range(len(dataset))])
    similarities = np.array([results[i] for i in range(len(dataset))])
    if dataset.has_caption_indexs:
        num_captions_per_img = dataset.num_captions_per_img
    else:
        num_captions_per_img = len(dataset.img_keys) * dataset.num_captions_per_img
    labels = np.reshape(labels, [-1, num_captions_per_img])
    similarities = np.reshape(similarities, [-1, num_captions_per_img])

    i2t_ranks, t2i_ranks = [], []
    for lab, sim in zip(labels, similarities):
        inds = np.argsort(sim)[::-1]
        rank = num_captions_per_img
        for r, ind in enumerate(inds):
            if lab[ind] == 1:
                rank = r
                break
        i2t_ranks.append(rank)
    if not dataset.has_caption_indexs:
        labels = np.swapaxes(labels, 0, 1)
        similarities = np.swapaxes(similarities, 0, 1)
        for lab, sim in zip(labels, similarities):
            inds = np.argsort(sim)[::-1]
            rank = num_captions_per_img
            for r, ind in enumerate(inds):
                if lab[ind] == 1:
                    rank = r
                    break
            t2i_ranks.append(rank)
    return i2t_ranks, t2i_ranks


def compute_ranks_t2i(dataset, results):
    labels = np.array([dataset.get_label(i) for i in range(len(dataset))])
    similarities = np.array([results[i] for i in range(len(dataset))])
    assert dataset.has_image_indexs
    num_images_per_cap = dataset.num_images_per_cap
    labels = np.reshape(labels, [-1, num_images_per_cap])
    similarities = np.reshape(similarities, [-1, num_images_per_cap])
    t2i_ranks = []
    for lab, sim in zip(labels, similarities):
        inds = np.argsort(sim)[::-1]
        rank = num_images_per_cap
        for r, ind in enumerate(inds):
            if lab[ind] == 1:
                rank = r
                break
        t2i_ranks.append(rank)
    return t2i_ranks


def compute_ranks_coarse(dataset, similarities):
    i2t_ranks, t2i_ranks = [], []
    i2t_index = {}
    t2i_index = {}
    # i2t
    for i in range(similarities.shape[0]):
        tmp_index = []
        inds = np.argsort(similarities[i,:])[::-1]
        rank = similarities.shape[1]
        for r, ind in enumerate(inds):
            if ind >= i*dataset.args.num_captions_per_img_train and ind < (i+1)*dataset.args.num_captions_per_img_train:
                rank = r
                break
        i2t_ranks.append(rank)
        for r, ind in enumerate(inds):
            if r >= dataset.args.num_captions_per_img_val:
                break
            cap_img_index = ind // dataset.args.num_captions_per_img_train
            cap_cap_index = ind % dataset.args.num_captions_per_img_train
            tmp_index.append((dataset.img_keys[cap_img_index], cap_cap_index))

        i2t_index[dataset.img_keys[i]] = tmp_index

    # t2i
    for i in range(similarities.shape[1]):
        tmp_index = []
        inds = np.argsort(similarities[:,i])[::-1]
        rank = similarities.shape[0]
        cap_img_index = i // dataset.args.num_captions_per_img_train
        cap_cap_index = i % dataset.args.num_captions_per_img_train
        for r, ind in enumerate(inds):
            if ind == i//dataset.args.num_captions_per_img_train:
                rank = r
                break
        t2i_ranks.append(rank)
        for r, ind in enumerate(inds):
            if r >= dataset.args.num_images_per_cap_val:
                break
            tmp_index.append(ind)

        t2i_index[(dataset.img_keys[cap_img_index], cap_cap_index)] = tmp_index
    return i2t_ranks, t2i_ranks, i2t_index, t2i_index


def save_checkpoint(model, tokenizer, args, epoch, global_step):
    checkpoint_dir = op.join(args.output_dir, 'checkpoint-{}-{}'.format(
        epoch, global_step))
    mkdir(checkpoint_dir)
    model_to_save = model.module if hasattr(model, 'module') else model
    save_num = 0
    while (save_num < 10):
        try:
            model_to_save.save_pretrained(checkpoint_dir)
            torch.save(args, op.join(checkpoint_dir, 'training_args.bin'))
            tokenizer.save_pretrained(checkpoint_dir)
            logger.info("Save checkpoint to {}".format(checkpoint_dir))
            break
        except:
            save_num += 1
    if save_num == 10:
        logger.info("Failed to save checkpoint after 10 trails.")
    return


def train(args, train_dataset, val_dataset, model, tokenizer):
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) 
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, 
            batch_size=args.train_batch_size, num_workers=args.num_workers)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // \
                args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps \
                * args.num_train_epochs

    # Prepare optimizer and scheduler
    no_decay = ['bias', 'LayerNorm.weight']
    grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not \
            any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if \
            any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    if args.scheduler == "constant":
        scheduler = WarmupConstantSchedule(
                optimizer, warmup_steps=args.warmup_steps)
    elif args.scheduler == "linear":
        scheduler = WarmupLinearSchedule(
                optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    else:
        raise ValueError("Unknown scheduler type: {}".format(args.scheduler))

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step, global_loss, global_acc =0,  0.0, 0.0
    global_r_loss, global_f_loss = 0.0, 0.0
    global_w_loss = 0.0
    model.zero_grad()
    log_json = []
    best_score = 0
    for epoch in range(int(args.num_train_epochs)):
        for step, (_, batch) in enumerate(train_dataloader):
            model.train()
            if hasattr(model, 'module'):
                model.module.forward_mod = 'train'
            else:
                model.forward_mod = 'train'
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {
                'input_ids_a':      batch[0],
                'attention_mask_a': batch[1],
                'token_type_ids_a': batch[2],
                'input_ids_b':      batch[3],
                'attention_mask_b': batch[4],
                'token_type_ids_b': batch[5],
                'img_feats':        batch[6],
                'max_tag_length': args.max_tag_length
            }
            if args.use_phrase:
                inputs.update({
                    'phrase_index': batch[7],
                    'img_index': batch[8],
                    'phrase_layer': args.phrase_layer
                })
            bs = batch[0].shape[0]
            outputs = model(**inputs)
            if args.use_phrase:
                loss, logits, r_loss, f_loss, pseudo_labels, wra_loss = outputs
                if args.n_gpu > 1:
                    wra_loss = wra_loss.mean()
                wra_loss = wra_loss.item()
            else:
                loss, logits, r_loss, f_loss, pseudo_labels = outputs
                wra_loss = 0
            if args.n_gpu > 1: 
                loss = loss.mean() # mean() to average on multi-gpu parallel training
                r_loss = r_loss.mean()
                f_loss = f_loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            # pseudo_labels = torch.cat([torch.ones(sim_mat.shape[0]), torch.zeros(bs)], dim=0).to(dtype=torch.long, device=logits.device)
            batch_score = compute_score_with_logits(logits, pseudo_labels).sum()
            batch_acc = batch_score.item() / (args.train_batch_size * 2)
            global_loss += loss.item()
            global_r_loss += r_loss.item()
            global_f_loss += f_loss.item()
            global_w_loss += wra_loss
            global_acc += batch_acc
            if (step + 1) % args.gradient_accumulation_steps == 0:
                global_step += 1
                scheduler.step()
                optimizer.step()
                model.zero_grad()
                if global_step % args.logging_steps == 0:
                    logger.info("Epoch: {}, global_step: {}, lr: {:.6f}, loss: {:.4f} ({:.4f}), " \
                        "CLIP_loss: {:.4f} ({:.4f}), HN_loss: {:.4f} ({:.4f}), wra_loss: {:.4f} ({:.4f}), score: {:.4f} ({:.4f})".format(epoch, global_step, 
                        optimizer.param_groups[0]["lr"], loss, global_loss / global_step, 
                        r_loss.item(), global_r_loss/global_step, f_loss.item(), global_f_loss/global_step, wra_loss, global_w_loss/global_step ,batch_acc, global_acc / global_step)
                    )

                if (args.save_steps > 0 and global_step % args.save_steps == 0) or \
                        global_step == t_total:
                    save_checkpoint(model, tokenizer, args, epoch, global_step) 
                    # evaluation
                    if args.evaluate_during_training: 
                        logger.info("Perform evaluation at step: %d" % (global_step))
                        # only VSE retrieval
                        coarse_sim = test_coarse(args, model, val_dataset)
                        eval_result, caption_index, image_index = evaluate_coarse(val_dataset, coarse_sim)
                        # caption index and image index
                        eval_i2t_result, _ = test_fine_i2t(args, model, val_dataset, caption_index=caption_index)
                        eval_t2i_result = test_fine_t2i(args, model, val_dataset, image_index=image_index)
                        print('fine inference:')
                        # print(eval_i2t_result, eval_t2i_result)
                        eval_result = evaluate_fine(eval_i2t_result, eval_t2i_result)

                        rank_accs = eval_result['i2t_retrieval']
                        if rank_accs['R@1'] > best_score:
                            best_score = rank_accs['R@1']
                        epoch_log = {'epoch': epoch, 'global_step': global_step, 
                                     'R1': rank_accs['R@1'], 'R5': rank_accs['R@5'], 
                                     'R10': rank_accs['R@10'], 'best_R1':best_score}
                        log_json.append(epoch_log)
                        with open(args.output_dir + '/eval_logs.json', 'w') as fp:
                            json.dump(log_json, fp) 
    return global_step, global_loss / global_step

def prepare_inputs(inputs, args):
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            if inputs[k].dtype != torch.int64:
                # NLP models inputs are int64 and those get adjusted to the right dtype of the
                # embedding. Other models such as wav2vec2's inputs are already float and thus
                # may need special handling to match the dtypes of the model
                inputs[k]=v.to(dtype=args.dtype)
    return inputs

def test_coarse(args, model, eval_dataset):
    # 2 stage evaluation
    if hasattr(model, 'module'):
        model.module.forward_mod = 'coarse'
    else:
        model.forward_mod = 'coarse'
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_dataset.unset_index()
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler,
            batch_size=args.eval_batch_size, num_workers=args.num_workers)
    
    logger.info("Num examples = {}".format(len(eval_dataset)))
    logger.info("Evaluation batch size = {}".format(args.eval_batch_size))
    model.eval()
    results = {}
    softmax = nn.Softmax(dim=1)
    full_txt_emb = []
    full_img_emb = []
    for indexs, batch in tqdm(eval_dataloader):
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {
                'input_ids_a':      batch[0],
                'attention_mask_a': batch[1],
                'token_type_ids_a': batch[2],
                'input_ids_b':      batch[3],
                'attention_mask_b': batch[4],
                'token_type_ids_b': batch[5],
                'img_feats':        batch[6],
                'max_tag_length': args.max_tag_length
            }
            inputs = prepare_inputs(inputs, args)
            global_txt, global_img = model(**inputs)[:2]
            full_txt_emb.append(global_txt)
            full_img_emb.append(global_img)
    with torch.no_grad():
        full_txt_emb = torch.cat(full_txt_emb, dim=0)
        full_img_emb = torch.cat(full_img_emb, dim=0)
        torch.save(full_txt_emb, '/opt/tiger/tmp_dir/txt_emb.pt')
        torch.save(full_img_emb, '/opt/tiger/tmp_dir/img_emb.pt')
        num_imgs = int(full_img_emb.shape[0] / args.num_captions_per_img_train)
        assert(full_img_emb.shape[0] % args.num_captions_per_img_train == 0)
        select_index = [i*args.num_captions_per_img_train for i in range(num_imgs)]
        full_img_emb = full_img_emb[select_index]
        full_sims = full_img_emb @ full_txt_emb.t()
        print(full_sims.shape)
    return full_sims.detach().cpu().numpy()

def test_fine_t2i(args, model, eval_dataset, image_index):
    # 2 stage evaluation
    if hasattr(model, 'module'):
        model.module.forward_mod = 'fine'
    else:
        model.forward_mod = 'fine'
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_dataset.set_image_index(image_index)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler,
            batch_size=args.eval_batch_size, num_workers=args.num_workers)
    
    logger.info("Num examples = {}".format(len(eval_dataset)))
    logger.info("Evaluation batch size = {}".format(args.eval_batch_size))
    model.eval()
    results = {}
    softmax = nn.Softmax(dim=1)
    for indexs, batch in tqdm(eval_dataloader):
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {
                'input_ids_a':      batch[0],
                'attention_mask_a': batch[1],
                'token_type_ids_a': batch[2],
                'input_ids_b':      batch[3],
                'attention_mask_b': batch[4],
                'token_type_ids_b': batch[5],
                'img_feats':        batch[6],
                'max_tag_length': args.max_tag_length
            }
            inputs = prepare_inputs(inputs, args)
            logits = model(**inputs)
            if args.num_labels == 2:
                probs = softmax(logits)
                result = probs[:, 1] # the confidence to be a matched pair
            else:
                result = logits
            result = [_.to(torch.device("cpu")) for _ in result]
            results.update({idx.item(): res.item() for idx, res in zip(indexs, result)})
    return compute_ranks_t2i(eval_dataset, results)


def test_fine_i2t(args, model, eval_dataset, caption_index):
    # 2 stage evaluation
    if hasattr(model, 'module'):
        model.module.forward_mod = 'fine'
    else:
        model.forward_mod = 'fine'
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_dataset.set_caption_index(caption_index)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler,
            batch_size=args.eval_batch_size, num_workers=args.num_workers)
    
    logger.info("Num examples = {}".format(len(eval_dataset)))
    logger.info("Evaluation batch size = {}".format(args.eval_batch_size))
    model.eval()
    results = {}
    softmax = nn.Softmax(dim=1)
    for indexs, batch in tqdm(eval_dataloader):
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {
                'input_ids_a':      batch[0],
                'attention_mask_a': batch[1],
                'token_type_ids_a': batch[2],
                'input_ids_b':      batch[3],
                'attention_mask_b': batch[4],
                'token_type_ids_b': batch[5],
                'img_feats':        batch[6],
                'max_tag_length': args.max_tag_length
            }
            inputs = prepare_inputs(inputs, args)
            logits = model(**inputs)
            # print(logits.shape)
            if args.num_labels == 2:
                probs = softmax(logits)
                result = probs[:, 1] # the confidence to be a matched pair
            else:
                result = logits
            result = [_.to(torch.device("cpu")) for _ in result]
            # print(indexs)
            results.update({idx.item(): res.item() for idx, res in zip(indexs, result)})
    return compute_ranks(eval_dataset,results)



def evaluate(eval_dataset, test_results):
    i2t_ranks, t2i_ranks = compute_ranks(eval_dataset, test_results)
    rank = [1, 5, 10]
    i2t_accs = [sum([_ < r for _ in i2t_ranks]) / len(i2t_ranks) for r in rank]
    logger.info("I2T Retrieval: {:.4f} @ R1, {:.4f} @ R5, {:.4f} @ R10".format(
                i2t_accs[0], i2t_accs[1], i2t_accs[2]))
    eval_result = {"i2t_retrieval": {"R@1": i2t_accs[0], "R@5": i2t_accs[1], "R@10": i2t_accs[2]}}
    if t2i_ranks:
        t2i_accs = [sum([_ < r for _ in t2i_ranks]) / len(t2i_ranks) for r in rank]
        logger.info("T2I Retrieval: {:.4f} @ R1, {:.4f} @ R5, {:.4f} @ R10".format(
                    t2i_accs[0], t2i_accs[1], t2i_accs[2]))
        eval_result["t2i_retrieval"] = {"R@1": t2i_accs[0], "R@5": t2i_accs[1], "R@10": t2i_accs[2]}
    return eval_result

def evaluate_fine(i2t_ranks, t2i_ranks):
    # i2t_ranks, t2i_ranks = compute_ranks(eval_dataset, test_results)
    rank = [1, 5, 10]
    i2t_accs = [sum([_ < r for _ in i2t_ranks]) / len(i2t_ranks) for r in rank]
    logger.info("I2T Retrieval: {:.4f} @ R1, {:.4f} @ R5, {:.4f} @ R10".format(
                i2t_accs[0], i2t_accs[1], i2t_accs[2]))
    eval_result = {"i2t_retrieval": {"R@1": i2t_accs[0], "R@5": i2t_accs[1], "R@10": i2t_accs[2]}}
    if t2i_ranks:
        t2i_accs = [sum([_ < r for _ in t2i_ranks]) / len(t2i_ranks) for r in rank]
        logger.info("T2I Retrieval: {:.4f} @ R1, {:.4f} @ R5, {:.4f} @ R10".format(
                    t2i_accs[0], t2i_accs[1], t2i_accs[2]))
        eval_result["t2i_retrieval"] = {"R@1": t2i_accs[0], "R@5": t2i_accs[1], "R@10": t2i_accs[2]}
    return eval_result


def evaluate_coarse(eval_dataset, test_results):
    i2t_ranks, t2i_ranks, caption_index, image_index = compute_ranks_coarse(eval_dataset, test_results)
    rank = [1, 5, 10]
    i2t_accs = [sum([_ < r for _ in i2t_ranks]) / len(i2t_ranks) for r in rank]
    logger.info("I2T Retrieval: {:.4f} @ R1, {:.4f} @ R5, {:.4f} @ R10".format(
                i2t_accs[0], i2t_accs[1], i2t_accs[2]))
    eval_result = {"i2t_retrieval": {"R@1": i2t_accs[0], "R@5": i2t_accs[1], "R@10": i2t_accs[2]}}
    if t2i_ranks:
        t2i_accs = [sum([_ < r for _ in t2i_ranks]) / len(t2i_ranks) for r in rank]
        logger.info("T2I Retrieval: {:.4f} @ R1, {:.4f} @ R5, {:.4f} @ R10".format(
                    t2i_accs[0], t2i_accs[1], t2i_accs[2]))
        eval_result["t2i_retrieval"] = {"R@1": t2i_accs[0], "R@5": t2i_accs[1], "R@10": t2i_accs[2]}
    return eval_result, caption_index, image_index


def get_predict_file(args):
    cc = []
    data = op.basename(op.join(args.data_dir, '')[:-1])
    if data != 'coco_ir':
        cc.append(data)
    cc.append(args.test_split)
    if args.add_od_labels:
        cc.append('wlabels{}'.format(args.od_label_type))
    return op.join(args.eval_model_dir, '{}.results.pt'.format('.'.join(cc))) 


def restore_training_settings(args):
    assert not args.do_train and (args.do_test or args.do_eval)
    train_args = torch.load(op.join(args.eval_model_dir, 'training_args.bin'))
    override_params = ['do_lower_case', 'img_feature_type', 'max_seq_length', 
            'max_img_seq_length', 'add_od_labels', 'od_label_type',
            'use_img_layernorm', 'img_layer_norm_eps']
    for param in override_params:
        if hasattr(train_args, param):
            train_v = getattr(train_args, param)
            test_v = getattr(args, param)
            if train_v != test_v:
                logger.warning('Override {} with train args: {} -> {}'.format(param,
                    test_v, train_v))
                setattr(args, param, train_v)
    return args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default='datasets/coco_ir', type=str, required=False,
                        help="The input data dir with all required files.")
    parser.add_argument("--img_feat_file", default='datasets/coco_ir/features.tsv', type=str, required=False,
                        help="The absolute address of the image feature file.")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=False,
                        help="Path to pre-trained model or model type. required for training.")
    parser.add_argument("--output_dir", default='output/', type=str, required=False,
                        help="The output directory to save checkpoint and test results.")
    parser.add_argument("--loss_type", default='sfmx', type=str, 
                        help="Loss function types: support kl, sfmx")
    parser.add_argument("--config_name", default="", type=str, 
                        help="Pretrained config name or path if not the same as model_name.")
    parser.add_argument("--tokenizer_name", default="", type=str, 
                        help="Pretrained tokenizer name or path if not the same as model_name.")
    parser.add_argument("--max_seq_length", default=70, type=int,
                        help="The maximum total input sequence length after tokenization. "
                             "Sequences longer than this will be truncated, "
                             "sequences shorter will be padded."
                             "This number is calculated on COCO dataset" 
                             "If add object detection labels, the suggested length should be 70.")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_test", action='store_true', help="Whether to run inference.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run performance valuation."
                       "do not activate if we want to inference on dataset without gt labels.")
    parser.add_argument("--test_split", default='test', type=str, help='data split name.')
    parser.add_argument("--eval_img_keys_file", default='', type=str, 
                        help="image key tsv to select a subset of images for evaluation. "
                        "This is useful in 5-folds evaluation. The topn index file is not " 
                        "needed in this case.")
    parser.add_argument("--eval_caption_index_file", default='', type=str, 
                        help="index of a list of (img_key, cap_idx) for each image."
                        "this is used to perform re-rank using hard negative samples."
                        "useful for validation set to monitor the performance during training.")
    parser.add_argument("--cross_image_eval", action='store_true', 
                        help="perform cross image inference, ie. each image with all texts from other images.")
    parser.add_argument("--add_od_labels", default=False, action='store_true', 
                        help="Whether to add object detection labels or not.")
    parser.add_argument("--od_label_type", default='vg', type=str, 
                        help="label type, support vg, gt, oid")
    parser.add_argument("--att_mask_type", default='CLR', type=str, 
                        help="attention mask type, support ['CL', 'CR', 'LR', 'CLR']"
                        "C: caption, L: labels, R: image regions; CLR is full attention by default."
                        "CL means attention between caption and labels."
                        "please pay attention to the order CLR, which is the default concat order.")
    parser.add_argument("--do_lower_case", action='store_true', 
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--drop_out", default=0.1, type=float, help="Drop out in BERT.")
    parser.add_argument("--max_img_seq_length", default=50, type=int, 
                        help="The maximum total input image sequence length.")
    parser.add_argument("--img_feature_dim", default=2054, type=int, 
                        help="The Image Feature Dimension.")
    parser.add_argument("--img_feature_type", default='frcnn', type=str,
                        help="Image feature type.")
    parser.add_argument("--use_img_layernorm", type=int, default=1,
                        help="Normalize image features with bertlayernorm")
    parser.add_argument("--img_layer_norm_eps", default=1e-12, type=float,
                        help="The eps in image feature laynorm layer")
    parser.add_argument("--per_gpu_train_batch_size", default=32, type=int, 
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=64, type=int, 
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--output_mode", default='classification', type=str,
                        help="output mode, support classification or regression.")
    parser.add_argument("--num_labels", default=2, type=int, 
                        help="num_labels is 2 for classification and 1 for regression.")
    parser.add_argument("--num_captions_per_img_train", default=5, type=int,
                        help="number of positive matched captions for each training image.")
    parser.add_argument("--num_captions_per_img_val", default=5, type=int,
                        help="number of captions for each testing image.")
    parser.add_argument('--num_images_per_cap_val', type=int, default=128)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before backward.")
    parser.add_argument("--learning_rate", default=2e-5, type=float, help="The initial lr.")
    parser.add_argument("--weight_decay", default=0.05, type=float, help="Weight deay.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup.")
    parser.add_argument("--scheduler", default='linear', type=str, help="constant or linear.")
    parser.add_argument("--num_workers", default=4, type=int, help="Workers in dataloader.")
    parser.add_argument("--num_train_epochs", default=20, type=int, 
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int, 
                        help="Total number of training steps. Override num_train_epochs.")
    parser.add_argument('--logging_steps', type=int, default=20, help="Log every X steps.")
    parser.add_argument('--save_steps', type=int, default=-1, 
                        help="Save checkpoint every X steps. Will also perform evaluatin.")
    parser.add_argument("--evaluate_during_training", action='store_true', 
                        help="Run evaluation during training at each save_steps.")
    parser.add_argument("--eval_model_dir", type=str, default='', 
                        help="Model directory for evaluation.")
    parser.add_argument("--no_cuda", action='store_true', help="Avoid using CUDA.")
    parser.add_argument('--seed', type=int, default=88, help="random seed for initialization.")
    parser.add_argument('--extra_concept', action='store_true', help="Whether to add more related concepts from the concept graph.")
    parser.add_argument('--num_extra_concept', type=int, default=5, help="Number of extra concapts added")
    parser.add_argument('--devices', type=str, default='0,1,2,3,4,5,6,7', help="Which GPUs to use")
    parser.add_argument('--half_evaluation', action='store_true', help='Whther to use half precision for evaluation')
    parser.add_argument('--dataset_name', type=str, default='flickr', help='which dataset is using')
    parser.add_argument('--max_phrases', type=int, default=5)
    parser.add_argument('--sent_sg_json', type=str, default=None)
    parser.add_argument('--id2node', type=str, default=None)
    parser.add_argument('--clip_neg_sampling', action='store_true')
    parser.add_argument('--clip_neg_prob', type=float, default=0.4)
    parser.add_argument('--max_tag_length', type=int, default=20)
    parser.add_argument('--phrase_layer', type=int, default=2)
    parser.add_argument('--use_phrase', action='store_true')
    parser.add_argument('--no_itm', action='store_true')
    parser.add_argument('--eval_all_checkpoints', action='store_true')
    args = parser.parse_args()

    global logger
    mkdir(args.output_dir)
    logger = setup_logger("vlpretrain", args.output_dir, 0)

    # os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    set_seed(args.seed, args.n_gpu)
    logger.warning("Device: %s, n_gpu: %s", args.device, args.n_gpu)
    logger.info('output_mode: {}, #Labels: {}'.format(args.output_mode, args.num_labels))
 
    config_class, tokenizer_class = BertConfig, BertTokenizer
    model_class = BiImageBertForRetrieval
    if args.do_train:
        config = config_class.from_pretrained(args.config_name if args.config_name else \
            args.model_name_or_path, num_labels=args.num_labels, finetuning_task='ir')
        tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name \
            else args.model_name_or_path, do_lower_case=args.do_lower_case)
        config.img_feature_dim = args.img_feature_dim
        config.img_feature_type = args.img_feature_type
        config.hidden_dropout_prob = args.drop_out
        config.loss_type = args.loss_type
        config.img_layer_norm_eps = args.img_layer_norm_eps
        config.use_img_layernorm = args.use_img_layernorm
        model = model_class.from_pretrained(args.model_name_or_path, 
            from_tf=bool('.ckpt' in args.model_name_or_path), config=config)
        args.dtype = torch.float32
    else:
        checkpoint = args.eval_model_dir
        assert op.isdir(checkpoint)
        config = config_class.from_pretrained(checkpoint)
        tokenizer = tokenizer_class.from_pretrained(checkpoint)
        logger.info("Evaluate the following checkpoint: %s", checkpoint)
        model = model_class.from_pretrained(checkpoint, config=config)
        if args.half_evaluation:
            model = model.half()
            args.dtype = torch.float16
        else:
            args.dtype = torch.float32

    model.to(args.device)
    logger.info("Training/evaluation parameters %s", args)
    if args.do_train:
        train_dataset = RetrievalDataset(tokenizer, args, 'train', is_train=True)
        if args.evaluate_during_training:
            if 'coco_ir' not in args.data_dir:
                val_split = 'val'
            else:
                val_split = 'minival'
            val_dataset = RetrievalDataset(tokenizer, args, val_split, is_train=False)
        else:
            val_dataset = None
        global_step, avg_loss = train(args, train_dataset, val_dataset, model, tokenizer)
        logger.info("Training done: total_step = %s, avg loss = %s", global_step, avg_loss)

    # inference and evaluation
    if args.do_test or args.do_eval:
        if args.eval_all_checkpoints:
            checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            args = restore_training_settings(args)
            test_dataset = RetrievalDataset(tokenizer, args, args.test_split, is_train=False)
            for checkpoint in checkpoints:
                assert op.isdir(checkpoint)
                logger.info("Evaluate the following checkpoint: %s", checkpoint)
                model = model_class.from_pretrained(checkpoint, config=config)
                if args.half_evaluation:
                    model = model.half()
                    args.dtype = torch.float16
                else:
                    args.dtype = torch.float32
                model.to(args.device)
                if args.n_gpu > 1:
                    model = torch.nn.DataParallel(model)

                #pred_file = get_predict_file(args)
                # if op.isfile(pred_file):
                #     logger.info("Prediction file exist, skip inference.")
                #     if args.do_eval:
                #         test_result = torch.load(pred_file)
                # else:
                #     test_result = test(args, model, test_dataset)
                #     torch.save(test_result, pred_file)
                #     logger.info("Prediction results saved to {}.".format(pred_file))

                coarse_sim = test_coarse(args, model, test_dataset)
                eval_result, caption_index, image_index = evaluate_coarse(test_dataset, coarse_sim)
                # caption index and image index
                eval_i2t_result, _ = test_fine_i2t(args, model, test_dataset, caption_index=caption_index)
                eval_t2i_result = test_fine_t2i(args, model, test_dataset, image_index=image_index)
                print('fine inference:')
                # print(eval_i2t_result, eval_t2i_result)
                if args.do_eval:
                    eval_result = evaluate_fine(eval_i2t_result, eval_t2i_result)
                    # result_file = op.splitext(pred_file)[0] + '.eval.json'
                    result_file = op.join(checkpoint, 'test_eval.json')
                    with open(result_file, 'w') as f:
                        json.dump(eval_result, f)
                    logger.info("Evaluation results saved to {}.".format(result_file))
        else:
            args = restore_training_settings(args)
            test_dataset = RetrievalDataset(tokenizer, args, args.test_split, is_train=False)
            checkpoint = args.eval_model_dir
            assert op.isdir(checkpoint)
            logger.info("Evaluate the following checkpoint: %s", checkpoint)
            model = model_class.from_pretrained(checkpoint, config=config)
            if args.half_evaluation:
                model = model.half()
                args.dtype = torch.float16
            else:
                args.dtype = torch.float32
            model.to(args.device)
            if args.n_gpu > 1:
                model = torch.nn.DataParallel(model)

            pred_file = get_predict_file(args)
            # if op.isfile(pred_file):
            #     logger.info("Prediction file exist, skip inference.")
            #     if args.do_eval:
            #         test_result = torch.load(pred_file)
            # else:
            #     test_result = test(args, model, test_dataset)
            #     torch.save(test_result, pred_file)
            #     logger.info("Prediction results saved to {}.".format(pred_file))

            coarse_sim = test_coarse(args, model, test_dataset)
            eval_result, caption_index, image_index = evaluate_coarse(test_dataset, coarse_sim)
            # caption index and image index
            eval_i2t_result, _ = test_fine_i2t(args, model, test_dataset, caption_index=caption_index)
            eval_t2i_result = test_fine_t2i(args, model, test_dataset, image_index=image_index)
            print('fine inference:')
            # print(eval_i2t_result, eval_t2i_result)
            if args.do_eval:
                eval_result = evaluate_fine(eval_i2t_result, eval_t2i_result)
                result_file = op.splitext(pred_file)[0] + '.eval.json'
                with open(result_file, 'w') as f:
                    json.dump(eval_result, f)
                logger.info("Evaluation results saved to {}.".format(result_file))


if __name__ == "__main__":
    main()
