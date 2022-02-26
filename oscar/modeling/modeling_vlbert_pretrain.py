# Copyright (c) 2020 Microsoft Corporation. Licensed under the MIT license.
 
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss

from transformers.pytorch_transformers.modeling_bert import (BertEmbeddings, 
        BertSelfAttention, BertAttention, BertEncoder, BertLayer, 
        BertSelfOutput, BertIntermediate, BertOutput,
        BertPooler, BertLayerNorm, BertPreTrainedModel,
		BertPredictionHeadTransform, BertOnlyMLMHead, BertLMPredictionHead,
        BertConfig, BERT_PRETRAINED_MODEL_ARCHIVE_MAP, GELU,
        load_tf_weights_in_bert)
from .modeling_utils import CaptionPreTrainedModel, ImgPreTrainedModel
from ..utils.cbs import ConstrainedBeamSearch, select_best_beam_with_constraints
import time, random, copy
import numpy as np
# from .clipp.clip import *
# from .clipp.model import *

logger = logging.getLogger(__name__)

def soft_cross_entropy(target, input_prob, reduction='mean'):
    logprobs = nn.functional.log_softmax(input_prob, dim=1)
    target = target.float()
    neg_target = 1-target
    target = torch.stack([neg_target, target], dim=1)
    batchloss = - torch.sum(target.view(target.shape[0], -1) * logprobs, dim=1)
    if reduction == 'none':
        return batchloss
    elif reduction == 'mean':
        return torch.mean(batchloss)
    elif reduction == 'sum':
        return torch.sum(batchloss)
    else:
        raise NotImplementedError('Unsupported reduction mode.')

def get_parameter_dtype(parameter):
    try:
        return next(parameter.parameters()).dtype
    except StopIteration:
        # For nn.DataParallel compatibility in PyTorch 1.5

        def find_tensor_attributes(module):
            tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
            return tuples

        gen = parameter._named_members(get_members_fn=find_tensor_attributes)
        first_tuple = next(gen)
        return first_tuple[1].dtype

class CaptionBertSelfAttention(BertSelfAttention):
    """
    Modified from BertSelfAttention to add support for output_hidden_states.
    """
    def __init__(self, config):
        super(CaptionBertSelfAttention, self).__init__(config)

    def forward(self, hidden_states, attention_mask, head_mask=None,
            history_state=None):
        if history_state is not None:
            x_states = torch.cat([history_state, hidden_states], dim=1)
            mixed_query_layer = self.query(hidden_states)
            mixed_key_layer = self.key(x_states)
            mixed_value_layer = self.value(x_states)
        else:
            mixed_query_layer = self.query(hidden_states)
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)
        return outputs


class CaptionBertAttention(BertAttention):
    """
    Modified from BertAttention to add support for output_hidden_states.
    """
    def __init__(self, config):
        super(CaptionBertAttention, self).__init__(config)
        self.self = CaptionBertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask, head_mask=None,
            history_state=None):
        self_outputs = self.self(input_tensor, attention_mask, head_mask, history_state)
        attention_output = self.output(self_outputs[0], input_tensor)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class CaptionBertEncoder(BertEncoder):
    """
    Modified from BertEncoder to add support for output_hidden_states.
    """
    def __init__(self, config):
        super(CaptionBertEncoder, self).__init__(config)
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.num_layers = config.num_hidden_layers
        self.layer = nn.ModuleList([CaptionBertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, head_mask=None,
                encoder_history_states=None, return_at_layer=None):
        all_hidden_states = ()
        all_attentions = ()
        stage_output = None
        mid_output = None
        if isinstance(attention_mask, list):
            num_layer_per_phase = math.ceil(self.num_layers / len(attention_mask))
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            history_state = None if encoder_history_states is None else encoder_history_states[i]
            if isinstance(attention_mask, list):
                layer_outputs = layer_module(
                    hidden_states, attention_mask[i//num_layer_per_phase], head_mask[i],
                    history_state)
                if i == num_layer_per_phase-1:
                    stage_output = layer_outputs[0]
            else:
                layer_outputs = layer_module(
                    hidden_states, attention_mask, head_mask[i],
                    history_state)
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
            if return_at_layer is not None and i==return_at_layer:
                mid_output = hidden_states

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        if stage_output is not None:
            outputs = outputs + (stage_output,)
        return outputs, mid_output  # outputs, (hidden states), (attentions), (stage outputs)


class CaptionBertLayer(BertLayer):
    """
    Modified from BertLayer to add support for output_hidden_states.
    """
    def __init__(self, config):
        super(CaptionBertLayer, self).__init__(config)
        self.attention = CaptionBertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask, head_mask=None,
                history_state=None):
        attention_outputs = self.attention(hidden_states, attention_mask,
                head_mask, history_state)
        attention_output = attention_outputs[0]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + attention_outputs[1:]  # add attentions if we output them
        return outputs


class BertImgModel(BertPreTrainedModel):
    """ Expand from BertModel to handle image region features as input
    """
    def __init__(self, config):
        super(BertImgModel, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = CaptionBertEncoder(config)
        self.pooler = BertPooler(config)

        self.img_dim = config.img_feature_dim
        logger.info('BertImgModel Image Dimension: {}'.format(self.img_dim))
        self.img_feature_type = config.img_feature_type
        if hasattr(config, 'use_img_layernorm'):
            self.use_img_layernorm = config.use_img_layernorm
        else:
            self.use_img_layernorm = None

        if config.img_feature_type == 'dis_code':
            self.code_embeddings = nn.Embedding(config.code_voc, config.code_dim, padding_idx=0)
            self.img_embedding = nn.Linear(config.code_dim, self.config.hidden_size, bias=True)
        elif config.img_feature_type == 'dis_code_t': # transpose
            self.code_embeddings = nn.Embedding(config.code_voc, config.code_dim, padding_idx=0)
            self.img_embedding = nn.Linear(config.code_size, self.config.hidden_size, bias=True)
        elif config.img_feature_type == 'dis_code_scale': # scaled
            self.input_embeddings = nn.Linear(config.code_dim, config.code_size, bias=True)
            self.code_embeddings = nn.Embedding(config.code_voc, config.code_dim, padding_idx=0)
            self.img_embedding = nn.Linear(config.code_dim, self.config.hidden_size, bias=True)
        else:
            self.img_embedding = nn.Linear(self.img_dim, self.config.hidden_size, bias=True)
            self.dropout = nn.Dropout(config.hidden_dropout_prob)
            if self.use_img_layernorm:
                self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.img_layer_norm_eps)

        self.apply(self.init_weights)

    def _resize_token_embeddings(self, new_num_tokens):
        old_embeddings = self.embeddings.word_embeddings
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
        self.embeddings.word_embeddings = new_embeddings
        return self.embeddings.word_embeddings

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
            position_ids=None, head_mask=None, img_feats=None,
            encoder_history_states=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        if isinstance(attention_mask, list):
            extended_attention_mask = []
            for i in range(len(attention_mask)):
                if attention_mask[i].dim() == 2:
                    extended_attention_mask_c = attention_mask[i].unsqueeze(1).unsqueeze(2)
                elif attention_mask[i].dim() == 3:
                    extended_attention_mask_c = attention_mask[i].unsqueeze(1)
                else:
                    raise NotImplementedError
                extended_attention_mask_c = extended_attention_mask_c.to(dtype=self.dtype) # fp16 compatibility (editted here to adapt to pytorch>=1.5.0)
                extended_attention_mask_c = (1.0 - extended_attention_mask_c) * -10000.0
                extended_attention_mask.append(extended_attention_mask_c)
        else:
            if attention_mask.dim() == 2:
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            elif attention_mask.dim() == 3:
                extended_attention_mask = attention_mask.unsqueeze(1)
            else:
                raise NotImplementedError

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            # extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
            extended_attention_mask = extended_attention_mask.to(dtype=self.dtype) # fp16 compatibility (editted here to adapt to pytorch>=1.5.0)
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
            # switch to float if needed + fp16 compatibility
            # head_mask = head_mask.to(dtype=next(self.parameters()).dtype) # switch to fload if need + fp16 compatibility
            head_mask = head_mask.to(dtype=self.dtype) # switch to fload if need + fp16 compatibility (edited here to adapt to pytorch>=1.5.0)
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(input_ids, position_ids=position_ids,
                token_type_ids=token_type_ids)
        if encoder_history_states:
            assert img_feats is None, "Cannot take image features while using encoder history states"

        if img_feats is not None:
            if self.img_feature_type == 'dis_code':
                code_emb = self.code_embeddings(img_feats)
                img_embedding_output = self.img_embedding(code_emb)
            elif self.img_feature_type == 'dis_code_t': # transpose
                code_emb = self.code_embeddings(img_feats)
                code_emb = code_emb.permute(0, 2, 1)
                img_embedding_output = self.img_embedding(code_emb)
            elif self.img_feature_type == 'dis_code_scale': # left scaled
                code_emb = self.code_embeddings(img_feats)
                img_embedding_output = self.img_embedding(code_emb)
            else:
                img_embedding_output = self.img_embedding(img_feats)
                if self.use_img_layernorm:
                    img_embedding_output = self.LayerNorm(img_embedding_output)

                # add dropout on image embedding
                img_embedding_output = self.dropout(img_embedding_output)

            # concatenate two embeddings
            embedding_output = torch.cat((embedding_output, img_embedding_output), 1)

        encoder_outputs = self.encoder(embedding_output,
                extended_attention_mask, head_mask=head_mask,
                encoder_history_states=encoder_history_states)
        sequence_output = encoder_outputs[0]
        if len(encoder_outputs) > 1:
            stage_outputs = encoder_outputs[1]
        pooled_output = self.pooler(sequence_output)

        # add hidden_states and attentions if they are here
        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]
        return outputs

    @property
    def dtype(self):
        return get_parameter_dtype(self)

class BiBertImgModel2(BertPreTrainedModel):
    """ Expand from BertImgModel to handle 2-stream input
    """
    def __init__(self, config):
        super(BiBertImgModel2, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        half_config = copy.deepcopy(config)
        half_config.num_hidden_layers = half_config.num_hidden_layers//2 # 2-phases
        self.vis_encoder = CaptionBertEncoder(half_config)
        self.txt_encoder = CaptionBertEncoder(half_config)
        self.mul_encoder = CaptionBertEncoder(half_config)
        scale = config.hidden_size ** -0.5
        self.txt_proj = nn.Parameter(scale * torch.randn(config.hidden_size, config.hidden_size))
        self.vis_proj = nn.Parameter(scale * torch.randn(config.hidden_size, config.hidden_size))
        self.pooler = BertPooler(config)

        self.img_dim = config.img_feature_dim
        logger.info('BertImgModel Image Dimension: {}'.format(self.img_dim))
        self.img_feature_type = config.img_feature_type
        if hasattr(config, 'use_img_layernorm'):
            self.use_img_layernorm = config.use_img_layernorm
        else:
            self.use_img_layernorm = None

        if config.img_feature_type == 'dis_code':
            self.code_embeddings = nn.Embedding(config.code_voc, config.code_dim, padding_idx=0)
            self.img_embedding = nn.Linear(config.code_dim, self.config.hidden_size, bias=True)
        elif config.img_feature_type == 'dis_code_t': # transpose
            self.code_embeddings = nn.Embedding(config.code_voc, config.code_dim, padding_idx=0)
            self.img_embedding = nn.Linear(config.code_size, self.config.hidden_size, bias=True)
        elif config.img_feature_type == 'dis_code_scale': # scaled
            self.input_embeddings = nn.Linear(config.code_dim, config.code_size, bias=True)
            self.code_embeddings = nn.Embedding(config.code_voc, config.code_dim, padding_idx=0)
            self.img_embedding = nn.Linear(config.code_dim, self.config.hidden_size, bias=True)
        else:
            self.img_embedding = nn.Linear(self.img_dim, self.config.hidden_size, bias=True)
            self.dropout = nn.Dropout(config.hidden_dropout_prob)
            if self.use_img_layernorm:
                self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.img_layer_norm_eps)

        self.apply(self.init_weights)

    def _resize_token_embeddings(self, new_num_tokens):
        old_embeddings = self.embeddings.word_embeddings
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
        self.embeddings.word_embeddings = new_embeddings
        return self.embeddings.word_embeddings

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(self, input_ids_a, token_type_ids_a=None, attention_mask_a=None, max_tag_length=None,
            position_ids_a=None, input_ids_b=None, token_type_ids_b=None, attention_mask_b=None, phrase_layer=None,
            position_ids_b=None, head_mask=None, img_feats=None, encoder_history_states=None, encode_hn=False):
        if attention_mask_a is None:
            attention_mask_a = torch.ones_like(input_ids_a)

        if attention_mask_b is None:
            attention_mask_b = torch.ones_like(input_ids_b)

        if token_type_ids_a is None:
            token_type_ids_a = torch.zeros_like(input_ids_a)

        if token_type_ids_b is None:
            token_type_ids_b = torch.zeros_like(input_ids_b)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        if attention_mask_a.dim() == 2:
            extended_attention_mask_a = attention_mask_a.unsqueeze(1).unsqueeze(2)
        elif attention_mask_a.dim() == 3:
            extended_attention_mask_a = attention_mask_a.unsqueeze(1)
        else:
            raise NotImplementedError

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        # extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask_a = extended_attention_mask_a.to(dtype=self.dtype) # fp16 compatibility (editted here to adapt to pytorch>=1.5.0)
        extended_attention_mask_a = (1.0 - extended_attention_mask_a) * -10000.0

        if attention_mask_b.dim() == 2:
            extended_attention_mask_b = attention_mask_b.unsqueeze(1).unsqueeze(2)
        elif attention_mask_b.dim() == 3:
            extended_attention_mask_b = attention_mask_b.unsqueeze(1)
        else:
            raise NotImplementedError

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        # extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask_b = extended_attention_mask_b.to(dtype=self.dtype) # fp16 compatibility (editted here to adapt to pytorch>=1.5.0)
        extended_attention_mask_b = (1.0 - extended_attention_mask_b) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers/2, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
            # switch to float if needed + fp16 compatibility
            # head_mask = head_mask.to(dtype=next(self.parameters()).dtype) # switch to fload if need + fp16 compatibility
            head_mask = head_mask.to(dtype=self.dtype) # switch to fload if need + fp16 compatibility (edited here to adapt to pytorch>=1.5.0)
        else:
            head_mask = [None] * (self.config.num_hidden_layers//2)

        embedding_output_a = self.embeddings(input_ids_a, position_ids=position_ids_a,
                token_type_ids=token_type_ids_a)
        embedding_output_b = self.embeddings(input_ids_b, position_ids=position_ids_b,
                token_type_ids=token_type_ids_b)
        if encoder_history_states:
            assert img_feats is None, "Cannot take image features while using encoder history states"

        if img_feats is not None:
            if self.img_feature_type == 'dis_code':
                code_emb = self.code_embeddings(img_feats)
                img_embedding_output = self.img_embedding(code_emb)
            elif self.img_feature_type == 'dis_code_t': # transpose
                code_emb = self.code_embeddings(img_feats)
                code_emb = code_emb.permute(0, 2, 1)
                img_embedding_output = self.img_embedding(code_emb)
            elif self.img_feature_type == 'dis_code_scale': # left scaled
                code_emb = self.code_embeddings(img_feats)
                img_embedding_output = self.img_embedding(code_emb)
            else:
                img_embedding_output = self.img_embedding(img_feats)
                if self.use_img_layernorm:
                    img_embedding_output = self.LayerNorm(img_embedding_output)

                # add dropout on image embedding
                img_embedding_output = self.dropout(img_embedding_output)

            # concatenate two embeddings
            embedding_output_b = torch.cat((embedding_output_b, img_embedding_output), 1)

        # single-stream encoder
        txt_encoder_outputs = self.txt_encoder(embedding_output_a, extended_attention_mask_a,
                        head_mask=head_mask, encoder_history_states=encoder_history_states)[0][0]

        vis_encoder_outputs = self.vis_encoder(embedding_output_b, extended_attention_mask_b,
                        head_mask=head_mask, encoder_history_states=encoder_history_states)[0][0]
        
        only_vis_embeddings = vis_encoder_outputs[:, max_tag_length:, :]
        only_vis_mask = extended_attention_mask_b[:, :, :, max_tag_length:]

        # VSE-based retrieval loss similar to CLIP
        global_txt = F.normalize(txt_encoder_outputs[:, 0, :] @ self.txt_proj, p=2, dim=-1)
        global_img = F.normalize(vis_encoder_outputs[:, 0, :] @ self.vis_proj, p=2, dim=-1)
        sim_mat = global_txt @ global_img.t()

        if encode_hn:
            masked_sim_mat = sim_mat - 2 * torch.eye(sim_mat.shape[0], dtype=sim_mat.dtype, device=sim_mat.device)
            # hardest in-batch sample
            hard_img_index = torch.max(masked_sim_mat, dim=1)[1]
            hard_txt_index = torch.max(masked_sim_mat, dim=0)[1]
            # text-hard_img pair
            hard_img_seq = torch.cat([txt_encoder_outputs, 
                            torch.index_select(only_vis_embeddings, dim=0, index=hard_img_index)], dim=1)
            hard_img_mask = torch.cat([extended_attention_mask_a,
                            torch.index_select(only_vis_mask, dim=0, index=hard_img_index)], dim=-1)
            # hard_text-img pair
            hard_txt_seq = torch.cat([torch.index_select(txt_encoder_outputs, dim=0, index=hard_txt_index),
                                    only_vis_embeddings], dim=1)
            hard_txt_mask = torch.cat([torch.index_select(extended_attention_mask_a, dim=0, index=hard_txt_index),
                                    only_vis_mask], dim=-1)

            # select hard-img or hard-text with 0.5/0.5 probability
            n_sample = hard_img_seq.shape[0]
            dice_index = torch.randperm(n_sample, device=hard_img_seq.device)
            hard_seqs = torch.cat([torch.index_select(hard_img_seq, dim=0, index=dice_index[:(n_sample//2)]),
                            torch.index_select(hard_txt_seq, dim=0, index=dice_index[(n_sample//2):])], dim=0)
            hard_mask = torch.cat([torch.index_select(hard_img_mask, dim=0, index=dice_index[:(n_sample//2)]),
                            torch.index_select(hard_txt_mask, dim=0, index=dice_index[(n_sample//2):])], dim=0)

            # arange the hard image and text index for later processing!
            # print('hard_txt_index', hard_txt_index, 'dice_index',dice_index)
            hard_txt_index_full = torch.cat([torch.index_select(torch.arange(n_sample, device=hard_img_seq.device), dim=0, index=dice_index[:(n_sample//2)]),
                                    torch.index_select(hard_txt_index, dim=0, index=dice_index[(n_sample//2):])], dim=0)
            hard_img_index_full = torch.cat([torch.index_select(hard_img_index, dim=0, index=dice_index[:(n_sample//2)]), 
                                torch.index_select(torch.arange(n_sample, device=hard_img_seq.device), dim=0, index=dice_index[(n_sample//2):])], dim=0)

            
            # encoding for hard-negatives
            hard_encoder_outputs, mid_hard = self.mul_encoder(hard_seqs, hard_mask, head_mask=head_mask,
                            encoder_history_states=encoder_history_states, return_at_layer=phrase_layer)
            hard_encoder_outputs = hard_encoder_outputs[0]
            hard_pooled_output = self.pooler(hard_encoder_outputs)
        else:
            hard_txt_index_full = None
            hard_img_index_full = None
            hard_encoder_outputs = None
            hard_pooled_output = None
            mid_hard = None

        joint_bi_output = torch.cat([txt_encoder_outputs, only_vis_embeddings], dim=1)
        joint_bi_mask = torch.cat([extended_attention_mask_a, only_vis_mask], dim=-1)

        encoder_outputs, mid_joint = self.mul_encoder(joint_bi_output,
                joint_bi_mask, head_mask=head_mask, return_at_layer=phrase_layer,
                encoder_history_states=encoder_history_states)
        sequence_output = encoder_outputs[0]

        pooled_output = self.pooler(sequence_output)

        # add hidden_states and attentions if they are here
        outputs = (sequence_output, pooled_output,) + (hard_encoder_outputs, hard_pooled_output)
        single_stream_output = (txt_encoder_outputs, vis_encoder_outputs, sim_mat)
        hard_indexes = (hard_txt_index_full, hard_img_index_full)
        mid_output = (mid_joint, mid_hard)
        return outputs, single_stream_output, hard_indexes, mid_output

    def forward_single(self, input_ids_a, token_type_ids_a=None, attention_mask_a=None, max_tag_length=None,
            position_ids_a=None, input_ids_b=None, token_type_ids_b=None, attention_mask_b=None,
            position_ids_b=None, head_mask=None, img_feats=None, encoder_history_states=None):
        if attention_mask_a is None:
            attention_mask_a = torch.ones_like(input_ids_a)

        if attention_mask_b is None:
            attention_mask_b = torch.ones_like(input_ids_b)

        if token_type_ids_a is None:
            token_type_ids_a = torch.zeros_like(input_ids_a)

        if token_type_ids_b is None:
            token_type_ids_b = torch.zeros_like(input_ids_b)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        if attention_mask_a.dim() == 2:
            extended_attention_mask_a = attention_mask_a.unsqueeze(1).unsqueeze(2)
        elif attention_mask_a.dim() == 3:
            extended_attention_mask_a = attention_mask_a.unsqueeze(1)
        else:
            raise NotImplementedError

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        # extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask_a = extended_attention_mask_a.to(dtype=self.dtype) # fp16 compatibility (editted here to adapt to pytorch>=1.5.0)
        extended_attention_mask_a = (1.0 - extended_attention_mask_a) * -10000.0

        if attention_mask_b.dim() == 2:
            extended_attention_mask_b = attention_mask_b.unsqueeze(1).unsqueeze(2)
        elif attention_mask_b.dim() == 3:
            extended_attention_mask_b = attention_mask_b.unsqueeze(1)
        else:
            raise NotImplementedError

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        # extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask_b = extended_attention_mask_b.to(dtype=self.dtype) # fp16 compatibility (editted here to adapt to pytorch>=1.5.0)
        extended_attention_mask_b = (1.0 - extended_attention_mask_b) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers/2, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
            # switch to float if needed + fp16 compatibility
            # head_mask = head_mask.to(dtype=next(self.parameters()).dtype) # switch to fload if need + fp16 compatibility
            head_mask = head_mask.to(dtype=self.dtype) # switch to fload if need + fp16 compatibility (edited here to adapt to pytorch>=1.5.0)
        else:
            head_mask = [None] * (self.config.num_hidden_layers//2)

        embedding_output_a = self.embeddings(input_ids_a, position_ids=position_ids_a,
                token_type_ids=token_type_ids_a)
        embedding_output_b = self.embeddings(input_ids_b, position_ids=position_ids_b,
                token_type_ids=token_type_ids_b)
        if encoder_history_states:
            assert img_feats is None, "Cannot take image features while using encoder history states"

        if img_feats is not None:
            if self.img_feature_type == 'dis_code':
                code_emb = self.code_embeddings(img_feats)
                img_embedding_output = self.img_embedding(code_emb)
            elif self.img_feature_type == 'dis_code_t': # transpose
                code_emb = self.code_embeddings(img_feats)
                code_emb = code_emb.permute(0, 2, 1)
                img_embedding_output = self.img_embedding(code_emb)
            elif self.img_feature_type == 'dis_code_scale': # left scaled
                code_emb = self.code_embeddings(img_feats)
                img_embedding_output = self.img_embedding(code_emb)
            else:
                img_embedding_output = self.img_embedding(img_feats)
                if self.use_img_layernorm:
                    img_embedding_output = self.LayerNorm(img_embedding_output)

                # add dropout on image embedding
                img_embedding_output = self.dropout(img_embedding_output)

            # concatenate two embeddings
            embedding_output_b = torch.cat((embedding_output_b, img_embedding_output), 1)

        # single-stream encoder
        txt_encoder_outputs = self.txt_encoder(embedding_output_a, extended_attention_mask_a,
                        head_mask=head_mask, encoder_history_states=encoder_history_states)[0][0]

        vis_encoder_outputs = self.vis_encoder(embedding_output_b, extended_attention_mask_b,
                        head_mask=head_mask, encoder_history_states=encoder_history_states)[0][0]
        
        # VSE-based retrieval loss similar to CLIP
        global_txt = F.normalize(txt_encoder_outputs[:, 0, :] @ self.txt_proj, p=2, dim=-1)
        global_img = F.normalize(vis_encoder_outputs[:, 0, :] @ self.vis_proj, p=2, dim=-1)

        single_stream_output = (global_txt, global_img)
        return single_stream_output


    @property
    def dtype(self):
        return get_parameter_dtype(self)



class BiBertImgModel(BertPreTrainedModel):
    """ Expand from BertImgModel to handle 2-stream input
    """
    def __init__(self, config):
        super(BiBertImgModel, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        half_config = copy.deepcopy(config)
        half_config.num_hidden_layers = half_config.num_hidden_layers//2 # 2-phases
        self.vis_encoder = CaptionBertEncoder(half_config)
        self.txt_encoder = CaptionBertEncoder(half_config)
        self.mul_encoder = CaptionBertEncoder(half_config)
        self.pooler = BertPooler(config)
        scale = config.hidden_size ** -0.5
        self.txt_proj = nn.Parameter(scale * torch.randn(config.hidden_size, config.hidden_size))
        self.vis_proj = nn.Parameter(scale * torch.randn(config.hidden_size, config.hidden_size))

        self.img_dim = config.img_feature_dim
        logger.info('BertImgModel Image Dimension: {}'.format(self.img_dim))
        self.img_feature_type = config.img_feature_type
        if hasattr(config, 'use_img_layernorm'):
            self.use_img_layernorm = config.use_img_layernorm
        else:
            self.use_img_layernorm = None

        if config.img_feature_type == 'dis_code':
            self.code_embeddings = nn.Embedding(config.code_voc, config.code_dim, padding_idx=0)
            self.img_embedding = nn.Linear(config.code_dim, self.config.hidden_size, bias=True)
        elif config.img_feature_type == 'dis_code_t': # transpose
            self.code_embeddings = nn.Embedding(config.code_voc, config.code_dim, padding_idx=0)
            self.img_embedding = nn.Linear(config.code_size, self.config.hidden_size, bias=True)
        elif config.img_feature_type == 'dis_code_scale': # scaled
            self.input_embeddings = nn.Linear(config.code_dim, config.code_size, bias=True)
            self.code_embeddings = nn.Embedding(config.code_voc, config.code_dim, padding_idx=0)
            self.img_embedding = nn.Linear(config.code_dim, self.config.hidden_size, bias=True)
        else:
            self.img_embedding = nn.Linear(self.img_dim, self.config.hidden_size, bias=True)
            self.dropout = nn.Dropout(config.hidden_dropout_prob)
            if self.use_img_layernorm:
                self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.img_layer_norm_eps)

        self.apply(self.init_weights)

    def _resize_token_embeddings(self, new_num_tokens):
        old_embeddings = self.embeddings.word_embeddings
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
        self.embeddings.word_embeddings = new_embeddings
        return self.embeddings.word_embeddings

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(self, input_ids_a, token_type_ids_a=None, attention_mask_a=None, max_tag_length=None,
            position_ids_a=None, input_ids_b=None, token_type_ids_b=None, attention_mask_b=None,
            position_ids_b=None, head_mask=None, img_feats=None, encoder_history_states=None, encode_hn=False):
        if attention_mask_a is None:
            attention_mask_a = torch.ones_like(input_ids_a)

        if attention_mask_b is None:
            attention_mask_b = torch.ones_like(input_ids_b)

        if token_type_ids_a is None:
            token_type_ids_a = torch.zeros_like(input_ids_a)

        if token_type_ids_b is None:
            token_type_ids_b = torch.zeros_like(input_ids_b)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        if attention_mask_a.dim() == 2:
            extended_attention_mask_a = attention_mask_a.unsqueeze(1).unsqueeze(2)
        elif attention_mask_a.dim() == 3:
            extended_attention_mask_a = attention_mask_a.unsqueeze(1)
        else:
            raise NotImplementedError

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        # extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask_a = extended_attention_mask_a.to(dtype=self.dtype) # fp16 compatibility (editted here to adapt to pytorch>=1.5.0)
        extended_attention_mask_a = (1.0 - extended_attention_mask_a) * -10000.0

        if attention_mask_b.dim() == 2:
            extended_attention_mask_b = attention_mask_b.unsqueeze(1).unsqueeze(2)
        elif attention_mask_b.dim() == 3:
            extended_attention_mask_b = attention_mask_b.unsqueeze(1)
        else:
            raise NotImplementedError

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        # extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask_b = extended_attention_mask_b.to(dtype=self.dtype) # fp16 compatibility (editted here to adapt to pytorch>=1.5.0)
        extended_attention_mask_b = (1.0 - extended_attention_mask_b) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers/2, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
            # switch to float if needed + fp16 compatibility
            # head_mask = head_mask.to(dtype=next(self.parameters()).dtype) # switch to fload if need + fp16 compatibility
            head_mask = head_mask.to(dtype=self.dtype) # switch to fload if need + fp16 compatibility (edited here to adapt to pytorch>=1.5.0)
        else:
            head_mask = [None] * (self.config.num_hidden_layers//2)

        embedding_output_a = self.embeddings(input_ids_a, position_ids=position_ids_a,
                token_type_ids=token_type_ids_a)
        embedding_output_b = self.embeddings(input_ids_b, position_ids=position_ids_b,
                token_type_ids=token_type_ids_b)
        if encoder_history_states:
            assert img_feats is None, "Cannot take image features while using encoder history states"

        if img_feats is not None:
            if self.img_feature_type == 'dis_code':
                code_emb = self.code_embeddings(img_feats)
                img_embedding_output = self.img_embedding(code_emb)
            elif self.img_feature_type == 'dis_code_t': # transpose
                code_emb = self.code_embeddings(img_feats)
                code_emb = code_emb.permute(0, 2, 1)
                img_embedding_output = self.img_embedding(code_emb)
            elif self.img_feature_type == 'dis_code_scale': # left scaled
                code_emb = self.code_embeddings(img_feats)
                img_embedding_output = self.img_embedding(code_emb)
            else:
                img_embedding_output = self.img_embedding(img_feats)
                if self.use_img_layernorm:
                    img_embedding_output = self.LayerNorm(img_embedding_output)

                # add dropout on image embedding
                img_embedding_output = self.dropout(img_embedding_output)

            # concatenate two embeddings
            embedding_output_b = torch.cat((embedding_output_b, img_embedding_output), 1)

        # single-stream encoder
        txt_encoder_outputs = self.txt_encoder(embedding_output_a, extended_attention_mask_a,
                        head_mask=head_mask, encoder_history_states=encoder_history_states)[0][0]

        vis_encoder_outputs = self.vis_encoder(embedding_output_b, extended_attention_mask_b,
                        head_mask=head_mask, encoder_history_states=encoder_history_states)[0][0]
        
        only_vis_embeddings = vis_encoder_outputs[:, max_tag_length:, :]
        only_vis_mask = extended_attention_mask_b[:, :, :, max_tag_length:]

        # VSE-based retrieval loss similar to CLIP
        global_txt = F.normalize(txt_encoder_outputs[:, 0, :] @ self.txt_proj, p=2, dim=-1)
        global_img = F.normalize(vis_encoder_outputs[:, 0, :] @ self.vis_proj, p=2, dim=-1)
        sim_mat = global_txt @ global_img.t()

        if encode_hn:
            masked_sim_mat = sim_mat - 2 * torch.eye(sim_mat.shape[0], dtype=sim_mat.dtype, device=sim_mat.device)
            # hardest in-batch sample
            hard_img_index = torch.max(masked_sim_mat, dim=1)[1]
            hard_txt_index = torch.max(masked_sim_mat, dim=0)[1]
            # text-hard_img pair
            hard_img_seq = torch.cat([txt_encoder_outputs, 
                            torch.index_select(only_vis_embeddings, dim=0, index=hard_img_index)], dim=1)
            hard_img_mask = torch.cat([extended_attention_mask_a,
                            torch.index_select(only_vis_mask, dim=0, index=hard_img_index)], dim=-1)
            # hard_text-img pair
            hard_txt_seq = torch.cat([torch.index_select(txt_encoder_outputs, dim=0, index=hard_txt_index),
                                    only_vis_embeddings], dim=1)
            hard_txt_mask = torch.cat([torch.index_select(extended_attention_mask_a, dim=0, index=hard_txt_index),
                                    only_vis_mask], dim=-1)

            # select hard-img or hard-text with 0.5/0.5 probability
            n_sample = hard_img_seq.shape[0]
            dice_index = torch.randperm(n_sample, device=hard_img_seq.device)
            hard_seqs = torch.cat([torch.index_select(hard_img_seq, dim=0, index=dice_index[:(n_sample//2)]),
                            torch.index_select(hard_txt_seq, dim=0, index=dice_index[(n_sample//2):])], dim=0)
            hard_mask = torch.cat([torch.index_select(hard_img_mask, dim=0, index=dice_index[:(n_sample//2)]),
                            torch.index_select(hard_txt_mask, dim=0, index=dice_index[(n_sample//2):])], dim=0)

            # encoding for hard-negatives
            hard_encoder_outputs, hard_mid = self.mul_encoder(hard_seqs, hard_mask, head_mask=head_mask,
                            encoder_history_states=encoder_history_states)
            hard_encoder_outputs = hard_encoder_outputs[0]
            hard_pooled_output = self.pooler(hard_encoder_outputs)
        else:
            hard_encoder_outputs = None
            hard_pooled_output = None

        joint_bi_output = torch.cat([txt_encoder_outputs, only_vis_embeddings], dim=1)
        joint_bi_mask = torch.cat([extended_attention_mask_a, only_vis_mask], dim=-1)

        encoder_outputs, joint_mid = self.mul_encoder(joint_bi_output,
                joint_bi_mask, head_mask=head_mask,
                encoder_history_states=encoder_history_states)
        sequence_output = encoder_outputs[0]

        pooled_output = self.pooler(sequence_output)

        # add hidden_states and attentions if they are here
        outputs = (sequence_output, pooled_output,) + (hard_encoder_outputs, hard_pooled_output)
        single_stream_output = (txt_encoder_outputs, vis_encoder_outputs, sim_mat)
        return outputs, single_stream_output

    @property
    def dtype(self):
        return get_parameter_dtype(self)

    def forward_single(self, input_ids_a, token_type_ids_a=None, attention_mask_a=None, max_tag_length=None,
            position_ids_a=None, input_ids_b=None, token_type_ids_b=None, attention_mask_b=None,
            position_ids_b=None, head_mask=None, img_feats=None, encoder_history_states=None):
        if attention_mask_a is None:
            attention_mask_a = torch.ones_like(input_ids_a)

        if attention_mask_b is None:
            attention_mask_b = torch.ones_like(input_ids_b)

        if token_type_ids_a is None:
            token_type_ids_a = torch.zeros_like(input_ids_a)

        if token_type_ids_b is None:
            token_type_ids_b = torch.zeros_like(input_ids_b)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        if attention_mask_a.dim() == 2:
            extended_attention_mask_a = attention_mask_a.unsqueeze(1).unsqueeze(2)
        elif attention_mask_a.dim() == 3:
            extended_attention_mask_a = attention_mask_a.unsqueeze(1)
        else:
            raise NotImplementedError

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        # extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask_a = extended_attention_mask_a.to(dtype=self.dtype) # fp16 compatibility (editted here to adapt to pytorch>=1.5.0)
        extended_attention_mask_a = (1.0 - extended_attention_mask_a) * -10000.0

        if attention_mask_b.dim() == 2:
            extended_attention_mask_b = attention_mask_b.unsqueeze(1).unsqueeze(2)
        elif attention_mask_b.dim() == 3:
            extended_attention_mask_b = attention_mask_b.unsqueeze(1)
        else:
            raise NotImplementedError

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        # extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask_b = extended_attention_mask_b.to(dtype=self.dtype) # fp16 compatibility (editted here to adapt to pytorch>=1.5.0)
        extended_attention_mask_b = (1.0 - extended_attention_mask_b) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers/2, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
            # switch to float if needed + fp16 compatibility
            # head_mask = head_mask.to(dtype=next(self.parameters()).dtype) # switch to fload if need + fp16 compatibility
            head_mask = head_mask.to(dtype=self.dtype) # switch to fload if need + fp16 compatibility (edited here to adapt to pytorch>=1.5.0)
        else:
            head_mask = [None] * (self.config.num_hidden_layers//2)

        embedding_output_a = self.embeddings(input_ids_a, position_ids=position_ids_a,
                token_type_ids=token_type_ids_a)
        embedding_output_b = self.embeddings(input_ids_b, position_ids=position_ids_b,
                token_type_ids=token_type_ids_b)
        if encoder_history_states:
            assert img_feats is None, "Cannot take image features while using encoder history states"

        if img_feats is not None:
            if self.img_feature_type == 'dis_code':
                code_emb = self.code_embeddings(img_feats)
                img_embedding_output = self.img_embedding(code_emb)
            elif self.img_feature_type == 'dis_code_t': # transpose
                code_emb = self.code_embeddings(img_feats)
                code_emb = code_emb.permute(0, 2, 1)
                img_embedding_output = self.img_embedding(code_emb)
            elif self.img_feature_type == 'dis_code_scale': # left scaled
                code_emb = self.code_embeddings(img_feats)
                img_embedding_output = self.img_embedding(code_emb)
            else:
                img_embedding_output = self.img_embedding(img_feats)
                if self.use_img_layernorm:
                    img_embedding_output = self.LayerNorm(img_embedding_output)

                # add dropout on image embedding
                img_embedding_output = self.dropout(img_embedding_output)

            # concatenate two embeddings
            embedding_output_b = torch.cat((embedding_output_b, img_embedding_output), 1)

        # single-stream encoder
        txt_encoder_outputs = self.txt_encoder(embedding_output_a, extended_attention_mask_a,
                        head_mask=head_mask, encoder_history_states=encoder_history_states)[0]

        vis_encoder_outputs = self.vis_encoder(embedding_output_b, extended_attention_mask_b,
                        head_mask=head_mask, encoder_history_states=encoder_history_states)[0]
        
        # VSE-based retrieval loss similar to CLIP
        global_txt = F.normalize(txt_encoder_outputs[:, 0, :] @ self.txt_proj, p=2, dim=-1)
        global_img = F.normalize(vis_encoder_outputs[:, 0, :] @ self.vis_proj, p=2, dim=-1)

        # global_txt = F.normalize(txt_encoder_outputs[:, 0, :], p=2, dim=-1)
        # global_img = F.normalize(vis_encoder_outputs[:, 0, :], p=2, dim=-1)

        single_stream_output = (global_txt, global_img)
        return single_stream_output



def instance_bce_with_logits(logits, labels, reduction='mean', pos_weight=None):
    assert logits.dim() == 2
    loss = F.binary_cross_entropy_with_logits(logits, labels, reduction=reduction, pos_weight=pos_weight)
    if reduction == 'mean':
        loss *= labels.size(1)
    return loss


class ImageBertForSequenceClassification(BertPreTrainedModel):
    """
    Modified from BertForSequenceClassification to support oscar training.
    """
    def __init__(self, config):
        super(ImageBertForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.loss_type = config.loss_type
        self.config = config
        if config.img_feature_dim > 0:
            self.bert = BertImgModel(config)
        else:
            self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        if hasattr(config, 'classifier'):
            if not hasattr(config, 'cls_hidden_scale'): 
                config.cls_hidden_scale = 2

            if config.classifier == 'linear':
                self.classifier = nn.Linear(config.hidden_size,
                                            self.config.num_labels)
            elif config.classifier == 'mlp':
                self.classifier = nn.Sequential(
                    nn.Linear(config.hidden_size, config.hidden_size * config.cls_hidden_scale),
                    nn.ReLU(),
                    nn.Linear(config.hidden_size * config.cls_hidden_scale, self.config.num_labels)
                )
        else:
            self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)  # original
        self.apply(self.init_weights)

    def init_code_embedding(self, em):
        self.bert.code_embeddings.weight.data = em.clone()

    def reinit_cls_head(self):
        # make a re-initialization for the classifier
        self.classifier.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, soft_label=False,
            position_ids=None, head_mask=None, img_feats=None, concep_span=None, loss_weights=None):
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask, img_feats=img_feats)
        pooled_output = outputs[1]
        if concep_span is not None:
            if isinstance(concep_span, list):
                concept_emb = outputs[0][:, concep_span[0]:concep_span[1]]
            else:
                concept_emb = []
                for i in range(input_ids.shape[0]):
                    concept_emb.append(outputs[0][i, concep_span[i][0]:concep_span[i][1]])
                concept_emb = torch.cat([c.contiguous() for c in concept_emb], dim=0)

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        # print('in model batch size', logits.shape)
        if labels is not None:
            if self.num_labels == 1: #  doing regression
                loss_fct = MSELoss()
                labels = labels.to(torch.float)
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                if soft_label:
                    loss = soft_cross_entropy(labels, logits)
                elif self.loss_type == 'kl':
                    # KL Loss: https://github.com/uclanlp/visualbert/blob/master/pytorch_pretrained_bert/modeling.py
                    loss_fct = torch.nn.KLDivLoss(reduction="batchmean")
                    log_softmax = torch.nn.LogSoftmax(dim=-1)
                    reshaped_logits = logits.contiguous().view(-1, 3129)
                    reshaped_logits = log_softmax(reshaped_logits)
                    loss = loss_fct(reshaped_logits, labels.contiguous())
                elif self.loss_type == 'bce': # [VQA]
                    loss = instance_bce_with_logits(logits, labels, pos_weight=loss_weights)
                else: # cross_entropy [GQA, Retrieval, Captioning]
                    loss_fct = CrossEntropyLoss(weight=loss_weights)
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        if concep_span is not None:
            outputs += (concept_emb, )
        return outputs


class ImageBertForSequenceClassification_ML(BertPreTrainedModel):
    """
    Modified from ImageBertForSequenceClassification to support multi-label.
    """
    def __init__(self, config):
        super(ImageBertForSequenceClassification_ML, self).__init__(config)
        self.num_labels = config.num_labels
        self.loss_type = config.loss_type
        self.config = config
        if config.img_feature_dim > 0:
            self.bert = BertImgModel(config)
        else:
            self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        if hasattr(config, 'classifier'):
            if not hasattr(config, 'cls_hidden_scale'): 
                config.cls_hidden_scale = 2

            if config.classifier == 'linear':
                self.classifier = nn.Linear(config.hidden_size,
                                            self.config.num_labels)
            elif config.classifier == 'mlp':
                self.classifier = nn.Sequential(
                    nn.Linear(config.hidden_size, config.hidden_size * config.cls_hidden_scale),
                    nn.ReLU(),
                    nn.Linear(config.hidden_size * config.cls_hidden_scale, self.config.num_labels)
                )
        else:
            self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)  # original
        self.apply(self.init_weights)

    def init_code_embedding(self, em):
        self.bert.code_embeddings.weight.data = em.clone()

    def reinit_cls_head(self):
        # make a re-initialization for the classifier
        self.classifier.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, soft_label=False,
            position_ids=None, head_mask=None, img_feats=None, concep_span=None, loss_weights=None):
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask, img_feats=img_feats)
        pooled_output = outputs[1]
        if concep_span is not None:
            if isinstance(concep_span, list):
                concept_emb = outputs[0][:, concep_span[0]:concep_span[1]]
            else:
                concept_emb = []
                for i in range(input_ids.shape[0]):
                    concept_emb.append(outputs[0][i, concep_span[i][0]:concep_span[i][1]])
                concept_emb = torch.cat([c.contiguous() for c in concept_emb], dim=0)

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        # print('in model batch size', logits.shape)
        if labels is not None:
            if self.num_labels == 1: #  doing regression
                loss_fct = MSELoss()
                labels = labels.to(torch.float)
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                if soft_label:
                    loss = soft_cross_entropy(labels, logits)
                elif self.loss_type == 'kl':
                    # KL Loss: https://github.com/uclanlp/visualbert/blob/master/pytorch_pretrained_bert/modeling.py
                    loss_fct = torch.nn.KLDivLoss(reduction="batchmean")
                    log_softmax = torch.nn.LogSoftmax(dim=-1)
                    reshaped_logits = logits.contiguous().view(-1, 3129)
                    reshaped_logits = log_softmax(reshaped_logits)
                    loss = loss_fct(reshaped_logits, labels.contiguous())
                elif self.loss_type == 'bce': # [VQA]
                    loss = instance_bce_with_logits(logits, labels)
                else: # cross_entropy [GQA, Retrieval, Captioning]
                    loss_fct = CrossEntropyLoss(weight=loss_weights)
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        if concep_span is not None:
            outputs += (concept_emb, )
        return outputs



class ImageBertForSequenceClassification2(BertPreTrainedModel):
    """
    Modified from BertForSequenceClassification to support oscar training.
    """
    def __init__(self, config):
        super(ImageBertForSequenceClassification2, self).__init__(config)
        self.num_labels = config.num_labels
        self.loss_type = config.loss_type
        self.config = config
        if config.img_feature_dim > 0:
            self.bert = BertImgModel(config)
        else:
            self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        if hasattr(config, 'classifier'):
            if not hasattr(config, 'cls_hidden_scale'): 
                config.cls_hidden_scale = 2

            if config.classifier == 'linear':
                self.classifier = nn.Linear(config.hidden_size,
                                            self.config.num_labels)
            elif config.classifier == 'mlp':
                self.classifier = nn.Sequential(
                    nn.Linear(config.hidden_size, config.hidden_size * config.cls_hidden_scale),
                    nn.ReLU(),
                    nn.Linear(config.hidden_size * config.cls_hidden_scale, self.config.num_labels)
                )
        else:
            self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)  # original
        # self.con_classifier = nn.Sequential(
        #             nn.Linear(config.hidden_size, config.hidden_size * 2),
        #             nn.Dropout(config.hidden_dropout_prob),
        #             nn.ReLU(),
        #             nn.Linear(config.hidden_size * 2, 1)
        #         )
        self.con_classifier = nn.Linear(config.hidden_size, 1)
        self.apply(self.init_weights)

    def init_code_embedding(self, em):
        self.bert.code_embeddings.weight.data = em.clone()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, 
            position_ids=None, head_mask=None, img_feats=None, concep_span=None, con_pos=None, con_lab=None, con_lambda=0):
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask, img_feats=img_feats)
        pooled_output = outputs[1]

        if con_pos is not None:
            bs = outputs[2].shape[0]
            con_emb = outputs[2][:, con_pos, :][torch.arange(bs),torch.arange(bs),:]
            num_con_per_ins = con_pos.shape[1]
            con_emb = con_emb.reshape(bs*num_con_per_ins, -1)
            con_lab = con_lab.reshape(-1)
            con_logits = self.con_classifier(con_emb).squeeze()
            con_logits = torch.sigmoid(con_logits)
            con_loss_fn = nn.BCELoss()
            con_loss = con_loss_fn(con_logits, con_lab)
        else:
            con_loss = 0


        if concep_span is not None:
            if isinstance(concep_span, list):
                concept_emb = outputs[0][:, concep_span[0]:concep_span[1]]
            else:
                concept_emb = []
                for i in range(input_ids.shape[0]):
                    concept_emb.append(outputs[0][i, concep_span[i][0]:concep_span[i][1]])
                concept_emb = torch.cat([c.contiguous() for c in concept_emb], dim=0)

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        if con_lab is not None:
            outputs = (logits, con_logits,) + outputs[2:]
        else:
            outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        # print('in model batch size', logits.shape)
        if labels is not None:
            if self.num_labels == 1: #  doing regression
                loss_fct = MSELoss()
                labels = labels.to(torch.float)
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                if self.loss_type == 'kl':
                    # KL Loss: https://github.com/uclanlp/visualbert/blob/master/pytorch_pretrained_bert/modeling.py
                    loss_fct = torch.nn.KLDivLoss(reduction="batchmean")
                    log_softmax = torch.nn.LogSoftmax(dim=-1)
                    reshaped_logits = logits.contiguous().view(-1, 3129)
                    reshaped_logits = log_softmax(reshaped_logits)
                    loss = loss_fct(reshaped_logits, labels.contiguous())
                elif self.loss_type == 'bce': # [VQA]
                    loss = instance_bce_with_logits(logits, labels)
                else: # cross_entropy [GQA, Retrieval, Captioning]
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            loss += con_lambda * con_loss
            outputs = (loss,) + outputs
        if concep_span is not None:
            outputs += (concept_emb, )
        
        return outputs


class ImageBertForMultipleChoice(BertPreTrainedModel):
    """
    Modified from BertForMultipleChoice to support oscar training.
    """
    def __init__(self, config):
        super(ImageBertForMultipleChoice, self).__init__(config)
        self.loss_type = config.loss_type
        if config.img_feature_dim > 0:
            self.bert = BertImgModel(config) # ImageBERT
        else:
            self.bert = BertModel(config)  # original BERT

        if hasattr(config, 'use_img_layernorm'):
            self.use_img_layernorm = config.use_img_layernorm
        else:
            self.use_img_layernorm = None

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        if hasattr(config, 'classifier'):
            if not hasattr(config, 'cls_hidden_scale'): config.cls_hidden_scale = 2
            if config.classifier == 'linear':
                self.classifier = nn.Linear(config.num_choice*config.hidden_size, self.config.num_labels)
            elif config.classifier == 'mlp':
                if self.use_img_layernorm:
                    self.classifier = nn.Sequential(
                    nn.Linear(config.num_choice*config.hidden_size, config.hidden_size*config.cls_hidden_scale),
                    nn.ReLU(),
                    BertLayerNorm(config.hidden_size*config.cls_hidden_scale, eps=config.layer_norm_eps),
                    nn.Linear(config.hidden_size*config.cls_hidden_scale, self.config.num_labels)
                )
                else:
                    self.classifier = nn.Sequential(
                        nn.Linear(config.num_choice*config.hidden_size, config.hidden_size*config.cls_hidden_scale),
                        nn.ReLU(),
                        nn.Linear(config.hidden_size*config.cls_hidden_scale, self.config.num_labels)
                    )
        else:
            self.classifier = nn.Linear(config.num_choice*config.hidden_size, self.config.num_labels)  # original

        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None, img_feats=None):
        num_choices = input_ids.shape[1]

        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None

        flat_img_feats = img_feats.view(-1, img_feats.size(-2), img_feats.size(-1)) if img_feats is not None else None

        if isinstance(self.bert, BertImgModel):
            outputs = self.bert(flat_input_ids, position_ids=flat_position_ids, token_type_ids=flat_token_type_ids,
                            attention_mask=flat_attention_mask, head_mask=head_mask, img_feats=flat_img_feats)
        else:
            outputs = self.bert(flat_input_ids, position_ids=flat_position_ids, token_type_ids=flat_token_type_ids,
                                attention_mask=flat_attention_mask, head_mask=head_mask)
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)

        # reshaped_pool_output
        reshaped_pool_output = pooled_output.view(-1, self.config.num_choice*(pooled_output.shape[1]))
        logits = self.classifier(reshaped_pool_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.loss_type == 'bce':
                loss = instance_bce_with_logits(logits, labels.view(-1, self.config.num_labels))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits, labels)
            outputs = (loss,) + outputs
        return outputs

""" Oscar for Multiple Choice """
class OscarForMultipleChoice(BertPreTrainedModel):
    r"""
    Inputs:
        **input_ids**: ``torch.LongTensor`` of shape ``(batch_size, num_choices, sequence_length)``:
            Indices of input sequence tokens in the vocabulary.
            The second dimension of the input (`num_choices`) indicates the number of choices to score.
            To match pre-training, BERT input sequence should be formatted with [CLS] and [SEP] tokens as follows:

            (a) For sequence pairs:

                ``tokens:         [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]``

                ``token_type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1``

            (b) For single sequences:

                ``tokens:         [CLS] the dog is hairy . [SEP]``

                ``token_type_ids:   0   0   0   0  0     0   0``

            Indices can be obtained using :class:`pytorch_transformers.BertTokenizer`.
            See :func:`pytorch_transformers.PreTrainedTokenizer.encode` and
            :func:`pytorch_transformers.PreTrainedTokenizer.convert_tokens_to_ids` for details.
        **token_type_ids**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, num_choices, sequence_length)``:
            Segment token indices to indicate first and second portions of the inputs.
            The second dimension of the input (`num_choices`) indicates the number of choices to score.
            Indices are selected in ``[0, 1]``: ``0`` corresponds to a `sentence A` token, ``1``
            corresponds to a `sentence B` token
            (see `BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding`_ for more details).
        **attention_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(batch_size, num_choices, sequence_length)``:
            Mask to avoid performing attention on padding token indices.
            The second dimension of the input (`num_choices`) indicates the number of choices to score.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
        **head_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(num_heads,)`` or ``(num_layers, num_heads)``:
            Mask to nullify selected heads of the self-attention modules.
            Mask values selected in ``[0, 1]``:
            ``1`` indicates the head is **not masked**, ``0`` indicates the head is **masked**.
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the multiple choice classification loss.
            Indices should be in ``[0, ..., num_choices]`` where `num_choices` is the size of the second dimension
            of the input tensors. (see `input_ids` above)

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification loss.
        **classification_scores**: ``torch.FloatTensor`` of shape ``(batch_size, num_choices)`` where `num_choices` is the size of the second dimension
            of the input tensors. (see `input_ids` above).
            Classification scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        >>> config = BertConfig.from_pretrained('bert-base-uncased')
        >>> tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        >>>
        >>> model = BertForMultipleChoice(config)
        >>> choices = ["Hello, my dog is cute", "Hello, my cat is amazing"]
        >>> input_ids = torch.tensor([tokenizer.encode(s) for s in choices]).unsqueeze(0)  # Batch size 1, 2 choices
        >>> labels = torch.tensor(1).unsqueeze(0)  # Batch size 1
        >>> outputs = model(input_ids, labels=labels)
        >>> loss, classification_scores = outputs[:2]

    """

    def __init__(self, config):
        super(OscarForMultipleChoice, self).__init__(config)
        self.loss_type = config.loss_type

        if config.img_feature_dim > 0:
            self.bert = BertImgModel(config) # ImageBERT
        else:
            self.bert = BertModel(config)  # original BERT

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        if hasattr(config, 'classifier'):
            if not hasattr(config, 'cls_hidden_scale'): config.cls_hidden_scale = 2

            if config.classifier == 'linear':
                self.classifier = nn.Linear(config.hidden_size, 2) # original
                #self.classifier = weight_norm(nn.Linear(config.hidden_size, self.config.num_labels), dim=None)
            elif config.classifier == 'mlp':
                self.classifier = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size*config.cls_hidden_scale),
                                            nn.ReLU(),
                                            nn.Linear(config.hidden_size*config.cls_hidden_scale, 2)) # bce loss
        else:
            self.classifier = nn.Linear(config.hidden_size, config.num_labels)  # original

        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None, img_feats=None):
        num_choices = input_ids.shape[1]

        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None

        flat_img_feats = img_feats.view(-1, img_feats.size(-2), img_feats.size(-1)) if img_feats is not None else None

        if isinstance(self.bert, BertImgModel):
            outputs = self.bert(flat_input_ids, position_ids=flat_position_ids, token_type_ids=flat_token_type_ids,
                            attention_mask=flat_attention_mask, head_mask=head_mask, img_feats=flat_img_feats)
        else:
            outputs = self.bert(flat_input_ids, position_ids=flat_position_ids, token_type_ids=flat_token_type_ids,
                                attention_mask=flat_attention_mask, head_mask=head_mask)

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        #logger.info('pooled_output: {}, reshaped_pool_output: {}, logits: {}'.format(pooled_output.shape, reshaped_pool_output.shape, logits.shape))
        #logger.info('logits: {}, reshaped_logits: {}'.format(logits.shape, reshaped_logits.shape))
        #logger.info('labels: {}, labels.veiw: {}, labels.view(-1, num_labels): {}'.format(labels.shape, labels.view(-1).shape, labels.view(-1, self.config.num_labels).shape))
        if labels is not None:
            if self.loss_type == 'bce': #[batch_size, 2] v1
                #loss = instance_bce_with_logits(reshaped_logits, labels)
                loss = instance_bce_with_logits(logits, labels.view(-1, self.config.num_labels))
            elif self.loss_type == 'bxe':
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits, labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                #loss = loss_fct(reshaped_logits, labels)
                loss = loss_fct(logits, labels)
            outputs = (loss,) + outputs

        return outputs  # (loss), reshaped_logits, (hidden_states), (attentions)


class BertCaptioningLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.label_smoothing = getattr(config, 'label_smoothing', 0)
        self.drop_worst_ratio = getattr(config, 'drop_worst_ratio', 0)
        self.drop_worst_after = getattr(config, 'drop_worst_after', 0)
        self.log_soft = nn.LogSoftmax(dim=1)
        self.kl = nn.KLDivLoss(reduction='none')
        self.iter = 0

    def forward(self, logits, target):
        self.iter += 1
        eps = self.label_smoothing
        n_class = logits.size(1)
        one_hot = torch.zeros_like(logits).scatter(1, target.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = self.log_soft(logits)
        loss = self.kl(log_prb, one_hot).sum(1)

        if self.drop_worst_ratio > 0 and self.iter > self.drop_worst_after:
            loss, _ = torch.topk(loss,
                    k=int(loss.shape[0] * (1-self.drop_worst_ratio)),
                    largest=False)

        loss = loss.mean()

        return loss


class BertForImageCaptioning(CaptionPreTrainedModel):
    """
    Bert for Image Captioning.
    """
    def __init__(self, config):
        super(BertForImageCaptioning, self).__init__(config)
        self.config = config
        self.bert = BertImgModel(config)
        self.cls = BertOnlyMLMHead(config)
        self.loss = BertCaptioningLoss(config)

        self.apply(self.init_weights)
        self.tie_weights()

    def tie_weights(self):
        if hasattr(self.config, 'tie_weights') and self.config.tie_weights:
            self._tie_or_clone_weights(self.cls.predictions.decoder,
                                       self.bert.embeddings.word_embeddings)
        freeze = False
        if hasattr(self.config, 'freeze_embedding'):
            freeze = self.config.freeze_embedding
        self.bert.embeddings.word_embeddings.weight.requires_grad = not freeze

    def forward(self, *args, **kwargs):
        is_decode = kwargs.get('is_decode', False)
        if is_decode:
            return self.generate(*args, **kwargs)
        else:
            return self.encode_forward(*args, **kwargs)

    def encode_forward(self, input_ids, img_feats, attention_mask, masked_pos, masked_ids=None, 
            token_type_ids=None, position_ids=None, head_mask=None,
            is_training=True, encoder_history_states=None):
        outputs = self.bert(input_ids, img_feats=img_feats, attention_mask=attention_mask, 
                position_ids=position_ids, token_type_ids=token_type_ids,
                head_mask=head_mask,
                encoder_history_states=encoder_history_states)
        sequence_output = outputs[0][:, :masked_pos.shape[-1], :]

        if is_training:
            sequence_output = outputs[0][:, :masked_pos.shape[-1], :]
            # num_masks_in_batch * hidden_size
            sequence_output_masked = sequence_output[masked_pos==1, :]
            class_logits = self.cls(sequence_output_masked)
            masked_ids = masked_ids[masked_ids != 0]   # remove padding masks
            masked_loss = self.loss(class_logits.float(), masked_ids)
            outputs = (masked_loss, class_logits,) + outputs[2:]
        else:
            sequence_output = outputs[0][:, :input_ids.shape[-1], :]
            class_logits = self.cls(sequence_output)
            outputs = (class_logits,) + outputs[2:]
        return outputs


    def prepare_inputs_for_generation(self, curr_ids, past=None):
        # NOTE: if attention is on, it should be the token used to mask words in training
        mask_token_id = self.mask_token_id
        batch_size = curr_ids.shape[0]
        mask_ids = torch.full(
            (batch_size, 1), mask_token_id, dtype=torch.long, device=curr_ids.device
        )

        def _slice(t, start, end):
            if t is None:
                return t
            assert t.shape == (batch_size, self.max_seq_len + self.od_labels_len)
            return t[:, start: end]

        def _remove_elements(t, start, end):
            if t is None:
                return t
            assert t.shape == (batch_size, self.max_seq_len + self.od_labels_len)
            return torch.cat([t[:, :start], t[:, end:]], dim=1)

        if past is None:
            input_ids = torch.cat([curr_ids, mask_ids], dim=1)

            curr_len = input_ids.shape[1]
            full_len = self.max_seq_len + self.od_labels_len + self.img_seq_len
            assert self.full_attention_mask.shape == (batch_size,
                    full_len, full_len)

            def _remove_rows_cols(t, row_start, row_end, col_start, col_end):
                t00 = t[:, :row_start, :col_start]
                t01 = t[:, :row_start, col_end:]
                t10 = t[:, row_end:, :col_start]
                t11 = t[:, row_end:, col_end:]
                res = torch.cat([torch.cat([t00, t01], dim=2), torch.cat([t10, t11],
                            dim=2)], dim=1)
                assert res.shape == (t.shape[0], t.shape[1]-row_end+row_start,
                        t.shape[2]-col_end+col_start)
                return res

            seq_start = curr_len
            seq_end = self.max_seq_len
            attention_mask = _remove_rows_cols(self.full_attention_mask, seq_start,
                    seq_end, seq_start, seq_end)

            masked_pos = _remove_elements(self.full_masked_pos, seq_start, seq_end)
            token_type_ids = _remove_elements(self.full_token_type_ids, seq_start, seq_end)
            position_ids = _remove_elements(self.full_position_ids, seq_start, seq_end)
            img_feats = self.img_feats

            if self.add_od_labels:
                assert self.od_label_ids.shape[1] == self.od_labels_len
                input_ids = torch.cat([input_ids, self.od_label_ids], dim=1)
        else:
            last_token = curr_ids[:, -1:]
            # The representation of last token should be re-computed, because
            # it depends on both self-attention context and input tensor
            input_ids = torch.cat([last_token, mask_ids], dim=1)
            start_pos = curr_ids.shape[1] - 1
            end_pos = start_pos + input_ids.shape[1]
            masked_pos = _slice(self.full_masked_pos, start_pos, end_pos)
            token_type_ids = _slice(self.full_token_type_ids, start_pos, end_pos)
            position_ids = _slice(self.full_position_ids, start_pos, end_pos)

            img_feats = None
            assert past[0].shape[0] == batch_size
            if self.prev_encoded_layers is None:
                assert start_pos == 1  # the first token after BOS
                assert past[0].shape[1] == 2 + self.od_labels_len + self.img_seq_len
                # reorder to [od_labels, img_feats, sentence]
                self.prev_encoded_layers = [
                        torch.cat([x[:, 2:, :], x[:, :start_pos,:]], dim=1)
                        for x in past]
                s2s = self.full_attention_mask[:, :self.max_seq_len,
                        :self.max_seq_len]
                s2i = self.full_attention_mask[:, :self.max_seq_len,
                        self.max_seq_len:]
                i2s = self.full_attention_mask[:, self.max_seq_len:,
                        :self.max_seq_len]
                i2i = self.full_attention_mask[:, self.max_seq_len:,
                        self.max_seq_len:]
                self.full_attention_mask = torch.cat(
                        [torch.cat([i2i, i2s], dim=2),
                        torch.cat([s2i, s2s], dim=2)],
                        dim=1)
            else:
                assert start_pos > 1
                assert past[0].shape[1] == 2
                self.prev_encoded_layers = [torch.cat([x, p[:, :-1, :]], dim=1)
                        for x, p in zip(self.prev_encoded_layers, past)]

            attention_mask = self.full_attention_mask[:,
                self.od_labels_len+self.img_seq_len+start_pos: self.od_labels_len+self.img_seq_len+end_pos,
                :self.od_labels_len+self.img_seq_len+end_pos]

        return {'input_ids': input_ids, 'img_feats': img_feats,
            'masked_pos': masked_pos, 'attention_mask': attention_mask,
            'token_type_ids': token_type_ids, 'position_ids': position_ids,
            'is_training': False,
            'encoder_history_states': self.prev_encoded_layers}

    def get_output_embeddings(self):
        return self.decoder

    def generate(self, img_feats, attention_mask, masked_pos, token_type_ids=None,
            position_ids=None, head_mask=None, input_ids=None, max_length=None,
            do_sample=None, num_beams=None, temperature=None, top_k=None, top_p=None,
            repetition_penalty=None, bos_token_id=None, pad_token_id=None,
            eos_token_ids=None, mask_token_id=None, length_penalty=None,
            num_return_sequences=None,
            num_keep_best=1, is_decode=None,
            add_od_labels=False, od_labels_start_posid=None,
            use_cbs=False, fsm=None, num_constraints=None,
            min_constraints_to_satisfy=None, use_hypo=False,
            decoding_constraint_flag=None, bad_ending_ids=None,
            ):
        """ Generates captions given image features
        """
        assert is_decode
        batch_size = img_feats.shape[0]
        self.img_seq_len = img_feats.shape[1]
        self.max_seq_len = max_length
        self.mask_token_id = mask_token_id
        self.prev_encoded_layers = None
        # NOTE: num_keep_best is not equavilant to num_return_sequences
        # num_keep_best is the number of hypotheses to keep in beam search
        # num_return_sequences is the repeating times of input, coupled with
        # do_sample=True can generate more than one samples per image
        self.num_keep_best = num_keep_best

        vocab_size = self.config.vocab_size
        if not use_cbs:
            num_fsm_states = 1
        else:
            b, num_fsm_states, f1, v = fsm.shape
            assert b==batch_size and v==vocab_size and f1==num_fsm_states

        self.add_od_labels = add_od_labels
        # avoid position_ids collision of caption and od labels
        self.od_labels_start_posid = max(od_labels_start_posid, self.max_seq_len)
        if self.add_od_labels:
            # get od labels part from input_ids
            assert input_ids.shape[0] == batch_size
            od_label_ids = input_ids[:, self.max_seq_len:]
            self.od_labels_len = input_ids.shape[1] - self.max_seq_len
            input_ids = None
        else:
            self.od_labels_len = 0
            od_label_ids = None
            assert input_ids.shape == (batch_size, self.max_seq_len)
            input_ids = None

        if input_ids is None:
            input_ids = torch.full(
                (batch_size, 1), bos_token_id, dtype=torch.long, device=img_feats.device
            )
        else:
            assert input_ids.dim() == 2, "Input prompt should be of shape (batch_size, sequence length)."
            assert input_ids.shape[0] == batch_size, "Input batch size must match image features"

        cur_len = input_ids.shape[1]
        if  num_return_sequences != 1:
            # Expand input to num return sequences
            input_ids = self._expand_for_beams(input_ids, num_return_sequences)
            effective_batch_size = batch_size * num_return_sequences
        else:
            effective_batch_size = batch_size

        if position_ids is None:
            position_ids = torch.arange(self.max_seq_len, dtype=torch.long, device=input_ids.device)
            posids_len = self.max_seq_len
            if self.add_od_labels:
                od_labels_posids = torch.arange(
                        self.od_labels_start_posid,
                        self.od_labels_start_posid + self.od_labels_len, dtype=torch.long, device=input_ids.device)
                position_ids = torch.cat([position_ids, od_labels_posids])
                posids_len += self.od_labels_len
            position_ids = position_ids.unsqueeze(0).expand([batch_size, posids_len])

        num_expand = num_beams * num_fsm_states * num_return_sequences
        self.od_label_ids = self._expand_for_beams(od_label_ids, num_expand)
        self.img_feats = self._expand_for_beams(img_feats, num_expand)
        self.full_attention_mask = self._expand_for_beams(attention_mask, num_expand)
        self.full_masked_pos = self._expand_for_beams(masked_pos, num_expand)
        self.full_token_type_ids = self._expand_for_beams(token_type_ids, num_expand)
        self.full_position_ids = self._expand_for_beams(position_ids, num_expand)
        self.full_head_mask = self._expand_for_beams(head_mask, num_expand)

        if not use_cbs:
            if num_beams > 1:
                output = self._generate_beam_search(
                    input_ids,
                    cur_len,
                    max_length,
                    do_sample,
                    temperature,
                    top_k,
                    top_p,
                    repetition_penalty,
                    pad_token_id,
                    eos_token_ids,
                    effective_batch_size,
                    length_penalty,
                    num_beams,
                    vocab_size,
                )
            else:
                output = self._generate_no_beam_search(
                    input_ids,
                    cur_len,
                    max_length,
                    do_sample,
                    temperature,
                    top_k,
                    top_p,
                    repetition_penalty,
                    pad_token_id,
                    eos_token_ids,
                    effective_batch_size,
                )
        else:
            assert self.num_keep_best == 1, 'not supported n_best > 1 for CBS'
            searcher = ConstrainedBeamSearch(eos_token_ids, max_length,
                    num_beams)
            curr_ids, sum_logprobs = searcher.search(
                    input_ids,
                    None,
                    self._decode_step,
                    fsm,
            )
            curr_ids, logprobs = select_best_beam_with_constraints(
                curr_ids,
                sum_logprobs,
                num_constraints,
                min_constraints_to_satisfy,
                eos_token_ids,
            )
            # (batch_size, n_best, max_len), (batch_size, n_best)
            output = (curr_ids.unsqueeze(1), logprobs.unsqueeze(1))

        return output

    def _expand_for_beams(self, x, num_expand):
        if x is None or num_expand == 1:
            return x

        input_shape = list(x.shape)
        expanded_shape = input_shape[:1] + [num_expand] + input_shape[1:]
        x = x.unsqueeze(1).expand(expanded_shape)
        # (batch_size * num_expand, ...)
        x = x.contiguous().view([input_shape[0] * num_expand] + input_shape[1:])
        return x

    def _do_output_past(self, outputs):
        return len(outputs) > 1


class BertPreTrainingHeads(nn.Module):
    def __init__(self, config, only_vocab=False):
        super(BertPreTrainingHeads, self).__init__()
        self.predictions = BertLMPredictionHead(config, only_vocab=only_vocab)
        num_seq_relations = config.num_contrast_classes if hasattr(config, "num_contrast_classes") else 2
        self.seq_relationship = nn.Linear(config.hidden_size, num_seq_relations)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class VLBertPreTrainingHeads(nn.Module):
    def __init__(self, config, img_emb_weight):
        super(VLBertPreTrainingHeads, self).__init__()
        self.predictions = BertLMPredictionHead(config)
        num_seq_relations = config.num_contrast_classes if hasattr(config, "num_contrast_classes") else 2
        self.seq_relationship = nn.Linear(config.hidden_size, num_seq_relations)
        self.MRF_predictor = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                                           GELU(),
                                           BertLayerNorm(config.hidden_size, eps=config.img_layer_norm_eps))
        self.MRF_bias = nn.Parameter(torch.zeros(config.img_feature_dim))
        self.MRF_weight = img_emb_weight

        self.MRC_predictor = nn.Linear(config.hidden_size, config.od_tag_size)

    def forward(self, sequence_output, pooled_output, txt_length=None):
        if txt_length is not None:
            prediction_scores = self.predictions(sequence_output[:, :txt_length, :])
        else:
            prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        if txt_length is not None:
            mrf_hidden = self.MRF_predictor(sequence_output[:, txt_length:, :])
            mrf_feature = F.linear(mrf_hidden, self.MRF_weight.t(), self.MRF_bias)
            mrc_score = self.MRC_predictor(sequence_output[:, txt_length:, :])
        else:
            mrf_hidden = self.MRF_predictor(sequence_output)
            mrf_feature = F.linear(mrf_hidden, self.MRF_weight.t(), self.MRF_bias)
            mrc_score = self.MRC_predictor(sequence_output)
        return prediction_scores, seq_relationship_score, mrf_feature, mrc_score


class BertImgForPreTraining(ImgPreTrainedModel):
    r"""
        **masked_lm_labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Labels for computing the masked language modeling loss.
            Indices should be in ``[-1, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
            Tokens with indices set to ``-1`` are ignored (masked), the loss is only computed for the tokens with labels
            in ``[0, ..., config.vocab_size]``
        **next_sentence_label**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the next sequence prediction (classification) loss. Input should be a sequence pair (see ``input_ids`` docstring)
            Indices should be in ``[0, 1]``.
            ``0`` indicates sequence B is a continuation of sequence A,
            ``1`` indicates sequence B is a random sequence.

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when both ``masked_lm_labels`` and ``next_sentence_label`` are provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Total loss as the sum of the masked language modeling loss and the next sequence prediction (classification) loss.
        **prediction_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, config.vocab_size)``
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        **seq_relationship_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, 2)``
            Prediction scores of the next sequence prediction (classification) head (scores of True/False continuation before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        >>> config = BertConfig.from_pretrained('bert-base-uncased')
        >>> tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        >>>
        >>> model = BertImgForPreTraining(config)
        >>> input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        >>> outputs = model(input_ids)
        >>> prediction_scores, seq_relationship_scores = outputs[:2]

    """
    config_class = BertConfig
    pretrained_model_archive_map = BERT_PRETRAINED_MODEL_ARCHIVE_MAP
    load_tf_weights = load_tf_weights_in_bert
    base_model_prefix = "bert"

    def __init__(self, config):
        super(BertImgForPreTraining, self).__init__(config)

        #self.bert = BertModel(config) # original BERT
        self.bert = BertImgModel(config)
        self.cls = BertPreTrainingHeads(config)
        self.num_seq_relations = config.num_contrast_classes if hasattr(config, "num_contrast_classes") else 2
        self.max_text_seq_length = config.max_text_seq_length if hasattr(config, "max_text_seq_length") else None
        # self.max_text_seq_length = None

        self.apply(self.init_weights)
        self.tie_weights()

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0,
                                       std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(self.cls.predictions.decoder,
                                   self.bert.embeddings.word_embeddings)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None,
            next_sentence_label=None, position_ids=None, head_mask=None, img_feats=None):
        time_start = time.time()
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask, img_feats=img_feats)

        sequence_length = input_ids.shape[1]
        img_length = img_feats.shape[1]
        time_1 = time.time()
        sequence_output, pooled_output = outputs[:2]
        if self.max_text_seq_length is not None:
            prediction_scores, seq_relationship_score = self.cls(sequence_output[:, :self.max_text_seq_length,:], pooled_output)
        else:
            prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)
        outputs = (prediction_scores, seq_relationship_score,) + outputs[2:]  # add hidden states and attention if they are here
        time_2 = time.time()
        if masked_lm_labels is not None and next_sentence_label is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            if self.max_text_seq_length is not None:
                # print('before:', masked_lm_labels.shape)
                masked_lm_labels = masked_lm_labels[:, :self.max_text_seq_length].contiguous()
                # print('after:', masked_lm_labels.shape)
                # print('prediction:', prediction_scores.shape)
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.reshape(-1))
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, self.num_seq_relations), next_sentence_label.view(-1))
            total_loss = masked_lm_loss + next_sentence_loss
            outputs = (total_loss,) + outputs + (masked_lm_loss,)
        time_3 = time.time()
        return outputs  # (loss), prediction_scores, seq_relationship_score, (hidden_states), (attentions)


class ImageBertForSequenceClassification_F(BertPreTrainedModel):
    """
    Modified from BertForSequenceClassification to support oscar training
    for background information fusion
    """
    def __init__(self, config):
        super(ImageBertForSequenceClassification_F, self).__init__(config)
        self.num_labels = config.num_labels
        self.loss_type = config.loss_type
        self.config = config
        if config.img_feature_dim > 0:
            self.bert = BertImgModel(config)
        else:
            self.bert = BertModel(config)

        self.fusion_layer = nn.ModuleList([CaptionBertLayer(config) for _ in range(config.num_fusion_layers)])
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        if hasattr(config, 'classifier'):
            if not hasattr(config, 'cls_hidden_scale'): 
                config.cls_hidden_scale = 2

            if config.classifier == 'linear':
                self.classifier = nn.Linear(config.hidden_size,
                                            self.config.num_labels)
            elif config.classifier == 'mlp':
                self.classifier = nn.Sequential(
                    nn.Linear(config.hidden_size, config.hidden_size * config.cls_hidden_scale),
                    nn.ReLU(),
                    nn.Linear(config.hidden_size * config.cls_hidden_scale, self.config.num_labels)
                )
        else:
            self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)  # original
        self.apply(self.init_weights)

    def init_code_embedding(self, em):
        self.bert.code_embeddings.weight.data = em.clone()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, 
            position_ids=None, head_mask=None, img_feats=None, bg_input_ids=None, bg_token_type_ids=None, 
            bg_attention_mask=None, bg_position_ids=None, bg_head_mask=None, bg_img_feats=None):
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask, img_feats=img_feats)
        pooled_output = outputs[1]
        background = self.bert(bg_input_ids, position_ids=bg_position_ids, token_type_ids=bg_token_type_ids,
                                attention_mask=bg_attention_mask, head_mask=bg_head_mask, img_feats=bg_img_feats)
        fused_input = [outputs]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        # print('in model batch size', logits.shape)
        if labels is not None:
            if self.num_labels == 1: #  doing regression
                loss_fct = MSELoss()
                labels = labels.to(torch.float)
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                if self.loss_type == 'kl':
                    # KL Loss: https://github.com/uclanlp/visualbert/blob/master/pytorch_pretrained_bert/modeling.py
                    loss_fct = torch.nn.KLDivLoss(reduction="batchmean")
                    log_softmax = torch.nn.LogSoftmax(dim=-1)
                    reshaped_logits = logits.contiguous().view(-1, 3129)
                    reshaped_logits = log_softmax(reshaped_logits)
                    loss = loss_fct(reshaped_logits, labels.contiguous())
                elif self.loss_type == 'bce': # [VQA]
                    loss = instance_bce_with_logits(logits, labels)
                else: # cross_entropy [GQA, Retrieval, Captioning]
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        if concep_span is not None:
            outputs += (concept_emb, )
        return outputs



class VLBertImgForPreTraining(ImgPreTrainedModel):
    r"""
        **masked_lm_labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Labels for computing the masked language modeling loss.
            Indices should be in ``[-1, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
            Tokens with indices set to ``-1`` are ignored (masked), the loss is only computed for the tokens with labels
            in ``[0, ..., config.vocab_size]``
        **next_sentence_label**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the next sequence prediction (classification) loss. Input should be a sequence pair (see ``input_ids`` docstring)
            Indices should be in ``[0, 1]``.
            ``0`` indicates sequence B is a continuation of sequence A,
            ``1`` indicates sequence B is a random sequence.

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when both ``masked_lm_labels`` and ``next_sentence_label`` are provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Total loss as the sum of the masked language modeling loss and the next sequence prediction (classification) loss.
        **prediction_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, config.vocab_size)``
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        **seq_relationship_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, 2)``
            Prediction scores of the next sequence prediction (classification) head (scores of True/False continuation before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        >>> config = BertConfig.from_pretrained('bert-base-uncased')
        >>> tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        >>>
        >>> model = BertImgForPreTraining(config)
        >>> input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        >>> outputs = model(input_ids)
        >>> prediction_scores, seq_relationship_scores = outputs[:2]

    """
    config_class = BertConfig
    pretrained_model_archive_map = BERT_PRETRAINED_MODEL_ARCHIVE_MAP
    load_tf_weights = load_tf_weights_in_bert
    base_model_prefix = "bert"

    def __init__(self, config):
        super(VLBertImgForPreTraining, self).__init__(config)

        #self.bert = BertModel(config) # original BERT
        self.bert = BertImgModel(config)
        self.cls = VLBertPreTrainingHeads(config, self.bert.img_embedding.weight)
        self.num_seq_relations = config.num_contrast_classes if hasattr(config, "num_contrast_classes") else 2
        self.max_text_seq_length = config.max_text_seq_length if hasattr(config, "max_text_seq_length") else None

        self.apply(self.init_weights)
        self.tie_weights()

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0,
                                       std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(self.cls.predictions.decoder,
                                   self.bert.embeddings.word_embeddings)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None,
            next_sentence_label=None, position_ids=None, head_mask=None, img_feats=None, 
            masked_region_labels=None, masked_target_feature=None, masked_region_id=None):
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask, img_feats=img_feats)

        sequence_output, pooled_output = outputs[:2]
        if self.max_text_seq_length is not None:
            prediction_scores, seq_relationship_score, mrf_feat, mrc_score = self.cls(sequence_output, pooled_output, self.max_text_seq_length)
        else:
            prediction_scores, seq_relationship_score, mrf_feat, mrc_score = self.cls(sequence_output, pooled_output)

        outputs = (prediction_scores, seq_relationship_score,) + outputs[2:]  # add hidden states and attention if they are here

        if masked_lm_labels is not None and next_sentence_label is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            if self.max_text_seq_length is not None:
                masked_lm_labels = masked_lm_labels[:, :self.max_text_seq_length].contiguous()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, self.num_seq_relations), next_sentence_label.view(-1))
            total_loss = masked_lm_loss + next_sentence_loss
            outputs = outputs + (masked_lm_loss,)
        else:
            total_loss = 0

        if masked_region_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            masked_mrc_loss = loss_fct(mrc_score.view(-1, mrc_score.shape[-1]), masked_region_labels.view(-1))
            mrf_feat = torch.masked_select(mrf_feat, masked_region_id.unsqueeze(-1).bool()).view(-1, mrf_feat.shape[-1])
            target_feat = torch.masked_select(masked_target_feature, masked_region_id.unsqueeze(-1).bool()).view(-1, mrf_feat.shape[-1])
            # masked_mrf_loss = torch.mean(torch.norm(mrf_feat-target_feat, p=2, dim=-1)**2)
            masked_mrf_loss = F.mse_loss(mrf_feat, target_feat)
            total_loss = total_loss + masked_mrc_loss+masked_mrf_loss
            outputs = outputs + (masked_mrf_loss, masked_mrc_loss)

        outputs = (total_loss,) + outputs
        return outputs  # (loss), prediction_scores, seq_relationship_score, (hidden_states), (attentions)



class BertImgForPreTraining2(ImgPreTrainedModel): # quick version
    r"""
        **masked_lm_labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Labels for computing the masked language modeling loss.
            Indices should be in ``[-1, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
            Tokens with indices set to ``-1`` are ignored (masked), the loss is only computed for the tokens with labels
            in ``[0, ..., config.vocab_size]``
        **next_sentence_label**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the next sequence prediction (classification) loss. Input should be a sequence pair (see ``input_ids`` docstring)
            Indices should be in ``[0, 1]``.
            ``0`` indicates sequence B is a continuation of sequence A,
            ``1`` indicates sequence B is a random sequence.

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when both ``masked_lm_labels`` and ``next_sentence_label`` are provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Total loss as the sum of the masked language modeling loss and the next sequence prediction (classification) loss.
        **prediction_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, config.vocab_size)``
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        **seq_relationship_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, 2)``
            Prediction scores of the next sequence prediction (classification) head (scores of True/False continuation before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        >>> config = BertConfig.from_pretrained('bert-base-uncased')
        >>> tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        >>>
        >>> model = BertImgForPreTraining(config)
        >>> input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        >>> outputs = model(input_ids)
        >>> prediction_scores, seq_relationship_scores = outputs[:2]

    """
    config_class = BertConfig
    pretrained_model_archive_map = BERT_PRETRAINED_MODEL_ARCHIVE_MAP
    load_tf_weights = load_tf_weights_in_bert
    base_model_prefix = "bert"

    def __init__(self, config):
        super(BertImgForPreTraining2, self).__init__(config)

        #self.bert = BertModel(config) # original BERT
        self.bert = BertImgModel(config)
        self.cls = BertPreTrainingHeads(config)
        self.num_seq_relations = config.num_contrast_classes if hasattr(config, "num_contrast_classes") else 2
        self.max_text_seq_length = config.max_text_seq_length if hasattr(config, "max_text_seq_length") else None
        # self.max_text_seq_length = None

        self.apply(self.init_weights)
        self.tie_weights()
        self.last_time = None

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0,
                                       std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(self.cls.predictions.decoder,
                                   self.bert.embeddings.word_embeddings)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None,
            next_sentence_label=None, position_ids=None, head_mask=None, img_feats=None):
        time_start = time.time()

        sequence_length = input_ids.shape[1]
        img_length = img_feats.shape[1]
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask, img_feats=img_feats)
        time_1 = time.time()
        sequence_output, pooled_output = outputs[:2]
        lm_mask = masked_lm_labels > -1
        masked_sequence_output = torch.masked_select(sequence_output, lm_mask.unsqueeze(-1)).reshape(-1, self.config.hidden_size)
        prediction_scores, seq_relationship_score = self.cls(masked_sequence_output, pooled_output)
        outputs = (prediction_scores, seq_relationship_score,) + outputs[2:]  # add hidden states and attention if they are here
        time_2 = time.time()

        if masked_lm_labels is not None and next_sentence_label is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            masked_lm_labels = torch.masked_select(masked_lm_labels, lm_mask).reshape(-1)
            masked_lm_loss = loss_fct(prediction_scores, masked_lm_labels.reshape(-1))
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, self.num_seq_relations), next_sentence_label.view(-1))
            total_loss = masked_lm_loss + next_sentence_loss
            outputs = (total_loss,) + outputs + (masked_lm_loss,)
        time_3 = time.time()
        if self.last_time is not None:
            data_time = time_start - self.last_time
        else:
            data_time = 0
        self.last_time = time_3
        # print('Token sequence length:{}, Image sequence length {}'.format(sequence_length, img_length), 'Bert encoding time:', time_1-time_start, 'Head time:', time_2-time_1, 'Loss time:', time_3-time_2, 'data time', data_time)
        return outputs  # (loss), prediction_scores, seq_relationship_score, (hidden_states), (attentions)



class ImageBertForSequenceClassification_MLM(BertPreTrainedModel):
    """
    Modified from BertForSequenceClassification to support oscar training.
    adding possible MLM loss
    """
    def __init__(self, config):
        super(ImageBertForSequenceClassification_MLM, self).__init__(config)
        self.num_labels = config.num_labels
        self.loss_type = config.loss_type
        self.config = config
        if config.img_feature_dim > 0:
            self.bert = BertImgModel(config)
        else:
            self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.mlm_head = BertLMPredictionHead(config)
        self.mlm_weight = config.mlm_weight

        if hasattr(config, 'classifier'):
            if not hasattr(config, 'cls_hidden_scale'): 
                config.cls_hidden_scale = 2

            if config.classifier == 'linear':
                self.classifier = nn.Linear(config.hidden_size,
                                            self.config.num_labels)
            elif config.classifier == 'mlp':
                self.classifier = nn.Sequential(
                    nn.Linear(config.hidden_size, config.hidden_size * config.cls_hidden_scale),
                    nn.ReLU(),
                    nn.Linear(config.hidden_size * config.cls_hidden_scale, self.config.num_labels)
                )
        else:
            self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)  # original
        self.apply(self.init_weights)
        self.tie_weights()

    def init_code_embedding(self, em):
        self.bert.code_embeddings.weight.data = em.clone()

    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(self.mlm_head.decoder,
                                   self.bert.embeddings.word_embeddings)

    def reinit_cls_head(self):
        # make a re-initialization for the classifier
        self.classifier.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, masked_lm_labels=None,
            position_ids=None, head_mask=None, img_feats=None, concep_span=None, loss_weights=None):
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask, img_feats=img_feats)
        pooled_output = outputs[1]
        sequence_output = outputs[0]
        if concep_span is not None:
            if isinstance(concep_span, list):
                concept_emb = outputs[0][:, concep_span[0]:concep_span[1]]
            else:
                concept_emb = []
                for i in range(input_ids.shape[0]):
                    concept_emb.append(outputs[0][i, concep_span[i][0]:concep_span[i][1]])
                concept_emb = torch.cat([c.contiguous() for c in concept_emb], dim=0)

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        # print('in model batch size', logits.shape)
        if labels is not None:
            if self.num_labels == 1: #  doing regression
                loss_fct = MSELoss()
                labels = labels.to(torch.float)
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                if self.loss_type == 'kl':
                    # KL Loss: https://github.com/uclanlp/visualbert/blob/master/pytorch_pretrained_bert/modeling.py
                    loss_fct = torch.nn.KLDivLoss(reduction="batchmean")
                    log_softmax = torch.nn.LogSoftmax(dim=-1)
                    reshaped_logits = logits.contiguous().view(-1, 3129)
                    reshaped_logits = log_softmax(reshaped_logits)
                    loss = loss_fct(reshaped_logits, labels.contiguous())
                elif self.loss_type == 'bce': # [VQA]
                    loss = instance_bce_with_logits(logits, labels)
                else: # cross_entropy [GQA, Retrieval, Captioning]
                    loss_fct = CrossEntropyLoss(weight=loss_weights)
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            # outputs = (loss,) + outputs
        if concep_span is not None:
            outputs += (concept_emb, )

        if masked_lm_labels is not None:
            mlm_lfn = CrossEntropyLoss(ignore_index=-1)
            lm_mask = masked_lm_labels > -1
            masked_sequence_output = torch.masked_select(sequence_output, lm_mask.unsqueeze(-1)).reshape(-1, self.config.hidden_size)
            prediction_scores = self.mlm_head(masked_sequence_output)
            masked_lm_labels = torch.masked_select(masked_lm_labels, lm_mask).reshape(-1)
            mlm_loss = mlm_lfn(prediction_scores, masked_lm_labels)
            outputs = (loss+self.mlm_weight*mlm_loss, ) + outputs
        else:
            outputs = (loss,) + outputs


        return outputs




class ImageBertForSequenceClassificationR(BertPreTrainedModel):
    """
    Modified from BertForSequenceClassification to support oscar training.
    """
    def __init__(self, config):
        super(ImageBertForSequenceClassificationR, self).__init__(config)
        self.num_labels = config.num_labels
        self.loss_type = config.loss_type
        self.config = config
        if config.img_feature_dim > 0:
            self.bert = BertImgModel(config)
        else:
            self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.r_lambda = self.config.r_lambda
        self.reason_penalty = self.config.reason_penalty

        if hasattr(config, 'classifier'):
            if not hasattr(config, 'cls_hidden_scale'): 
                config.cls_hidden_scale = 2

            if config.classifier == 'linear':
                self.classifier = nn.Linear(config.hidden_size,
                                            self.config.num_labels)
                self.r_cls = nn.Linear(config.hidden_size,
                                            self.config.num_reasons)
            elif config.classifier == 'mlp':
                self.classifier = nn.Sequential(
                    nn.Linear(config.hidden_size, config.hidden_size * config.cls_hidden_scale),
                    nn.ReLU(),
                    nn.Linear(config.hidden_size * config.cls_hidden_scale, self.config.num_labels)
                )
                self.r_cls = nn.Sequential(
                    nn.Linear(config.hidden_size, config.hidden_size * config.cls_hidden_scale),
                    nn.ReLU(),
                    nn.Linear(config.hidden_size * config.cls_hidden_scale, self.config.num_reasons)
                )
        else:
            self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)  # original
            self.r_cls = nn.Linear(config.hidden_size, self.config.num_reasons)
        self.apply(self.init_weights)

    def init_code_embedding(self, em):
        self.bert.code_embeddings.weight.data = em.clone()

    def reinit_cls_head(self):
        # make a re-initialization for the classifier
        self.classifier.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, soft_label=False, r_lamba=None,
            position_ids=None, head_mask=None, img_feats=None, concep_span=None, loss_weights=None, r_labels=None, r_weights=None):
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask, img_feats=img_feats)
        pooled_output = outputs[1]
        if concep_span is not None:
            if isinstance(concep_span, list):
                concept_emb = outputs[0][:, concep_span[0]:concep_span[1]]
            else:
                concept_emb = []
                for i in range(input_ids.shape[0]):
                    concept_emb.append(outputs[0][i, concep_span[i][0]:concep_span[i][1]])
                concept_emb = torch.cat([c.contiguous() for c in concept_emb], dim=0)

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        r_logits = self.r_cls(pooled_output)

        outputs = (logits, r_logits) + outputs[2:]  # add hidden states and attention if they are here
        # print('in model batch size', logits.shape)
        if labels is not None:
            if self.num_labels == 1: #  doing regression
                loss_fct = MSELoss()
                labels = labels.to(torch.float)
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                if soft_label:
                    loss = soft_cross_entropy(labels, logits)
                elif self.loss_type == 'kl':
                    # KL Loss: https://github.com/uclanlp/visualbert/blob/master/pytorch_pretrained_bert/modeling.py
                    loss_fct = torch.nn.KLDivLoss(reduction="batchmean")
                    log_softmax = torch.nn.LogSoftmax(dim=-1)
                    reshaped_logits = logits.contiguous().view(-1, 3129)
                    reshaped_logits = log_softmax(reshaped_logits)
                    loss = loss_fct(reshaped_logits, labels.contiguous())
                elif self.loss_type == 'bce': # [VQA]
                    loss = instance_bce_with_logits(logits, labels, pos_weight=loss_weights)
                else: # cross_entropy [GQA, Retrieval, Captioning]
                    loss_fct = CrossEntropyLoss(weight=loss_weights)
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                if r_labels is not None:
                    loss += self.r_lambda * instance_bce_with_logits(r_logits, r_labels, pos_weight=r_weights)
                    if self.reason_penalty:
                        pos_prob = F.softmax(logits)[:, 1]
                        r_prob = torch.sigmoid(r_logits)
                        loss += torch.mean(pos_prob * r_prob.max(dim=1)[0])
            outputs = (loss,) + outputs
        if concep_span is not None:
            outputs += (concept_emb, )
        return outputs


class BiBertImgForPreTraining(ImgPreTrainedModel): # a version with weakly-supervised grounding
    r"""
        **masked_lm_labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Labels for computing the masked language modeling loss.
            Indices should be in ``[-1, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
            Tokens with indices set to ``-1`` are ignored (masked), the loss is only computed for the tokens with labels
            in ``[0, ..., config.vocab_size]``
        **next_sentence_label**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the next sequence prediction (classification) loss. Input should be a sequence pair (see ``input_ids`` docstring)
            Indices should be in ``[0, 1]``.
            ``0`` indicates sequence B is a continuation of sequence A,
            ``1`` indicates sequence B is a random sequence.

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when both ``masked_lm_labels`` and ``next_sentence_label`` are provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Total loss as the sum of the masked language modeling loss and the next sequence prediction (classification) loss.
        **prediction_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, config.vocab_size)``
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        **seq_relationship_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, 2)``
            Prediction scores of the next sequence prediction (classification) head (scores of True/False continuation before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        >>> config = BertConfig.from_pretrained('bert-base-uncased')
        >>> tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        >>>
        >>> model = BertImgForPreTraining(config)
        >>> input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        >>> outputs = model(input_ids)
        >>> prediction_scores, seq_relationship_scores = outputs[:2]

    """
    config_class = BertConfig
    pretrained_model_archive_map = BERT_PRETRAINED_MODEL_ARCHIVE_MAP
    load_tf_weights = load_tf_weights_in_bert
    base_model_prefix = "bert"

    def __init__(self, config):
        super(BiBertImgForPreTraining, self).__init__(config)

        #self.bert = BertModel(config) # original BERT
        # tmp = BiBertImgModel2
        self.bert = BiBertImgModel(config)
        self.cls = BertPreTrainingHeads(config, only_vocab=True)
        self.half_mlm = BertLMPredictionHead(config, only_vocab=True)
        self.only_vocab_size = config.only_word_size
        self.num_seq_relations = config.num_contrast_classes if hasattr(config, "num_contrast_classes") else 2
        self.max_text_seq_length = config.max_text_seq_length if hasattr(config, "max_text_seq_length") else None
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        # self.max_text_seq_length = None

        self.apply(self.init_weights)
        self.tie_weights()

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0,
                                       std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(self.cls.predictions.decoder,
                                   self.bert.embeddings.word_embeddings, only_vocab=True, only_word_size=self.only_vocab_size)

        self._tie_or_clone_weights(self.half_mlm.decoder, self.bert.embeddings.word_embeddings,
                                   only_vocab=True, only_word_size=self.only_vocab_size)

    def forward(self, input_ids_a, token_type_ids_a=None, attention_mask_a=None, masked_lm_labels_a=None,
            input_ids_b=None, token_type_ids_b=None, attention_mask_b=None, masked_lm_labels_b=None, max_tag_length=20,
            position_ids_a=None, position_ids_b=None, head_mask=None, img_feats=None,is_img_match=None, img_index=None, phrase_index=None):
        outputs, single_stream_output = self.bert(input_ids_a=input_ids_a, position_ids_a=position_ids_a, token_type_ids_a=token_type_ids_a,
                            attention_mask_a=attention_mask_a, head_mask=head_mask, img_feats=img_feats,
                            input_ids_b=input_ids_b, position_ids_b=position_ids_b, token_type_ids_b=token_type_ids_b,
                            attention_mask_b=attention_mask_b, max_tag_length=max_tag_length, encode_hn=True)

        txt_encoder_outputs, vis_encoder_outputs, sim_mat = single_stream_output

        ce_loss = CrossEntropyLoss(ignore_index=-1)

        # visual encoder MLM
        vis_lm_mask = masked_lm_labels_b > -1
        vis_masked_seq_output = torch.masked_select(vis_encoder_outputs, vis_lm_mask.unsqueeze(-1)).reshape(-1, self.config.hidden_size)
        vis_lm_score = self.half_mlm(vis_masked_seq_output)
        masked_vis_lm_labels = torch.masked_select(masked_lm_labels_b, vis_lm_mask).reshape(-1)
        vis_mlm_loss = ce_loss(vis_lm_score, masked_vis_lm_labels)

        # VSE cross entropy loss similar to CLIP
        logit_scale = self.logit_scale.exp()
        retrieval_logit_mat = sim_mat * logit_scale
        pseudo_label = torch.arange(sim_mat.shape[0], device=sim_mat.device)
        retrieval_loss = (ce_loss(retrieval_logit_mat, pseudo_label) + ce_loss(retrieval_logit_mat.t(), pseudo_label))/2

        
        sequence_output, pooled_output, hard_sequence_output, hard_pooled_output = outputs
        lm_mask = masked_lm_labels_a > -1
        masked_sequence_output = torch.masked_select(sequence_output[:, :input_ids_a.shape[1], :], lm_mask.unsqueeze(-1)).reshape(-1, self.config.hidden_size)
        prediction_scores, seq_relationship_score = self.cls(masked_sequence_output, torch.cat([pooled_output, hard_pooled_output], dim=0))
        masked_lm_labels = torch.masked_select(masked_lm_labels_a, lm_mask).reshape(-1)
        masked_lm_loss = ce_loss(prediction_scores, masked_lm_labels.reshape(-1))
        next_sentence_label = torch.cat([torch.zeros(pooled_output.shape[0], dtype=torch.long), torch.ones(hard_pooled_output.shape[0], dtype=torch.long)], dim=0).to(seq_relationship_score.device)
        next_sentence_loss = ce_loss(seq_relationship_score.view(-1, self.num_seq_relations), next_sentence_label.view(-1))
        # outputs = (prediction_scores, seq_relationship_score,) + outputs[2:]  # add hidden states and attention if they are here
        
        # total_loss = vis_mlm_loss+retrieval_loss+masked_lm_loss+next_sentence_loss
        total_loss = retrieval_loss+masked_lm_loss+next_sentence_loss
        outputs = (vis_mlm_loss, retrieval_loss, masked_lm_loss, next_sentence_loss)

        # if masked_lm_labels is not None and next_sentence_label is not None:
        #     loss_fct = CrossEntropyLoss(ignore_index=-1)
        #     masked_lm_labels = torch.masked_select(masked_lm_labels, lm_mask).reshape(-1)
        #     masked_lm_loss = loss_fct(prediction_scores, masked_lm_labels.reshape(-1))
        #     next_sentence_loss = loss_fct(seq_relationship_score.view(-1, self.num_seq_relations), next_sentence_label.view(-1))
        #     total_loss = masked_lm_loss + next_sentence_loss
        #     outputs = outputs + (masked_lm_loss,)

        if phrase_index is not None:
            # weakly supervised phrase grouding
            # start_time = time.time()
            valid_phrases = F.normalize(mask_slice_and_stack(sequence_output, phrase_index), p=2, dim=-1)
            valid_images = F.normalize(mask_slice_and_stack(sequence_output, img_index), p=2, dim=-1)
            # time_1 = time.time()
            full_sims = valid_phrases @ valid_images.t()
            # time_2 = time.time()
            pos_sims, neg_sims = get_pos_neg_sims(full_sims, phrase_index, img_index)
            # time_3 = time.time()
            wra_loss = torch.clamp(neg_sims + 0.2 - pos_sims, min=0)
            # wra_loss = torch.max(wra_loss, dim=1)[0]
            # time_4 = time.time()
            wra_valid_mask = (phrase_index[:, 1] - phrase_index[:, 0])>0
            # wra_valid_mask = torch.bitwise_and(wra_valid_mask, is_img_match==0)
            wra_loss = torch.mean(torch.masked_select(wra_loss, wra_valid_mask))
            total_loss = total_loss + wra_loss
            # time_5 = time.time()
            # print('scatter time:', time_1-start_time, 'matmul time:', time_2-time_1, 'pos_neg_sim:', time_3-time_2,
                    # 'loss_time', time_4-time_3, 'mask loss time', time_5-time_4)
            outputs = (total_loss,) + outputs + (wra_loss,)
        else:
            outputs = (total_loss,) + outputs

        return outputs  # (loss), prediction_scores, seq_relationship_score, (hidden_states), (attentions)

class BiBertImgForPreTraining2(ImgPreTrainedModel): # a version with weakly-supervised grounding
    r"""
        with a contrastive learning for wrong qa pairs!
        **masked_lm_labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Labels for computing the masked language modeling loss.
            Indices should be in ``[-1, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
            Tokens with indices set to ``-1`` are ignored (masked), the loss is only computed for the tokens with labels
            in ``[0, ..., config.vocab_size]``
        **next_sentence_label**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the next sequence prediction (classification) loss. Input should be a sequence pair (see ``input_ids`` docstring)
            Indices should be in ``[0, 1]``.
            ``0`` indicates sequence B is a continuation of sequence A,
            ``1`` indicates sequence B is a random sequence.

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when both ``masked_lm_labels`` and ``next_sentence_label`` are provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Total loss as the sum of the masked language modeling loss and the next sequence prediction (classification) loss.
        **prediction_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, config.vocab_size)``
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        **seq_relationship_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, 2)``
            Prediction scores of the next sequence prediction (classification) head (scores of True/False continuation before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        >>> config = BertConfig.from_pretrained('bert-base-uncased')
        >>> tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        >>>
        >>> model = BertImgForPreTraining(config)
        >>> input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        >>> outputs = model(input_ids)
        >>> prediction_scores, seq_relationship_scores = outputs[:2]

    """
    config_class = BertConfig
    pretrained_model_archive_map = BERT_PRETRAINED_MODEL_ARCHIVE_MAP
    load_tf_weights = load_tf_weights_in_bert
    base_model_prefix = "bert"

    def __init__(self, config):
        super(BiBertImgForPreTraining2, self).__init__(config)

        #self.bert = BertModel(config) # original BERT
        self.bert = BiBertImgModel2(config)
        self.cls = BertPreTrainingHeads(config, only_vocab=True)
        self.half_mlm = BertLMPredictionHead(config, only_vocab=True)
        # self.qa_head = nn.Linear(config.hidden_size, config.qa_answer_size)
        self.only_vocab_size = config.only_word_size
        self.qa_seq_relation = nn.Linear(config.hidden_size, 2)
        self.num_seq_relations = config.num_contrast_classes if hasattr(config, "num_contrast_classes") else 2
        self.max_text_seq_length = config.max_text_seq_length if hasattr(config, "max_text_seq_length") else None
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        # self.max_text_seq_length = None

        self.apply(self.init_weights)
        self.tie_weights()

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0,
                                       std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(self.cls.predictions.decoder,
                                   self.bert.embeddings.word_embeddings, only_vocab=True, only_word_size=self.only_vocab_size)

        self._tie_or_clone_weights(self.half_mlm.decoder, self.bert.embeddings.word_embeddings,
                                   only_vocab=True, only_word_size=self.only_vocab_size)

    def forward(self, input_ids_a, token_type_ids_a=None, attention_mask_a=None, masked_lm_labels_a=None, qa_is_next=None,
            input_ids_b=None, token_type_ids_b=None, attention_mask_b=None, masked_lm_labels_b=None, max_tag_length=20, phrase_layer=None,
            position_ids_a=None, position_ids_b=None, head_mask=None, img_feats=None,is_img_match=None, img_index=None, phrase_index=None, phrase_mod='sample'):
        outputs, single_stream_output, hard_indexes, mid_output = self.bert(input_ids_a=input_ids_a, position_ids_a=position_ids_a, token_type_ids_a=token_type_ids_a,
                            attention_mask_a=attention_mask_a, head_mask=head_mask, img_feats=img_feats, phrase_layer=phrase_layer,
                            input_ids_b=input_ids_b, position_ids_b=position_ids_b, token_type_ids_b=token_type_ids_b,
                            attention_mask_b=attention_mask_b, max_tag_length=max_tag_length, encode_hn=True)

        txt_encoder_outputs, vis_encoder_outputs, sim_mat = single_stream_output

        ce_loss = CrossEntropyLoss(ignore_index=-1)

        # visual encoder MLM
        vis_lm_mask = masked_lm_labels_b > -1
        vis_masked_seq_output = torch.masked_select(vis_encoder_outputs, vis_lm_mask.unsqueeze(-1)).reshape(-1, self.config.hidden_size)
        vis_lm_score = self.half_mlm(vis_masked_seq_output)
        masked_vis_lm_labels = torch.masked_select(masked_lm_labels_b, vis_lm_mask).reshape(-1)
        vis_mlm_loss = ce_loss(vis_lm_score, masked_vis_lm_labels)

        # VSE cross entropy loss similar to CLIP
        logit_scale = self.logit_scale.exp()
        retrieval_logit_mat = sim_mat * logit_scale
        pseudo_label = torch.arange(sim_mat.shape[0], device=sim_mat.device)
        retrieval_loss = (ce_loss(retrieval_logit_mat, pseudo_label) + ce_loss(retrieval_logit_mat.t(), pseudo_label))/2

        
        sequence_output, pooled_output, hard_sequence_output, hard_pooled_output = outputs
        lm_mask = masked_lm_labels_a > -1
        masked_sequence_output = torch.masked_select(sequence_output[:, :input_ids_a.shape[1], :], lm_mask.unsqueeze(-1)).reshape(-1, self.config.hidden_size)
        prediction_scores, seq_relationship_score = self.cls(masked_sequence_output, torch.cat([pooled_output, hard_pooled_output], dim=0))
        masked_lm_labels = torch.masked_select(masked_lm_labels_a, lm_mask).reshape(-1)
        masked_lm_loss = ce_loss(prediction_scores, masked_lm_labels.reshape(-1))
        next_sentence_label = torch.cat([torch.zeros(pooled_output.shape[0], dtype=torch.long), torch.ones(hard_pooled_output.shape[0], dtype=torch.long)], dim=0).to(seq_relationship_score.device)
        next_sentence_loss = ce_loss(seq_relationship_score.view(-1, self.num_seq_relations), next_sentence_label.view(-1))
        # outputs = (prediction_scores, seq_relationship_score,) + outputs[2:]  # add hidden states and attention if they are here
        
        total_loss = vis_mlm_loss+retrieval_loss+masked_lm_loss+next_sentence_loss
        outputs = (vis_mlm_loss, retrieval_loss, masked_lm_loss, next_sentence_loss)

        # if masked_lm_labels is not None and next_sentence_label is not None:
        #     loss_fct = CrossEntropyLoss(ignore_index=-1)
        #     masked_lm_labels = torch.masked_select(masked_lm_labels, lm_mask).reshape(-1)
        #     masked_lm_loss = loss_fct(prediction_scores, masked_lm_labels.reshape(-1))
        #     next_sentence_loss = loss_fct(seq_relationship_score.view(-1, self.num_seq_relations), next_sentence_label.view(-1))
        #     total_loss = masked_lm_loss + next_sentence_loss
        #     outputs = outputs + (masked_lm_loss,)
        if qa_is_next is not None:
            qa_logits = self.qa_seq_relation(pooled_output)
            qa_rel_loss = ce_loss(qa_logits, qa_is_next)
            total_loss += qa_rel_loss
            outputs = outputs + (qa_rel_loss,)

        if phrase_index is not None:
            if phrase_layer is not None:
                sequence_output_mid, hard_sequence_output_mid = mid_output
            else:
                sequence_output_mid, hard_sequence_output_mid = sequence_output, hard_sequence_output
            if phrase_mod == 'hard':
                hard_txt_index, hard_img_index = hard_indexes
                hard_phrase_index = torch.index_select(phrase_index, dim=0, index=hard_txt_index)
                hard_object_index = torch.index_select(img_index, dim=0, index=hard_img_index)
                pos_sims = get_pos_sims(sequence_output=sequence_output_mid, text_index=phrase_index, img_index=img_index)
                neg_sims = get_pos_sims(sequence_output=hard_sequence_output_mid, text_index=hard_phrase_index, img_index=hard_object_index)
                wra_loss = torch.clamp(neg_sims + 0.2 - pos_sims, min=0)
                wra_valid_mask1 = (phrase_index[:, 1] - phrase_index[:, 0])>0
                wra_valid_mask2 = (hard_phrase_index[:, 1] - hard_phrase_index[:, 0])>0
                wra_valid_mask = torch.bitwise_and(wra_valid_mask1, wra_valid_mask2)
                wra_loss = torch.mean(torch.masked_select(wra_loss, wra_valid_mask))
                total_loss = total_loss + wra_loss
                outputs = (total_loss,) + outputs + (wra_loss,)

            elif phrase_mod == 'sample':
                # weakly supervised phrase grouding
                # start_time = time.time()
                valid_phrases = F.normalize(mask_slice_and_stack(sequence_output_mid, phrase_index), p=2, dim=-1)
                valid_images = F.normalize(mask_slice_and_stack(sequence_output_mid, img_index), p=2, dim=-1)
                # time_1 = time.time()
                full_sims = valid_phrases @ valid_images.t()
                # time_2 = time.time()
                pos_sims, neg_sims = get_pos_neg_sims(full_sims, phrase_index, img_index)
                # time_3 = time.time()
                wra_loss = torch.clamp(neg_sims + 0.2 - pos_sims, min=0)
                # wra_loss = torch.max(wra_loss, dim=1)[0]
                # time_4 = time.time()
                wra_valid_mask = (phrase_index[:, 1] - phrase_index[:, 0])>0
                # wra_valid_mask = torch.bitwise_and(wra_valid_mask, is_img_match==0)
                wra_loss = torch.mean(torch.masked_select(wra_loss, wra_valid_mask))
                total_loss = total_loss + wra_loss
                # time_5 = time.time()
                # print('scatter time:', time_1-start_time, 'matmul time:', time_2-time_1, 'pos_neg_sim:', time_3-time_2,
                        # 'loss_time', time_4-time_3, 'mask loss time', time_5-time_4)
                outputs = (total_loss,) + outputs + (wra_loss,)
            else:
                raise NotImplementedError
        else:
            outputs = (total_loss,) + outputs

        return outputs  # (loss), prediction_scores, seq_relationship_score, (hidden_states), (attentions)

def get_pos_sims(sequence_output, text_index, img_index):
    # only calculate the pos similarity in a batch
    pos_sims = []

    for idx in range(text_index.shape[0]):
        cur_valid_features = sequence_output[idx]
        cur_valid_txt_index = torch.arange(text_index[idx,0], text_index[idx,1], device=sequence_output.device)
        cur_valid_txt_feat = F.normalize(cur_valid_features.index_select(0, cur_valid_txt_index), p=2, dim=-1)
        cur_valid_img_index = torch.arange(img_index[idx,0], img_index[idx,1], device=sequence_output.device)
        cur_valid_img_feat = F.normalize(cur_valid_features.index_select(0, cur_valid_img_index), p=2, dim=-1)
        if cur_valid_txt_feat.shape[0] == 0:
            pos_sims.append(torch.zeros(1, dtype=cur_valid_img_feat.dtype, device=cur_valid_img_feat.device).squeeze())
        else:
            pos_sims.append(t2i_sim(cur_valid_txt_feat @ cur_valid_img_feat.t()))

    pos_sims = torch.stack(pos_sims)

    return pos_sims

def mask_slice_and_stack(features, valid_index):
    stacker = []
    for idx in range(features.shape[0]):
        cur_valid_features = features[idx]
        cur_valid_index = torch.arange(valid_index[idx,0], valid_index[idx,1], device=features.device)
        stacker.append(cur_valid_features.index_select(0, cur_valid_index))
    return torch.cat(stacker, dim=0)


def t2i_sim(sim_matrix):
    if sim_matrix.shape[0] == 0:
        return torch.zeros(1, dtype=sim_matrix.dtype, device=sim_matrix.device).squeeze()
    # f_sim = sim_matrix.max(dim=1)[0]
    f_sim = sim_matrix.topk(3, dim=1)[0]
    rand_index = torch.randint(0, 3, (f_sim.shape[0],), device=f_sim.device)
    f_sim = f_sim[torch.arange(f_sim.shape[0], device=f_sim.device), rand_index]
    return torch.mean(f_sim)


def get_pos_neg_sims(sims, text_index, img_index):
    text_n_input = text_index[:, 1] - text_index[:, 0]
    img_n_input = img_index[:, 1] - img_index[:, 0]
    text_index_border = text_n_input.cumsum(dim=0)
    img_index_border = img_n_input.cumsum(dim=0)
    my_zero = torch.zeros(1, device=sims.device, dtype=text_n_input.dtype)

    text_index_border = torch.cat([my_zero, text_index_border], dim=0)
    img_index_border = torch.cat([my_zero, img_index_border], dim=0)

    doc2pos_sim = {}
    doc2neg_img_sims = {}

    for text_idx in range(text_index.shape[0]):
        doc2pos_sim[text_idx] = t2i_sim
        text_start = text_index_border[text_idx]
        text_end = text_index_border[text_idx+1]
        img_start = img_index_border[text_idx]
        img_end = img_index_border[text_idx+1]
        doc2pos_sim[text_idx] = t2i_sim(sims[text_start:text_end, img_start:img_end])
        neg_img_indexs = list(range(0, text_idx)) + list(range(text_idx + 1, img_index.shape[0]))
        neg_img_idx = random.choice(neg_img_indexs)
        neg_img_start = img_index_border[neg_img_idx]
        neg_img_end = img_index_border[neg_img_idx+1]
        doc2neg_img_sims[text_idx] = t2i_sim(sims[text_start:text_end, neg_img_start:neg_img_end])
        # for img_idx in range(img_index.shape[0]):
        #     text_start = text_index_border[text_idx]
        #     text_end = text_index_border[text_idx+1]
        #     img_start = img_index_border[img_idx]
        #     img_end = img_index_border[img_idx+1]
        #     c_sim = t2i_sim(sims[text_start:text_end, img_start:img_end])
        #     if text_idx == img_idx:
        #         doc2pos_sim[text_idx] = c_sim
        #     else:
        #         doc2neg_img_sims[text_idx].append(c_sim)

    pos_sims, neg_sims = [], []
    for idx in range(text_index.shape[0]):
        pos_sims.append(doc2pos_sim[idx])
        neg_sims.append(doc2neg_img_sims[idx])

    pos_sims = torch.stack(pos_sims)
    neg_sims = torch.stack(neg_sims)

    return pos_sims, neg_sims


class BiImageBertForRetrieval(BertPreTrainedModel):
    """
    Modified from BertForSequenceClassification to support oscar training.
    """
    def __init__(self, config):
        super(BiImageBertForRetrieval, self).__init__(config)
        self.num_labels = 2
        self.loss_type = config.loss_type
        self.config = config
        if config.img_feature_dim > 0:
            self.bert = BiBertImgModel2(config)
        else:
            self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.forward_mod = 'train'

        if hasattr(config, 'classifier'):
            if not hasattr(config, 'cls_hidden_scale'): 
                config.cls_hidden_scale = 2

            if config.classifier == 'linear':
                self.classifier = nn.Linear(config.hidden_size,
                                            self.config.num_labels)
            elif config.classifier == 'mlp':
                self.classifier = nn.Sequential(
                    nn.Linear(config.hidden_size, config.hidden_size * config.cls_hidden_scale),
                    nn.ReLU(),
                    nn.Linear(config.hidden_size * config.cls_hidden_scale, self.config.num_labels)
                )
        else:
            self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)  # original
        self.apply(self.init_weights)

    def init_code_embedding(self, em):
        self.bert.code_embeddings.weight.data = em.clone()

    def reinit_cls_head(self):
        # make a re-initialization for the classifier
        self.classifier.apply(self.init_weights)

    def forward(self, input_ids_a, token_type_ids_a=None, attention_mask_a=None, phrase_index=None, img_index=None,
            input_ids_b=None, token_type_ids_b=None, attention_mask_b=None, max_tag_length=20, phrase_layer=None, no_itm=False,
            position_ids_a=None, position_ids_b=None, head_mask=None, img_feats=None):
        if self.forward_mod == 'train':
            return self.forward_train(input_ids_a=input_ids_a, token_type_ids_a=token_type_ids_a, attention_mask_a=attention_mask_a, no_itm=no_itm,
            input_ids_b=input_ids_b, token_type_ids_b=token_type_ids_b, attention_mask_b=attention_mask_b, img_feats=img_feats, max_tag_length=max_tag_length,
            position_ids_a=position_ids_a, position_ids_b=position_ids_b, head_mask=head_mask, phrase_index=phrase_index, img_index=img_index, phrase_layer=phrase_layer)
        elif self.forward_mod == 'coarse':
            return self.forward_emb(input_ids_a=input_ids_a, token_type_ids_a=token_type_ids_a, attention_mask_a=attention_mask_a,
            input_ids_b=input_ids_b, token_type_ids_b=token_type_ids_b, attention_mask_b=attention_mask_b, img_feats=img_feats, max_tag_length=max_tag_length,
            position_ids_a=position_ids_a, position_ids_b=position_ids_b, head_mask=head_mask)
        elif self.forward_mod == 'fine':
            return self.forward_fine(input_ids_a=input_ids_a, token_type_ids_a=token_type_ids_a, attention_mask_a=attention_mask_a,
            input_ids_b=input_ids_b, token_type_ids_b=token_type_ids_b, attention_mask_b=attention_mask_b, img_feats=img_feats, max_tag_length=max_tag_length,
            position_ids_a=position_ids_a, position_ids_b=position_ids_b, head_mask=head_mask)
        else:
            raise NotImplementedError


    def forward_train(self, input_ids_a, token_type_ids_a=None, attention_mask_a=None, phrase_index=None, img_index=None,
            input_ids_b=None, token_type_ids_b=None, attention_mask_b=None, max_tag_length=20, phrase_layer=None,
            position_ids_a=None, position_ids_b=None, head_mask=None, img_feats=None, no_itm=False):
        # print(input_ids_a.shape)
        outputs, single_stream_output, hard_indexes, mid_output = self.bert(input_ids_a=input_ids_a, position_ids_a=position_ids_a, token_type_ids_a=token_type_ids_a,
                            attention_mask_a=attention_mask_a, head_mask=head_mask, img_feats=img_feats, phrase_layer=phrase_layer,
                            input_ids_b=input_ids_b, position_ids_b=position_ids_b, token_type_ids_b=token_type_ids_b,
                            attention_mask_b=attention_mask_b, max_tag_length=max_tag_length, encode_hn=True)

        txt_encoder_outputs, vis_encoder_outputs, sim_mat = single_stream_output

        ce_loss = CrossEntropyLoss(ignore_index=-1)

        # VSE cross entropy loss similar to CLIP
        logit_scale = self.logit_scale.exp()
        retrieval_logit_mat = sim_mat * logit_scale
        pseudo_label = torch.arange(sim_mat.shape[0], device=sim_mat.device)
        retrieval_loss = (ce_loss(retrieval_logit_mat, pseudo_label) + ce_loss(retrieval_logit_mat.t(), pseudo_label))/2

        
        sequence_output, pooled_output, hard_sequence_output, hard_pooled_output = outputs
        seq_relationship_score = self.classifier(self.dropout(torch.cat([pooled_output, hard_pooled_output], dim=0)))
        next_sentence_label = torch.cat([torch.ones(pooled_output.shape[0], dtype=torch.long), torch.zeros(hard_pooled_output.shape[0], dtype=torch.long)], dim=0).to(seq_relationship_score.device)
        next_sentence_loss = ce_loss(seq_relationship_score.view(-1, self.num_labels), next_sentence_label.view(-1))
        
        if no_itm:
            total_loss = retrieval_loss
        else:
            total_loss = retrieval_loss + next_sentence_loss

        phrase_mod = 'hard'
        if phrase_index is not None:
            if phrase_layer is not None:
                sequence_output_mid, hard_sequence_output_mid = mid_output
            else:
                sequence_output_mid, hard_sequence_output_mid = sequence_output, hard_sequence_output
            if phrase_mod == 'hard':
                hard_txt_index, hard_img_index = hard_indexes
                hard_phrase_index = torch.index_select(phrase_index, dim=0, index=hard_txt_index)
                hard_object_index = torch.index_select(img_index, dim=0, index=hard_img_index)
                pos_sims = get_pos_sims(sequence_output=sequence_output_mid, text_index=phrase_index, img_index=img_index)
                neg_sims = get_pos_sims(sequence_output=hard_sequence_output_mid, text_index=hard_phrase_index, img_index=hard_object_index)
                wra_loss = torch.clamp(neg_sims + 0.2 - pos_sims, min=0)
                wra_valid_mask1 = (phrase_index[:, 1] - phrase_index[:, 0])>0
                wra_valid_mask2 = (hard_phrase_index[:, 1] - hard_phrase_index[:, 0])>0
                wra_valid_mask = torch.bitwise_and(wra_valid_mask1, wra_valid_mask2)
                wra_loss = torch.mean(torch.masked_select(wra_loss, wra_valid_mask))
                total_loss = total_loss + wra_loss
                # outputs = (total_loss,) + outputs + (wra_loss,)

            elif phrase_mod == 'sample':
                # weakly supervised phrase grouding
                # start_time = time.time()
                valid_phrases = F.normalize(mask_slice_and_stack(sequence_output_mid, phrase_index), p=2, dim=-1)
                valid_images = F.normalize(mask_slice_and_stack(sequence_output_mid, img_index), p=2, dim=-1)
                # time_1 = time.time()
                full_sims = valid_phrases @ valid_images.t()
                # time_2 = time.time()
                pos_sims, neg_sims = get_pos_neg_sims(full_sims, phrase_index, img_index)
                # time_3 = time.time()
                wra_loss = torch.clamp(neg_sims + 0.2 - pos_sims, min=0)
                # wra_loss = torch.max(wra_loss, dim=1)[0]
                # time_4 = time.time()
                wra_valid_mask = (phrase_index[:, 1] - phrase_index[:, 0])>0
                # wra_valid_mask = torch.bitwise_and(wra_valid_mask, is_img_match==0)
                wra_loss = torch.mean(torch.masked_select(wra_loss, wra_valid_mask))
                total_loss = total_loss + wra_loss
                # time_5 = time.time()
                # print('scatter time:', time_1-start_time, 'matmul time:', time_2-time_1, 'pos_neg_sim:', time_3-time_2,
                        # 'loss_time', time_4-time_3, 'mask loss time', time_5-time_4)
                # outputs = (total_loss,) + outputs + (wra_loss,)
            else:
                raise NotImplementedError
    
        outputs = (total_loss, seq_relationship_score, retrieval_loss, next_sentence_loss, next_sentence_label)
        if phrase_index is not None:
            outputs += (wra_loss,)
        # outputs = (prediction_scores, seq_relationship_score,) + outputs[2:]  # add hidden states and attention if they are here

        return outputs  # (loss), prediction_scores, seq_relationship_score, (hidden_states), (attentions)

    
    def forward_emb(self, input_ids_a, token_type_ids_a=None, attention_mask_a=None,
            input_ids_b=None, token_type_ids_b=None, attention_mask_b=None, max_tag_length=20,
            position_ids_a=None, position_ids_b=None, head_mask=None, img_feats=None):
        global_txt, global_img = self.bert.forward_single(input_ids_a=input_ids_a, position_ids_a=position_ids_a, token_type_ids_a=token_type_ids_a,
                            attention_mask_a=attention_mask_a, head_mask=head_mask, img_feats=img_feats,
                            input_ids_b=input_ids_b, position_ids_b=position_ids_b, token_type_ids_b=token_type_ids_b,
                            attention_mask_b=attention_mask_b, max_tag_length=max_tag_length)
        return (global_txt, global_img)  # (loss), prediction_scores, seq_relationship_score, (hidden_states), (attentions)

    def forward_fine(self, input_ids_a, token_type_ids_a=None, attention_mask_a=None,
            input_ids_b=None, token_type_ids_b=None, attention_mask_b=None, max_tag_length=20,
            position_ids_a=None, position_ids_b=None, head_mask=None, img_feats=None):
        outputs, single_stream_output, hard_indexes, mid_output = self.bert(input_ids_a=input_ids_a, position_ids_a=position_ids_a, token_type_ids_a=token_type_ids_a,
                            attention_mask_a=attention_mask_a, head_mask=head_mask, img_feats=img_feats,
                            input_ids_b=input_ids_b, position_ids_b=position_ids_b, token_type_ids_b=token_type_ids_b,
                            attention_mask_b=attention_mask_b, max_tag_length=max_tag_length, encode_hn=False)
        
        sequence_output, pooled_output, hard_sequence_output, hard_pooled_output = outputs
        seq_relationship_score = self.classifier(pooled_output)
        # outputs = (total_loss, sim_mat, seq_relationship_score)
        # outputs = (prediction_scores, seq_relationship_score,) + outputs[2:]  # add hidden states and attention if they are here

        return seq_relationship_score  # (loss), prediction_scores, seq_relationship_score, (hidden_states), (attentions)



class BiImageBertForSequenceClassification(BertPreTrainedModel):
    """
    Modified from BertForSequenceClassification to support oscar training.
    """
    def __init__(self, config):
        super(BiImageBertForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.loss_type = config.loss_type
        self.config = config
        if config.img_feature_dim > 0:
            self.bert = BiBertImgModel(config)
        else:
            self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        if hasattr(config, 'classifier'):
            if not hasattr(config, 'cls_hidden_scale'): 
                config.cls_hidden_scale = 2

            if config.classifier == 'linear':
                self.classifier = nn.Linear(config.hidden_size,
                                            self.config.num_labels)
            elif config.classifier == 'mlp':
                self.classifier = nn.Sequential(
                    nn.Linear(config.hidden_size, config.hidden_size * config.cls_hidden_scale),
                    nn.ReLU(),
                    nn.Linear(config.hidden_size * config.cls_hidden_scale, self.config.num_labels)
                )
        else:
            self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)  # original
        self.apply(self.init_weights)

    def init_code_embedding(self, em):
        self.bert.code_embeddings.weight.data = em.clone()

    def reinit_cls_head(self):
        # make a re-initialization for the classifier
        self.classifier.apply(self.init_weights)

    def forward(self, input_ids_a, token_type_ids_a=None, attention_mask_a=None, labels=None,
            input_ids_b=None, token_type_ids_b=None, attention_mask_b=None, max_tag_length=20,
            position_ids_a=None, position_ids_b=None, head_mask=None, img_feats=None, soft_label=False):
        outputs, single_stream_output = self.bert(input_ids_a=input_ids_a, position_ids_a=position_ids_a, token_type_ids_a=token_type_ids_a,
                            attention_mask_a=attention_mask_a, head_mask=head_mask, img_feats=img_feats,
                            input_ids_b=input_ids_b, position_ids_b=position_ids_b, token_type_ids_b=token_type_ids_b,
                            attention_mask_b=attention_mask_b, max_tag_length=max_tag_length, encode_hn=False)
        sequence_output, pooled_output, hard_sequence_output, hard_pooled_output = outputs

        pooled_output = self.dropout(pooled_output)
        # pooled_output = self.dropout(sequence_output[:,0])
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        # print('in model batch size', logits.shape)
        if labels is not None:
            if self.num_labels == 1: #  doing regression
                loss_fct = MSELoss()
                labels = labels.to(torch.float)
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                if soft_label:
                    loss = soft_cross_entropy(labels, logits)
                elif self.loss_type == 'kl':
                    # KL Loss: https://github.com/uclanlp/visualbert/blob/master/pytorch_pretrained_bert/modeling.py
                    loss_fct = torch.nn.KLDivLoss(reduction="batchmean")
                    log_softmax = torch.nn.LogSoftmax(dim=-1)
                    reshaped_logits = logits.contiguous().view(-1, 3129)
                    reshaped_logits = log_softmax(reshaped_logits)
                    loss = loss_fct(reshaped_logits, labels.contiguous())
                elif self.loss_type == 'bce': # [VQA]
                    loss = instance_bce_with_logits(logits, labels)
                else: # cross_entropy [GQA, Retrieval, Captioning]
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        return outputs


class BiImageBertForRE(BertPreTrainedModel):
    """
    Modified from BertForSequenceClassification to support oscar training.
    """
    def __init__(self, config):
        super(BiImageBertForRE, self).__init__(config)
        self.num_labels = 1
        self.loss_type = config.loss_type
        self.config = config
        if config.img_feature_dim > 0:
            self.bert = BiBertImgModel2(config)
        else:
            self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        if hasattr(config, 'classifier'):
            if not hasattr(config, 'cls_hidden_scale'): 
                config.cls_hidden_scale = 2

            if config.classifier == 'linear':
                self.classifier = nn.Linear(config.hidden_size,
                                            self.config.num_labels)
            elif config.classifier == 'mlp':
                self.classifier = nn.Sequential(
                    nn.Linear(config.hidden_size, config.hidden_size * config.cls_hidden_scale),
                    nn.ReLU(),
                    nn.Linear(config.hidden_size * config.cls_hidden_scale, self.config.num_labels)
                )
        else:
            self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)  # original
        self.apply(self.init_weights)

    def init_code_embedding(self, em):
        self.bert.code_embeddings.weight.data = em.clone()

    def freeze_backbone(self):
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.bert.parameters():
            param.requires_grad = True

    def reinit_cls_head(self):
        # make a re-initialization for the classifier
        self.classifier.apply(self.init_weights)

    def forward(self, input_ids_a, token_type_ids_a=None, attention_mask_a=None, labels=None, phrase_layer=None,
            input_ids_b=None, token_type_ids_b=None, attention_mask_b=None, max_tag_length=20, mod=1,
            position_ids_a=None, position_ids_b=None, head_mask=None, img_feats=None, soft_label=False):
        outputs, single_stream_output, hard_indexes, mid_out = self.bert(input_ids_a=input_ids_a, position_ids_a=position_ids_a, token_type_ids_a=token_type_ids_a,
                            attention_mask_a=attention_mask_a, head_mask=head_mask, img_feats=img_feats, phrase_layer=phrase_layer,
                            input_ids_b=input_ids_b, position_ids_b=position_ids_b, token_type_ids_b=token_type_ids_b,
                            attention_mask_b=attention_mask_b, max_tag_length=max_tag_length, encode_hn=False)
        if phrase_layer is not None:
            sequence_output = mid_out[0]
        else:
            sequence_output = outputs[0]

        
        seq_a_length = input_ids_a.shape[1]
        sequence_output = self.dropout(sequence_output)
        vis_sequence_output = sequence_output[:, seq_a_length:]
        cls_output = sequence_output[:, 0]

        if mod == 1:
            # mse with cosine similarity
            vis_sequence_output = F.normalize(vis_sequence_output, p=2, dim=-1)
            cls_output = F.normalize(cls_output, p=2 ,dim=-1).unsqueeze(-1)
            logits = torch.bmm(vis_sequence_output, cls_output).squeeze()
            label_mask = labels >= 0
            mse = MSELoss()
            # mse = torch.nn.L1Loss()
            loss = mse(torch.masked_select(labels, label_mask), torch.masked_select(logits, label_mask))

        elif mod == 2:
            # bce with cls as classifier
            logits = torch.bmm(vis_sequence_output, cls_output.unsqueeze(-1)).squeeze()
            label_mask = labels >= 0
            hard_labels = (labels >= 0.5).float()
            loss = F.binary_cross_entropy_with_logits(torch.masked_select(logits, label_mask), torch.masked_select(hard_labels, label_mask))
            logits = torch.sigmoid(logits)

        elif mod == 3:
            logits = self.classifier(vis_sequence_output).squeeze()
            label_mask = labels >= 0
            hard_labels = (labels >= 0.5).float()
            loss = F.binary_cross_entropy_with_logits(torch.masked_select(logits, label_mask), torch.masked_select(labels, label_mask))
        
        else:
            raise NotImplementedError

        outputs = (loss, logits)
        return outputs
