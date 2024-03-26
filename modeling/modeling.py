from .utils import ACT2FN
from .loss import CrossEntropyLoss_, FocalLoss_, CrossEntropyLossForTrans_
from . import predefined_constant
from .config import BertForSeq2SeqConfig
from .convert_state_dict import get_checkpoint_from_transformer_cache, state_dict_convert
from .predefined_constant import LAYOUTLM_PRETRAINED_CONFIG_ARCHIVE_MAP, LAYOUTLM_PRETRAINED_MODEL_ARCHIVE_MAP, BERT_PRETRAINED_MODEL_ARCHIVE_MAP
from .beam_search import StoppingCriteriaList, MaxLengthCriteria, BeamSearchScorer, MinLengthLogitsProcessor, LogitsProcessorList, beam_search

import torch
from torch import nn
import torch.nn.functional as F

from transformers.modeling_bert import BertPreTrainedModel


import math
import os
import copy

import logging
logger = logging.getLogger(__name__)

# ************************* The basic word reordering task. ********************************

class BertPreTrainedForSeq2SeqModel(BertPreTrainedModel):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    config_class = BertForSeq2SeqConfig
    supported_convert_pretrained_model_archive_map = {
        "bert": BERT_PRETRAINED_MODEL_ARCHIVE_MAP,
        "layoutlm": LAYOUTLM_PRETRAINED_MODEL_ARCHIVE_MAP,
    }
    base_model_prefix = "bert_for_seq2seq"
    pretrained_model_archive_map = {
        **BERT_PRETRAINED_MODEL_ARCHIVE_MAP,
        **LAYOUTLM_PRETRAINED_MODEL_ARCHIVE_MAP,
    }

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path,
                        *model_args, **kwargs):
        base_model_type = kwargs.pop('base_model_type', None)
        if base_model_type is not None and "state_dict" not in kwargs:
            if base_model_type in cls.supported_convert_pretrained_model_archive_map:
                pretrained_model_archive_map = cls.supported_convert_pretrained_model_archive_map[base_model_type]
                if pretrained_model_name_or_path in pretrained_model_archive_map:
                    state_dict = get_checkpoint_from_transformer_cache(
                        archive_file=pretrained_model_archive_map[pretrained_model_name_or_path],
                        pretrained_model_name_or_path=pretrained_model_name_or_path,
                        pretrained_model_archive_map=pretrained_model_archive_map,
                        cache_dir=kwargs.get("cache_dir", None), force_download=kwargs.get("force_download", None),
                        proxies=kwargs.get("proxies", None), resume_download=kwargs.get("resume_download", None),
                    )
                    state_dict = state_dict_convert[base_model_type](state_dict)
                    kwargs["state_dict"] = state_dict
                elif os.path.isfile(pretrained_model_name_or_path):
                    kwargs["state_dict"] = torch.load(pretrained_model_name_or_path, map_location='cpu')

        if kwargs["state_dict"] is None:
            logger.info("s2s-ft does't support the model !")
            raise NotImplementedError()

        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

BertLayerNorm = torch.nn.LayerNorm

class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, position_ids=None, inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = inputs_embeds + position_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings # shape = [batch_size, max_src_len, embedding_dim].



class LayoutlmEmbeddings(nn.Module):
    def __init__(self, config):
        super(LayoutlmEmbeddings, self).__init__()

        self.only_layout_flag = config.layoutlm_only_layout 

        if not config.layoutlm_only_layout:
            self.word_embeddings = nn.Embedding(
                config.vocab_size, config.hidden_size
            )
        else:
            self.word_embeddings = None

        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )

        self.x_position_embeddings = nn.Embedding(
            config.max_2d_position_embeddings, config.hidden_size
        )
        self.y_position_embeddings = nn.Embedding(
            config.max_2d_position_embeddings, config.hidden_size
        )
        self.h_position_embeddings = nn.Embedding(
            config.max_2d_position_embeddings, config.hidden_size
        )
        self.w_position_embeddings = nn.Embedding(
            config.max_2d_position_embeddings, config.hidden_size
        )

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        input_ids,
        bbox,
        position_ids=None,
        inputs_embeds=None,
    ):
        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(
                seq_length, dtype=torch.long, device=input_ids.device
            )
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        left_position_embeddings = self.x_position_embeddings(bbox[:, :, 0])
        upper_position_embeddings = self.y_position_embeddings(bbox[:, :, 1])
        right_position_embeddings = self.x_position_embeddings(bbox[:, :, 2])
        lower_position_embeddings = self.y_position_embeddings(bbox[:, :, 3])
        h_position_embeddings = self.h_position_embeddings(
            bbox[:, :, 3] - bbox[:, :, 1]
        )
        w_position_embeddings = self.w_position_embeddings(
            bbox[:, :, 2] - bbox[:, :, 0]
        )

        position_embeddings = self.position_embeddings(position_ids)

        embeddings = (
                left_position_embeddings
                + upper_position_embeddings
                + right_position_embeddings
                + lower_position_embeddings
                + h_position_embeddings
                + w_position_embeddings
                + position_embeddings
                # + token_type_embeddings
        )

        if not self.only_layout_flag:
            words_embeddings = self.word_embeddings(input_ids)
            embeddings = embeddings + words_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings # shape = [batch_size, max_src_len, embedding_dim].
    
class BertSelfAttention(nn.Module):

    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.output_attentions = config.output_attentions

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3) # shape: [batch_size, num_attn_heads, seq_len, attn_head_size]

    def multi_head_attention(self, query, key, value, attention_mask):
        query_layer = self.transpose_for_scores(query)
        key_layer = self.transpose_for_scores(key)
        value_layer = self.transpose_for_scores(value)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) # shape: [batch_size, num_attn_heads, seq_len, seq_len]
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores) # shape: [batch_size, num_attn_heads, seq_len, seq_len]

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer) # shape: [batch_size, num_attn_heads, seq_len, attn_head_size]

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape) # shape: [batch_size, seq_len, hidden_size]

        return (context_layer, attention_probs) if self.output_attentions else (context_layer,)

    def forward(self, hidden_states, attention_mask, encoder_hidden_states=None):
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        if encoder_hidden_states is not None:
            mixed_key_layer = self.key(encoder_hidden_states)
            mixed_value_layer = self.value(encoder_hidden_states)
        else:
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

        outputs = self.multi_head_attention(
            mixed_query_layer, mixed_key_layer, mixed_value_layer, attention_mask)
        return outputs 

class BertSelfAttentionIncr(nn.Module):

    def __init__(self, config):
        super(BertSelfAttentionIncr, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3) # shape: [batch_size, num_attn_heads, seq_len, attn_head_size]

    def multi_head_attention(self, query, key, value, attention_mask):
        query_layer = self.transpose_for_scores(query)
        key_layer = self.transpose_for_scores(key)
        value_layer = self.transpose_for_scores(value)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) # shape: [batch_size, num_attn_heads, 1, accu_decoding_steps]
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores) # shape: [batch_size, num_attn_heads, 1, accu_decoding_steps]


        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer) # shape: [batch_size, num_attn_heads, 1, attn_head_size]

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape) # shape: [batch_size, 1, hidden_size]

        return (context_layer,)

    def forward(self, accu_hidden_states, cur_time_hidden_states, attention_mask, encoder_hidden_states=None):
        """
        accu_hidden_states: shape = [batch_size, accu_decoding_steps, hidden_dim]. accu_decoding_steps including current decoding_steps.
        cur_time_hidden_states: shape = [batch_size, 1, hidden_dim].
        attention_mask: shape = [batch_size, 1, accu_decoding_steps].
        """

        mixed_query_layer = self.query(cur_time_hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        if encoder_hidden_states is not None:
            mixed_key_layer = self.key(encoder_hidden_states)
            mixed_value_layer = self.value(encoder_hidden_states)
        else:
            mixed_key_layer = self.key(accu_hidden_states)
            mixed_value_layer = self.value(accu_hidden_states)



        outputs = self.multi_head_attention(
            mixed_query_layer, mixed_key_layer, mixed_value_layer, attention_mask) # shape: [batch_size, 1, hidden_size]
        return outputs 
    
class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
    
class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = ACT2FN[config.hidden_act] \
            if isinstance(config.hidden_act, str) else config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states
    
class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
    
class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, hidden_states, attention_mask, encoder_hidden_states=None):
        self_outputs = self.self(
            hidden_states, attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states)
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:] 
        return outputs
    
class BertAttentionIncr(nn.Module):
    def __init__(self, config):
        super(BertAttentionIncr, self).__init__()
        self.self = BertSelfAttentionIncr(config)
        self.output = BertSelfOutput(config)

    def forward(self, accu_hidden_states, cur_time_hidden_states, attention_mask, encoder_hidden_states=None):
        """
        accu_hidden_states: shape = [batch_size, accu_decoding_steps, hidden_dim]. accu_decoding_steps including current decoding_steps.
        cur_time_hidden_states: shape = [batch_size, 1, hidden_dim].
        attention_mask: shape = [batch_size, 1, accu_decoding_steps].
        """
        self_outputs = self.self(
            accu_hidden_states, cur_time_hidden_states, attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states)
        attention_output = self.output(self_outputs[0], cur_time_hidden_states)
        outputs = (attention_output,) + self_outputs[1:] 
        return outputs # shape: [batch_size, 1, hidden_size]

class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask):
        self_attention_outputs = self.attention(
            hidden_states, attention_mask)
        attention_output = self_attention_outputs[0]

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + self_attention_outputs[1:]
        return outputs

    
class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        self.output_attentions = config.output_attentions
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask):
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):

            layer_outputs = layer_module(hidden_states, attention_mask)
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        outputs = (hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all attentions)


class BertModel(BertPreTrainedForSeq2SeqModel):

    def __init__(self, config):
        super(BertModel, self).__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)

    def forward(self, input_ids, attention_mask,
                position_ids=None, inputs_embeds=None, return_emb=False):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]

        # Provided a padding mask of dimensions [batch_size, seq_length]
        # - if the model is a decoder, apply a causal mask in addition to the padding mask
        # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, inputs_embeds=inputs_embeds)
        encoder_outputs = self.encoder(
            embedding_output, attention_mask=extended_attention_mask)
        sequence_output = encoder_outputs[0]

        outputs = (sequence_output, ) + encoder_outputs[1:]  # add hidden_states and attentions if they are here. last-layer hidden_state, (all attentions).

        if return_emb:
            outputs += (embedding_output,)

        return outputs  # last-layer hidden_state, (all attentions), (embedding_output).
        
class BertCrossAttention(nn.Module):

    def __init__(self, config):
        super(BertCrossAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.output_attentions = config.output_attentions

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3) # shape: [batch_size, num_attn_heads, seq_len, attn_head_size].

    def multi_head_attention(self, query, key, value, attention_mask):
        query_layer = self.transpose_for_scores(query)
        key_layer = self.transpose_for_scores(key)
        value_layer = self.transpose_for_scores(value)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) # shape: [batch_size, num_attn_heads, seq_len, seq_len]
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores) # shape: [batch_size, num_attn_heads, seq_len, seq_len]

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer) # shape: [batch_size, num_attn_heads, seq_len, attn_head_size]

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape) # shape: [batch_size, seq_len, hidden_size]

        return (context_layer, attention_probs) if self.output_attentions else (context_layer,)

    def forward(self, encoder_context, decoder_hidden_states, cross_attention_mask):
        """
            encoder_context: shape = [batch_size, max_src_len, hidden_dim].
            decoder_hidden_states: shape = [batch_size, max_tgt_len, hidden_dim].
            cross_attention_mask: shape = [batch_size, max_tgt_len, max_src_len].
        """
        mixed_query_layer = self.query(decoder_hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        mixed_key_layer = self.key(encoder_context)
        mixed_value_layer = self.value(encoder_context)

        outputs = self.multi_head_attention(
            mixed_query_layer, mixed_key_layer, mixed_value_layer, cross_attention_mask)
        return outputs 
    
class BertCrossAttentionIncr(nn.Module):

    def __init__(self, config):
        super(BertCrossAttentionIncr, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3) # shape: [batch_size, num_attn_heads, seq_len, attn_head_size].

    def multi_head_attention(self, query, key, value, attention_mask):
        query_layer = self.transpose_for_scores(query)
        key_layer = self.transpose_for_scores(key)
        value_layer = self.transpose_for_scores(value)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) # shape: [batch_size, num_attn_heads, seq_len, seq_len]
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores) # shape: [batch_size, num_attn_heads, seq_len, seq_len]

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer) # shape: [batch_size, num_attn_heads, seq_len, attn_head_size]

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape) # shape: [batch_size, seq_len, hidden_size]

        return (context_layer,)

    def forward(self, encoder_context, decoder_cur_time_hidden_states, cross_attention_mask):
        """
            encoder_context: shape = [batch_size, max_src_len, hidden_dim].
            decoder_cur_time_hidden_states: shape = [batch_size, 1, hidden_dim].
            cross_attention_mask: shape = [batch_size, 1, max_src_len].
        """
        mixed_query_layer = self.query(decoder_cur_time_hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        mixed_key_layer = self.key(encoder_context)
        mixed_value_layer = self.value(encoder_context)

        outputs = self.multi_head_attention(
            mixed_query_layer, mixed_key_layer, mixed_value_layer, cross_attention_mask)
        return outputs 

class BertCrossAttentionLayer(nn.Module):
    def __init__(self, config):
        super(BertCrossAttentionLayer, self).__init__()
        self.cross = BertCrossAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, encoder_context, decoder_hidden_states, cross_attention_mask):
        cross_outputs = self.cross(
            encoder_context, decoder_hidden_states, cross_attention_mask)
        attention_output = self.output(cross_outputs[0], decoder_hidden_states)
        outputs = (attention_output,) + cross_outputs[1:] 
        return outputs

class BertCrossAttentionLayerIncr(nn.Module):
    def __init__(self, config):
        super(BertCrossAttentionLayerIncr, self).__init__()
        self.cross = BertCrossAttentionIncr(config)
        self.output = BertSelfOutput(config)

    def forward(self, encoder_context, decoder_cur_time_hidden_states, cross_attention_mask):
        """
            encoder_context: shape = [batch_size, max_src_len, hidden_dim].
            decoder_cur_time_hidden_states: shape = [batch_size, 1, hidden_dim].
            cross_attention_mask: shape = [batch_size, 1, max_src_len].
        """
        cross_outputs = self.cross(
            encoder_context, decoder_cur_time_hidden_states, cross_attention_mask)
        attention_output = self.output(cross_outputs[0], decoder_cur_time_hidden_states)
        outputs = (attention_output,) + cross_outputs[1:] 
        return outputs

class BertLayerForDecoder(nn.Module):
    def __init__(self, config):
        super(BertLayerForDecoder, self).__init__()
        self.self_attention = BertAttention(config)
        self.cross_attention = BertCrossAttentionLayer(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, encoder_context, decoder_hidden_states, cross_attention_mask, attention_mask):
        """
            encoder_context: shape = [batch_size, max_src_len, hidden_dim].
            decoder_hidden_states: shape = [batch_size, max_tgt_len, hidden_dim].
            cross_attention_mask: shape = [batch_size, max_tgt_len, max_src_len].
            attention_mask: shape = [batch_size, max_tgt_len, max_tgt_len]
        """

        self_attention_outputs = self.self_attention(
            decoder_hidden_states, attention_mask)
        self_attention_output = self_attention_outputs[0]

        cross_attention_outputs = self.cross_attention(
            encoder_context, self_attention_output, cross_attention_mask)
        cross_attention_output = cross_attention_outputs[0]

        intermediate_output = self.intermediate(cross_attention_output)
        layer_output = self.output(intermediate_output, cross_attention_output)
        outputs = (layer_output,) + self_attention_outputs[1:] + cross_attention_outputs[1:]
        return outputs
        # layer_output: shape = [batch_size, max_tgt_len, hidden_dim].
        # self_attention_outputs[1:]: self_attention_probs of shape = [batch_size, max_src_len, max_src_len].
        # cross_attention_outputs[1:]: cross_attention_probs of shape = [batch_size, max_tgt_len, max_src_len].

class BertLayerForDecoderIncr(nn.Module):
    def __init__(self, config):
        super(BertLayerForDecoderIncr, self).__init__()
        self.self_attention = BertAttentionIncr(config)
        self.cross_attention = BertCrossAttentionLayerIncr(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, encoder_context, decoder_accu_hidden_states, decoder_cur_time_hidden_states, cross_attention_mask, attention_mask):
        """
            encoder_context: shape = [batch_size, max_src_len, hidden_dim].
            decoder_accu_hidden_states: shape = [batch_size, accu_decoding_steps, hidden_dim]. accu_decoding_steps including current decoding_steps.
            decoder_cur_time_hidden_states: shape = [batch_size, 1, hidden_dim]. 
            cross_attention_mask: shape = [batch_size, 1, max_src_len].
            attention_mask: shape = [batch_size, 1, accu_decoding_steps].
        """
        self_attention_outputs = self.self_attention(
            decoder_accu_hidden_states, decoder_cur_time_hidden_states, attention_mask)
        self_attention_output = self_attention_outputs[0] # shape = [batch_size, 1, hidden_dim].
        cross_attention_outputs = self.cross_attention(
            encoder_context, self_attention_output, cross_attention_mask) 
        cross_attention_output = cross_attention_outputs[0] # shape = [batch_size, 1, hidden_dim].

        intermediate_output = self.intermediate(cross_attention_output)
        layer_output = self.output(intermediate_output, cross_attention_output)
        outputs = (layer_output,)
        
        return outputs
        # layer_output: shape = [batch_size, 1, hidden_dim].

class BertDecoder(nn.Module):
    def __init__(self, config):
        super(BertDecoder, self).__init__()
        self.output_attentions = config.output_attentions
        self.layer = nn.ModuleList([BertLayerForDecoder(config) for _ in range(config.num_hidden_layers)])

    def forward(self, encoder_context, decoder_hidden_states, cross_attention_mask, attention_mask):

        all_decoder_self_attentions = ()
        all_decoder_cross_attentions = ()

        for i, layer_module in enumerate(self.layer):

            layer_outputs = layer_module(encoder_context, decoder_hidden_states, cross_attention_mask, attention_mask)
            if self.output_attentions:
                decoder_hidden_states, decoder_self_attentions, decoder_cross_attentions = layer_outputs
            else:
                decoder_hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_decoder_self_attentions = all_decoder_self_attentions + (decoder_self_attentions, )
                all_decoder_cross_attentions = all_decoder_cross_attentions + (decoder_cross_attentions, )

        outputs = (decoder_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_decoder_self_attentions,) + (all_decoder_cross_attentions,)
        return outputs  # last-layer hidden state, (all_decoder_self_attentions), (all_decoder_cross_attentions)
    
class BertDecoderIncr(nn.Module):
    def __init__(self, config):
        super(BertDecoderIncr, self).__init__()
        self.layer = nn.ModuleList([BertLayerForDecoderIncr(config) for _ in range(config.num_hidden_layers)])

    def forward(self, encoder_context, decoder_all_layers_historical_hidden_states, decoder_cur_time_hidden_states, cross_attention_mask, attention_mask):
        """
            encoder_context: shape = [batch_size, max_src_len, hidden_dim].
            decoder_all_layers_historical_hidden_states: a tuple containing: first decoder embedding output, then layer_0 - layer_(n-1) historical hidden_states, each element is of the shape = [batch_size, historical_decoding_steps, hidden_dim]. or () for the first decoding step.
            decoder_cur_time_hidden_states: shape = [batch_size, 1, hidden_dim].
            cross_attention_mask: shape = [batch_size, 1, max_src_len].
            attention_mask: shape = [batch_size, 1, accu_decoding_steps].
        """

        decoder_all_layers_cur_time_hidden_states = (decoder_cur_time_hidden_states, )

        if decoder_all_layers_historical_hidden_states:
            decoder_all_layers_accu_hidden_states = (torch.cat((decoder_all_layers_historical_hidden_states[0], decoder_cur_time_hidden_states), dim=1), )
        else:
            decoder_all_layers_accu_hidden_states = (decoder_cur_time_hidden_states, )

        for i, layer_module in enumerate(self.layer):
            if decoder_all_layers_historical_hidden_states:
                layer_decoder_historical_hidden_states = decoder_all_layers_historical_hidden_states[i]
                layer_decoder_accu_hidden_states = torch.cat(
                    (layer_decoder_historical_hidden_states, decoder_cur_time_hidden_states), dim=1
                )
            else:
                layer_decoder_accu_hidden_states = decoder_cur_time_hidden_states
            
            
            layer_outputs = layer_module(
                    encoder_context,
                    layer_decoder_accu_hidden_states,
                    decoder_cur_time_hidden_states,
                    cross_attention_mask,
                    attention_mask,
                )
            decoder_cur_time_hidden_states = layer_outputs[0]

            decoder_all_layers_cur_time_hidden_states += (decoder_cur_time_hidden_states, )


        if decoder_all_layers_historical_hidden_states:
            decoder_all_layers_accu_hidden_states = ()
            for historical_hidden_states, cur_time_hidden_states in zip(decoder_all_layers_historical_hidden_states, decoder_all_layers_cur_time_hidden_states):
                decoder_all_layers_accu_hidden_states += (torch.cat((historical_hidden_states, cur_time_hidden_states), dim=1), )
        else:
            decoder_all_layers_accu_hidden_states = decoder_all_layers_cur_time_hidden_states
        
        return (decoder_all_layers_accu_hidden_states, )
        # first the decoder_embedding_output, then decoder layer_0 - layer_(n-1)'s hidden_states, accumulated up to the current decoding step.
        # Each element is of the shape = [batch_size, accu_decoding_steps, hidden_dim].




class BertModelForDecoder(BertPreTrainedForSeq2SeqModel):

    def __init__(self, config, encoder):
        super(BertModelForDecoder, self).__init__(config)
        self.config = config

        self.embeddings = encoder.embeddings
        # self.embeddings = BertEmbeddings(config)
        self.decoder = BertDecoder(config)

    def forward(self, encoder_context, decoder_input_ids, attention_mask, cross_attention_mask, 
                position_ids=None, inputs_embeds=None):
        if decoder_input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif decoder_input_ids is not None:
            input_shape = decoder_input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = decoder_input_ids.device if decoder_input_ids is not None else inputs_embeds.device

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]

        # Provided a padding mask of dimensions [batch_size, seq_length]
        # - if the model is a decoder, apply a causal mask in addition to the padding mask
        # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # for cross_attention_mask.
        if cross_attention_mask.dim() == 3:
            extended_cross_attention_mask = cross_attention_mask[:, None, :, :]
        if cross_attention_mask.dim() == 2:
            extended_cross_attention_mask = cross_attention_mask[:, None, None, :]
        extended_cross_attention_mask = extended_cross_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_cross_attention_mask = (1.0 - extended_cross_attention_mask) * -10000.0


        embedding_output = self.embeddings(
            input_ids=decoder_input_ids, position_ids=position_ids, inputs_embeds=inputs_embeds)
        decoder_outputs = self.decoder(
            encoder_context, embedding_output, extended_cross_attention_mask, extended_attention_mask)
        sequence_output = decoder_outputs[0]

        outputs = (sequence_output, ) + decoder_outputs[1:]  # add hidden_states and attentions if they are here. last-layer hidden state, (all_decoder_self_attentions), (all_decoder_cross_attentions)

        return outputs  # last-layer hidden_state, (all_decoder_self_attentions), (all_decoder_cross_attentions).
    
class BertModelForDecoderIncr(BertPreTrainedForSeq2SeqModel):

    def __init__(self, config, encoder):
        super(BertModelForDecoderIncr, self).__init__(config)
        self.config = config

        self.embeddings = encoder.embeddings
        # self.embeddings = BertEmbeddings(config)
        self.decoder = BertDecoderIncr(config)

    def forward(self, encoder_context, decoder_all_layers_historical_hidden_states, decoder_cur_time_input_ids, attention_mask, cross_attention_mask, 
                position_ids=None, inputs_embeds=None):
        """
            encoder_context: shape = [batch_size, max_src_len, hidden_dim].
            decoder_all_layers_historical_hidden_states: a tuple containing: first decoder embedding output, then layer_0 - layer_(n-1) historical hidden_states, each element is of the shape = [batch_size, historical_decoding_steps, hidden_dim]. or () for the first decoding step.
            decoder_cur_time_input_ids: shape = [batch_size, 1].
            attention_mask: shape = [batch_size, 1, accu_decoding_steps]. Accu_decoding_steps including current decoding step.
            cross_attention_mask: shape = [batch_size, 1, max_src_len].

        """

        if decoder_cur_time_input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif decoder_cur_time_input_ids is not None:
            input_shape = decoder_cur_time_input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = decoder_cur_time_input_ids.device if decoder_cur_time_input_ids is not None else inputs_embeds.device

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]

        # Provided a padding mask of dimensions [batch_size, seq_length]
        # - if the model is a decoder, apply a causal mask in addition to the padding mask
        # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # for cross_attention_mask.
        if cross_attention_mask.dim() == 3:
            extended_cross_attention_mask = cross_attention_mask[:, None, :, :]
        if cross_attention_mask.dim() == 2:
            extended_cross_attention_mask = cross_attention_mask[:, None, None, :]
        extended_cross_attention_mask = extended_cross_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_cross_attention_mask = (1.0 - extended_cross_attention_mask) * -10000.0

        batch_size = decoder_cur_time_input_ids.shape[0]
        cur_decoding_step = decoder_all_layers_historical_hidden_states[0].shape[1] if decoder_all_layers_historical_hidden_states else 0

        cur_time_position_ids = torch.ones((batch_size, 1), dtype=torch.long, device=decoder_cur_time_input_ids.device) * cur_decoding_step

        embedding_output = self.embeddings(
            input_ids=decoder_cur_time_input_ids, position_ids=cur_time_position_ids, inputs_embeds=inputs_embeds) # shape = [batch_size, 1, embedding_dim].
        decoder_all_layers_accu_hidden_states = self.decoder(
            encoder_context, decoder_all_layers_historical_hidden_states, embedding_output, extended_cross_attention_mask, extended_attention_mask)[0]

        return (decoder_all_layers_accu_hidden_states, )  
        # first the decoder_embedding_output, then decoder layer_0 - layer_(n-1)'s hidden_states, accumulated up to the current decoding step.
        # Each element is of the shape = [batch_size, accu_decoding_steps, hidden_dim].



class LayoutlmModel(BertPreTrainedForSeq2SeqModel):

    def __init__(self, config):
        super(LayoutlmModel, self).__init__(config)
        self.config = config

        self.embeddings = LayoutlmEmbeddings(config)
        self.encoder = BertEncoder(config)

    def forward(self,
                input_ids,
                bbox,
                attention_mask,
                position_ids=None,
                inputs_embeds=None,
                return_emb=False):

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]

        # Provided a padding mask of dimensions [batch_size, seq_length]
        # - if the model is a decoder, apply a causal mask in addition to the padding mask
        # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # embedding_output = self.embeddings(
        #     input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds).
        embedding_output = self.embeddings(
            input_ids, bbox, position_ids=position_ids, inputs_embeds=inputs_embeds)
        encoder_outputs = self.encoder(
            embedding_output, attention_mask=extended_attention_mask)
        sequence_output = encoder_outputs[0]

        outputs = (sequence_output, ) + encoder_outputs[1:]  # add hidden_states and attentions if they are here. last-layer hidden_state, (all attentions).

        if return_emb:
            outputs += (embedding_output,)

        return outputs  # last-layer hidden_state, (all attentions), (embedding_output).
    

class LayoutlmModelForDecoder(BertPreTrainedForSeq2SeqModel):

    def __init__(self, config, encoder):
        super(LayoutlmModelForDecoder, self).__init__(config)
        self.config = config

        self.embeddings = encoder.embeddings
        # self.embeddings = LayoutlmEmbeddings(config)
        self.decoder = BertDecoder(config)

    def forward(self, encoder_context, decoder_input_ids, bbox, attention_mask, cross_attention_mask, 
                position_ids=None, inputs_embeds=None):
        if decoder_input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif decoder_input_ids is not None:
            input_shape = decoder_input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = decoder_input_ids.device if decoder_input_ids is not None else inputs_embeds.device

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]

        # Provided a padding mask of dimensions [batch_size, seq_length]
        # - if the model is a decoder, apply a causal mask in addition to the padding mask
        # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # for cross_attention_mask.
        if cross_attention_mask.dim() == 3:
            extended_cross_attention_mask = cross_attention_mask[:, None, :, :]
        if cross_attention_mask.dim() == 2:
            extended_cross_attention_mask = cross_attention_mask[:, None, None, :]
        extended_cross_attention_mask = extended_cross_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_cross_attention_mask = (1.0 - extended_cross_attention_mask) * -10000.0


        embedding_output = self.embeddings(
            input_ids=decoder_input_ids, bbox=bbox, position_ids=position_ids, inputs_embeds=inputs_embeds)
        decoder_outputs = self.decoder(
            encoder_context, embedding_output, extended_cross_attention_mask, extended_attention_mask)
        sequence_output = decoder_outputs[0]

        outputs = (sequence_output, ) + decoder_outputs[1:]  # add hidden_states and attentions if they are here. last-layer hidden state, (all_decoder_self_attentions), (all_decoder_cross_attentions)

        return outputs  # last-layer hidden_state, (all_decoder_self_attentions), (all_decoder_cross_attentions).
    

class LayoutlmModelForDecoderIncr(BertPreTrainedForSeq2SeqModel):

    def __init__(self, config, encoder):
        super(LayoutlmModelForDecoderIncr, self).__init__(config)
        self.config = config

        self.embeddings = encoder.embeddings
        # self.embeddings = BertEmbeddings(config)
        self.decoder = BertDecoderIncr(config)

    def forward(self, encoder_context, decoder_all_layers_historical_hidden_states, decoder_cur_time_input_ids, bbox, attention_mask, cross_attention_mask, 
                position_ids=None, inputs_embeds=None):
        """
            encoder_context: shape = [batch_size, max_src_len, hidden_dim].
            decoder_all_layers_historical_hidden_states: a tuple containing: first decoder embedding output, then layer_0 - layer_(n-1) historical hidden_states, each element is of the shape = [batch_size, historical_decoding_steps, hidden_dim]. or () for the first decoding step.
            decoder_cur_time_input_ids: shape = [batch_size, 1].
            decoder_cur_time_bbox: shape = [batch_size, 1, 4].
            attention_mask: shape = [batch_size, 1, accu_decoding_steps]. Accu_decoding_steps including current decoding step.
            cross_attention_mask: shape = [batch_size, 1, max_src_len].

        """
        if decoder_cur_time_input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif decoder_cur_time_input_ids is not None:
            input_shape = decoder_cur_time_input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = decoder_cur_time_input_ids.device if decoder_cur_time_input_ids is not None else inputs_embeds.device

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]

        # Provided a padding mask of dimensions [batch_size, seq_length]
        # - if the model is a decoder, apply a causal mask in addition to the padding mask
        # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # for cross_attention_mask.
        if cross_attention_mask.dim() == 3:
            extended_cross_attention_mask = cross_attention_mask[:, None, :, :]
        if cross_attention_mask.dim() == 2:
            extended_cross_attention_mask = cross_attention_mask[:, None, None, :]
        extended_cross_attention_mask = extended_cross_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_cross_attention_mask = (1.0 - extended_cross_attention_mask) * -10000.0

        batch_size = decoder_cur_time_input_ids.shape[0]
        cur_decoding_step = decoder_all_layers_historical_hidden_states[0].shape[1] if decoder_all_layers_historical_hidden_states else 0

        cur_time_position_ids = torch.ones((batch_size, 1), dtype=torch.long, device=decoder_cur_time_input_ids.device) * cur_decoding_step

        embedding_output = self.embeddings(
            input_ids=decoder_cur_time_input_ids, position_ids=cur_time_position_ids, inputs_embeds=inputs_embeds, bbox=bbox) # shape = [batch_size, 1, embedding_dim].
        decoder_all_layers_accu_hidden_states = self.decoder(
            encoder_context, decoder_all_layers_historical_hidden_states, embedding_output, extended_cross_attention_mask, extended_attention_mask)[0]
        
        return (decoder_all_layers_accu_hidden_states, )  
        # first the decoder_embedding_output, then decoder layer_0 - layer_(n-1)'s hidden_states, accumulated up to the current decoding step.
        # Each element is of the shape = [batch_size, accu_decoding_steps, hidden_dim].

class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()
        self.transform_act_fn = ACT2FN[config.hidden_act] \
            if isinstance(config.hidden_act, str) else config.hidden_act
        hid_size = config.hidden_size
        if hasattr(config, 'relax_projection') and (config.relax_projection > 1):
            hid_size *= config.relax_projection
        self.dense = nn.Linear(config.hidden_size, hid_size)
        self.LayerNorm = BertLayerNorm(hid_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states
    
class LayoutlmSPLMPredictionHead(nn.Module):
    def __init__(self, config, src_len):
        super(LayoutlmSPLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)

        self.bias = nn.Parameter(torch.zeros(src_len))

    def forward(self, hidden_states, src_emb):
        hidden_states = self.transform(hidden_states)
        hidden_states = torch.einsum('btf,bsf->bts', hidden_states, src_emb) + self.bias
        # hidden_states = F.linear(hidden_states, weight=src_emb, bias=self.bias)
        return hidden_states # [batch_size, max_tgt_len, max_src_len].

class LayoutlmSPOnlyMLMHead(nn.Module):
    def __init__(self, config, src_len):
        super(LayoutlmSPOnlyMLMHead, self).__init__()
        self.predictions = LayoutlmSPLMPredictionHead(config, src_len=src_len)

    def forward(self, sequence_output, src_emb):
        """
            sequence_output: shape = [batch_size, max_tgt_len, hidden_dim].
            src_emb: shape = [batch_size, max_src_len, hidden_dim].
        """
        prediction_scores = self.predictions(sequence_output, src_emb=src_emb)
        return prediction_scores # [batch_size, max_tgt_len, max_src_len].
    


class LayoutlmEncoderDecoder(BertPreTrainedForSeq2SeqModel):

    def __init__(self, config):
        super(LayoutlmEncoderDecoder, self).__init__(config)

        self.config = config

        if config.base_model_type == 'layoutlm':
            self.encoder = LayoutlmModel(config)
            self.decoder = LayoutlmModelForDecoder(config, self.encoder)
        else:
            self.encoder = BertModel(config)
            self.decoder = BertModelForDecoder(config, self.encoder)

        self.prediction_head = LayoutlmSPOnlyMLMHead(config, src_len=config.max_source_length)
        
        # Random initializing the weights, which may be overwritten by the pretrained model weights depend on the code logic.
        self.init_weights()

        self.crit_mask_lm_smoothed = CrossEntropyLoss_(config.label_smoothing, config.max_source_length, ignore_index=None, reduction='none')

    @staticmethod
    def create_basic_mask(num_effective_tokens, max_seq_len):
        """
            num_effective_tokens: shape = [batch_size].
        """

        base_position_matrix = torch.arange(
            0, max_seq_len, dtype=num_effective_tokens.dtype, device=num_effective_tokens.device).view(1, -1) # [1, max_seq_len].
        mask = (base_position_matrix < (num_effective_tokens + 1).view(-1, 1)).type_as(num_effective_tokens) #  [batch_size, max_seq_len].
 
        return mask # shape = [batch_size, max_seq_len]

    @staticmethod
    def create_attention_mask(source_mask, target_mask, ):
        """
        inputs:
            source_mask: shape = [batch_size, max_src_len].
            target_mask: shape = [batch_size, max_tgt_len].

        returns:
            source_self_attn_mask: shape = [batch_size, max_src_len, max_src_len].
            target_self_attn_mask: shape = [batch_size, max_tgt_len, max_tgt_len].
            cross_attn_mask: shape = [batch_size, max_tgt_len, max_src_len].

        """
        
        # produce source_self_attn_mask.
        batch_size = source_mask.size(0)
        from_seq_len, to_seq_len = source_mask.size(1), source_mask.size(1)

        cls_token_attn_mask = torch.zeros((batch_size, from_seq_len, to_seq_len), dtype=source_mask.dtype, device=source_mask.device)
        cls_token_attn_mask[:, 0, :] = source_mask[:, :]

        effective_token_attn_mask = source_mask.unsqueeze(1).expand((batch_size, from_seq_len, to_seq_len))
        non_cls_pad_mask = source_mask.unsqueeze(2).expand((batch_size, from_seq_len, to_seq_len))
        effective_token_attn_mask = (effective_token_attn_mask & non_cls_pad_mask).type_as(source_mask)

        pad_token_attn_mask = torch.zeros((batch_size, from_seq_len, to_seq_len), dtype=source_mask.dtype, device=source_mask.device)
        ## produce the final source_self_attn_mask.
        source_self_attn_mask = (cls_token_attn_mask | effective_token_attn_mask | pad_token_attn_mask).type_as(source_mask)


        # produce target_self_attn_mask.
        batch_size = target_mask.size(0)
        from_seq_len, to_seq_len = target_mask.size(1), target_mask.size(1)
        sep_token_attn_mask = torch.zeros((batch_size, from_seq_len, to_seq_len), dtype=target_mask.dtype, device=target_mask.device)
        sep_token_attn_mask[:, 0, 0] = 1
        effective_token_attn_mask = torch.tril(torch.ones(from_seq_len, to_seq_len), diagonal=0).to(target_mask.dtype).to(target_mask.device).expand((batch_size, from_seq_len, to_seq_len))
        non_sep_pad_mask = target_mask.unsqueeze(2).expand((batch_size, from_seq_len, to_seq_len))
        effective_token_attn_mask = (effective_token_attn_mask & non_sep_pad_mask).type_as(target_mask)
        pad_token_attn_mask = torch.zeros((batch_size, from_seq_len, to_seq_len), dtype=target_mask.dtype, device=target_mask.device)
        ## produce the final target_self_attn_mask.
        target_self_attn_mask = (sep_token_attn_mask | effective_token_attn_mask | pad_token_attn_mask).type_as(target_mask)


        # produce cross_attn_mask.
        batch_size = target_mask.size(0)
        from_seq_len, to_seq_len = target_mask.size(1), source_mask.size(1)
        sep_token_attn_mask = torch.zeros((batch_size, from_seq_len, to_seq_len), dtype=source_mask.dtype, device=target_mask.device)
        sep_token_attn_mask[:, 0, :] = source_mask[:, :]
        effective_token_attn_mask = source_mask.unsqueeze(1).expand((batch_size, from_seq_len, to_seq_len))
        non_cls_pad_mask = target_mask.unsqueeze(2).expand((batch_size, from_seq_len, to_seq_len))
        effective_token_attn_mask = (effective_token_attn_mask & non_cls_pad_mask).type_as(source_mask)
        pad_token_attn_mask = torch.zeros((batch_size, from_seq_len, to_seq_len), dtype=source_mask.dtype, device=source_mask.device)
        ## produce the final cross_attn_mask.
        cross_attn_mask = (sep_token_attn_mask | effective_token_attn_mask | pad_token_attn_mask).type_as(source_mask)

        return source_self_attn_mask, target_self_attn_mask, cross_attn_mask

    def forward(self, source_idxys, target_idxys, target_index, pseudo_target_idxys, num_effective_source_tokens, num_effective_target_tokens):
        """
            source_idxys: shape = [batch_size, max_src_len, (4)].
            target_idxys: shape = [batch_size, max_tgt_len, (4)].
            target_index: shape = [batch_size, max_target_len].
            pseudo_target_idxys: shape = [batch_size, max_tgt_len, (4)].
            num_effective_source_tokens: shape = [batch_size].
            num_effective_target_tokens: shape = [batch_size].
        """

        max_src_len = source_idxys.size(1)
        max_tgt_len = target_idxys.size(1)
        max_pseudo_tgt_len = pseudo_target_idxys.size(1)
        assert max_tgt_len == max_pseudo_tgt_len
        assert max_src_len > 0 and max_tgt_len > 0

        if self.config.base_model_type == 'layoutlm':
            source_xys = source_idxys[:, :, 1:] # shape = [batch_size, max_src_len, 4].
            target_xys = target_idxys[:, :, 1:]
            pseudo_target_xys = pseudo_target_idxys[:, :, 1:]

            source_ids = source_idxys[:, :, 0] # shape = [batch_size, max_src_len].
            target_ids = target_idxys[:, :, 0]
            pseudo_target_ids = pseudo_target_idxys[:, :, 0]
        else:
            source_ids = source_idxys # shape = [batch_size, max_src_len].
            target_ids = target_idxys
            pseudo_target_ids = pseudo_target_idxys


        source_mask = self.create_basic_mask(num_effective_source_tokens, max_src_len) # shape = [batch_size, max_src_len], each element the value of 0/1, where 1 stands for "valid token" and 0 stands for "special token".
        target_mask = self.create_basic_mask(num_effective_target_tokens, max_tgt_len) # shape = [batch_size, max_tgt_len], each element the value of 0/1, where 1 stands for "valid token" and 0 stands for "special token".

        source_self_attention_mask, target_self_attention_mask, cross_attention_mask = \
            self.create_attention_mask(source_mask, target_mask)

        if self.config.base_model_type == 'layoutlm':
            # encoder
            encoder_outputs = self.encoder(input_ids=source_ids, bbox=source_xys, attention_mask=source_self_attention_mask, position_ids=None, inputs_embeds=None, return_emb=True)
            encoder_context, encoder_embedding_output = encoder_outputs[0], encoder_outputs[-1]
            # decoder
            decoder_outputs = self.decoder(encoder_context=encoder_context, decoder_input_ids=pseudo_target_ids, bbox=pseudo_target_xys, attention_mask=target_self_attention_mask, cross_attention_mask=cross_attention_mask, position_ids=None, inputs_embeds=None)
            decoder_hidden_states = decoder_outputs[0] # shape = [batch_size, max_tgt_len, hidden_dim].
        else:
            # encoder
            encoder_outputs = self.encoder(input_ids=source_ids,  attention_mask=source_self_attention_mask, position_ids=None, inputs_embeds=None, return_emb=True)
            encoder_context, encoder_embedding_output = encoder_outputs[0], encoder_outputs[-1]
            # decoder
            decoder_outputs = self.decoder(encoder_context=encoder_context, decoder_input_ids=pseudo_target_ids, attention_mask=target_self_attention_mask, cross_attention_mask=cross_attention_mask, position_ids=None, inputs_embeds=None)
            decoder_hidden_states = decoder_outputs[0] # shape = [batch_size, max_tgt_len, hidden_dim].
            
        # do the prediction and mask the prediction scores of the special tokens.
        prediction_scores = self.prediction_head(decoder_hidden_states, encoder_embedding_output) # shape: [batch_size, max_tgt_len, max_src_len].

        prediction_scores_masked = prediction_scores # Not leveraging the prior knowledge of num_effective_source_tokens.
        


        # loss calculation.
        loss_mask = target_mask[...]
        masked_loss = self.crit_mask_lm_smoothed(
                F.log_softmax(prediction_scores_masked.float(), dim=-1), target_index, num_effective_src_tokens=num_effective_source_tokens,
                loss_mask=loss_mask) # shape = [batch_size, max_tgt_len].

    
        return masked_loss


    def greedy_decode(self, source_idxys, num_effective_source_tokens, tokenizer):
        """
            source_idxys: shape = [batch_size, max_src_len, (5)].
            num_effective_source_tokens: shape = [batch_size].
        """

        batch_size = source_idxys.size(0)
        max_src_len = source_idxys.size(1)
        # assert max_src_len > 0

        max_source_tokens_cur_batch = int(torch.max(num_effective_source_tokens))

        if self.config.base_model_type == 'layoutlm':
            source_xys = source_idxys[:, :, 1:] # shape = [batch_size, max_src_len, 4].
            source_ids = source_idxys[:, :, 0] # shape = [batch_size, max_src_len].

            # construct the sep token_id and layout for the "start" of decoding.
            target_xys = torch.tensor(predefined_constant.SEP_TOKEN_LAYOUT, dtype=source_xys.dtype, device=source_xys.device).repeat((batch_size, 1, 1)) # shape = [batch_size, 1, 4].
            target_ids = torch.tensor(tokenizer.sep_token_id, dtype=source_ids.dtype, device=source_ids.device).repeat((batch_size, 1)) # shape = [batch_size, 1].
        else:
            source_ids = source_idxys # shape = [batch_size, max_src_len].

            target_ids = torch.tensor(tokenizer.sep_token_id, dtype=source_ids.dtype, device=source_ids.device).repeat((batch_size, 1)) # shape = [batch_size, 1].

        source_mask = self.create_basic_mask(num_effective_source_tokens, max_src_len) # shape = [batch_size, max_src_len].
        target_mask = torch.ones((batch_size, max_source_tokens_cur_batch+1), dtype=source_mask.dtype, device=source_mask.device) # shape = [batch_size, max_source_tokens_cur_batch].
        

        source_self_attention_mask, target_self_attention_mask, cross_attention_mask = \
            self.create_attention_mask(source_mask, target_mask)
        
        # get the encoder context and embedding output.
        encoder_inputs = {
            "input_ids": source_ids, 
            "attention_mask": source_self_attention_mask,
            "position_ids": None,
            "inputs_embeds": None,
            "return_emb": True,
        }
        if self.config.base_model_type == 'layoutlm':
            encoder_inputs["bbox"] = source_xys
        
        encoder_outputs = self.encoder(**encoder_inputs)
        encoder_context, encoder_embedding_output = encoder_outputs[0], encoder_outputs[-1]

        # prepare the source_mask_for_prection_scores.
        source_mask_for_prection_scores = source_mask.clone()
        # source_mask_for_prection_scores[:, 0] = 1
        prediction_scores_mask = ((1.0 - source_mask_for_prection_scores.unsqueeze(1)) * -1e5) # shape = [batch_size, 1, max_src_len].

        # autoregressively decoding the target seq.
        decoded_res = [] # store the decoded results. each item is a tensor of shape = [batch_size, 1], corresponding the the decoded token id at that time step.

        decoding_step = 0
        accu_decoder_input_ids = target_ids.clone() 
        if self.config.base_model_type == 'layoutlm':
            accu_decoder_input_xys = target_xys.clone() 

        cur_input_ids = target_ids.clone()
        if self.config.base_model_type == 'layoutlm':
            cur_input_xys = target_xys.clone()

        while decoding_step < max_source_tokens_cur_batch + 1:

            # prepare decoder inputs of current decoding step.
            cur_self_attention_mask = target_self_attention_mask[:, :(decoding_step+1), :(decoding_step+1)] # shape = [batch_size, decoding_step+1, decoding_step+1].
            cur_cross_attention_mask = cross_attention_mask[:, :(decoding_step+1), :] # shape = [batch_size, decoding_step+1, max_src_len].

            decoder_inputs = {
                "encoder_context": encoder_context,
                "decoder_input_ids": accu_decoder_input_ids,
                "attention_mask": cur_self_attention_mask,
                "cross_attention_mask": cur_cross_attention_mask,
                "position_ids": None,
                "inputs_embeds": None,
            }
            if self.config.base_model_type == 'layoutlm':
                decoder_inputs["bbox"] = accu_decoder_input_xys

            # forward.
            decoder_outputs = self.decoder(**decoder_inputs)
            decoder_hidden_states = decoder_outputs[0] # shape = [batch_size, decoding_step+1, hidden_dim].

            # prediction.
            cur_decoder_hidden_states = decoder_hidden_states[:, -1:, :] # shape = [batch_size, 1, hidden_dim].
            prediction_scores = self.prediction_head(cur_decoder_hidden_states, encoder_embedding_output) # shape = [batch_size, 1, max_src_len].
            # prediction_scores_masked = prediction_scores + prediction_scores_mask # shape = [batch_size, 1, max_src_len]. Leveraging the prior knowledge of num_effective_source_tokens.
            prediction_scores_masked = prediction_scores # Not leveraging the prior knowledge of num_effective_source_tokens.
            _, max_ids = torch.max(prediction_scores_masked, dim=-1) # shape = [batch_size, 1].

            # get the decoded token id (and layout) from the source seq, store the decoded token id, accumulate the decoder input ids and input layouts, and update the cur_input_ids/cur_input_xys for next decoding step.
            cur_decoded_ids = torch.gather(source_ids, 1, max_ids) # shape = [batch_size, 1].
            decoded_res.append(cur_decoded_ids)
            accu_decoder_input_ids = torch.cat((accu_decoder_input_ids, cur_decoded_ids), dim=1) 
            cur_input_ids = cur_decoded_ids
            if  self.config.base_model_type == 'layoutlm':
                _, _, layout_dim = source_xys.shape
                max_ids_ = max_ids.unsqueeze(-1).expand(max_ids.size(0), max_ids.size(1), layout_dim) # shape = [batch_size, 1, 4].
                cur_decoded_xys = torch.gather(source_xys, 1, max_ids_) # shape = [batch_size, 1, 4].
                accu_decoder_input_xys = torch.cat((accu_decoder_input_xys, cur_decoded_xys), dim=1)
                cur_input_xys = cur_decoded_xys


            decoding_step += 1

        return torch.cat(decoded_res, dim=1) # shape = [batch_size, max_source_tokens_cur_batch].


class LayoutlmEncoderDecoderIncr(BertPreTrainedForSeq2SeqModel):

    def __init__(self, config):
        super(LayoutlmEncoderDecoderIncr, self).__init__(config)

        self.config = config

        if config.base_model_type == 'layoutlm':
            self.encoder = LayoutlmModel(config)
            self.decoder = LayoutlmModelForDecoderIncr(config, self.encoder)
        else:
            self.encoder = BertModel(config)
            self.decoder = BertModelForDecoderIncr(config, self.encoder)

        self.prediction_head = LayoutlmSPOnlyMLMHead(config, src_len=config.max_source_length)
        
        # Random initializing the weights, which may be overwritten by the pretrained model weights depend on the code logic.
        self.init_weights()

        self.crit_mask_lm_smoothed = CrossEntropyLoss_(config.label_smoothing, config.max_source_length, ignore_index=None, reduction='none')

    @staticmethod
    def create_basic_mask(num_effective_tokens, max_seq_len):
        """
            num_effective_tokens: shape = [batch_size].
        """

        base_position_matrix = torch.arange(
            0, max_seq_len, dtype=num_effective_tokens.dtype, device=num_effective_tokens.device).view(1, -1) # [1, max_seq_len].
        mask = (base_position_matrix < (num_effective_tokens + 1).view(-1, 1)).type_as(num_effective_tokens) #  [batch_size, max_seq_len].
 
        return mask # shape = [batch_size, max_seq_len]

    @staticmethod
    def create_attention_mask(source_mask, target_mask, ):
        """
        inputs:
            source_mask: shape = [batch_size, max_src_len].
            target_mask: shape = [batch_size, max_tgt_len].

        returns:
            source_self_attn_mask: shape = [batch_size, max_src_len, max_src_len].
            target_self_attn_mask: shape = [batch_size, max_tgt_len, max_tgt_len].
            cross_attn_mask: shape = [batch_size, max_tgt_len, max_src_len].

        """
        
        # produce source_self_attn_mask.
        batch_size = source_mask.size(0)
        from_seq_len, to_seq_len = source_mask.size(1), source_mask.size(1)
        cls_token_attn_mask = torch.zeros((batch_size, from_seq_len, to_seq_len), dtype=source_mask.dtype, device=source_mask.device)
        cls_token_attn_mask[:, 0, :] = source_mask[:, :]
        effective_token_attn_mask = source_mask.unsqueeze(1).expand((batch_size, from_seq_len, to_seq_len))
        non_cls_pad_mask = source_mask.unsqueeze(2).expand((batch_size, from_seq_len, to_seq_len))
        effective_token_attn_mask = (effective_token_attn_mask & non_cls_pad_mask).type_as(source_mask)
        pad_token_attn_mask = torch.zeros((batch_size, from_seq_len, to_seq_len), dtype=source_mask.dtype, device=source_mask.device)
        ## produce the final source_self_attn_mask.
        source_self_attn_mask = (cls_token_attn_mask | effective_token_attn_mask | pad_token_attn_mask).type_as(source_mask)


        # produce target_self_attn_mask.
        batch_size = target_mask.size(0)
        from_seq_len, to_seq_len = target_mask.size(1), target_mask.size(1)
        sep_token_attn_mask = torch.zeros((batch_size, from_seq_len, to_seq_len), dtype=target_mask.dtype, device=target_mask.device)
        sep_token_attn_mask[:, 0, 0] = 1
        effective_token_attn_mask = torch.tril(torch.ones(from_seq_len, to_seq_len), diagonal=0).to(target_mask.dtype).to(target_mask.device).expand((batch_size, from_seq_len, to_seq_len))
        non_sep_pad_mask = target_mask.unsqueeze(2).expand((batch_size, from_seq_len, to_seq_len))
        effective_token_attn_mask = (effective_token_attn_mask & non_sep_pad_mask).type_as(target_mask)
        pad_token_attn_mask = torch.zeros((batch_size, from_seq_len, to_seq_len), dtype=target_mask.dtype, device=target_mask.device)
        ## produce the final target_self_attn_mask.
        target_self_attn_mask = (sep_token_attn_mask | effective_token_attn_mask | pad_token_attn_mask).type_as(target_mask)


        # produce cross_attn_mask.
        batch_size = target_mask.size(0)
        from_seq_len, to_seq_len = target_mask.size(1), source_mask.size(1)
        sep_token_attn_mask = torch.zeros((batch_size, from_seq_len, to_seq_len), dtype=source_mask.dtype, device=target_mask.device)
        sep_token_attn_mask[:, 0, :] = source_mask[:, :]
        effective_token_attn_mask = source_mask.unsqueeze(1).expand((batch_size, from_seq_len, to_seq_len))
        non_cls_pad_mask = target_mask.unsqueeze(2).expand((batch_size, from_seq_len, to_seq_len))
        effective_token_attn_mask = (effective_token_attn_mask & non_cls_pad_mask).type_as(source_mask)
        pad_token_attn_mask = torch.zeros((batch_size, from_seq_len, to_seq_len), dtype=source_mask.dtype, device=source_mask.device)
        ## produce the final cross_attn_mask.
        cross_attn_mask = (sep_token_attn_mask | effective_token_attn_mask | pad_token_attn_mask).type_as(source_mask)

        return source_self_attn_mask, target_self_attn_mask, cross_attn_mask


    def greedy_decode(self, source_idxys, num_effective_source_tokens, tokenizer):
        """
            source_idxys: shape = [batch_size, max_src_len, (5)].
            num_effective_source_tokens: shape = [batch_size].
        """

        batch_size = source_idxys.size(0)
        max_src_len = source_idxys.size(1)
        # assert max_src_len > 0

        max_source_tokens_cur_batch = int(torch.max(num_effective_source_tokens))

        if self.config.base_model_type == 'layoutlm':
            source_xys = source_idxys[:, :, 1:] # shape = [batch_size, max_src_len, 4].
            source_ids = source_idxys[:, :, 0] # shape = [batch_size, max_src_len].

            # construct the sep token_id and layout for the "start" of decoding.
            target_xys = torch.tensor(predefined_constant.SEP_TOKEN_LAYOUT, dtype=source_xys.dtype, device=source_xys.device).repeat((batch_size, 1, 1)) # shape = [batch_size, 1, 4].
            target_ids = torch.tensor(tokenizer.sep_token_id, dtype=source_ids.dtype, device=source_ids.device).repeat((batch_size, 1)) # shape = [batch_size, 1].
        else:
            source_ids = source_idxys # shape = [batch_size, max_src_len].

            target_ids = torch.tensor(tokenizer.sep_token_id, dtype=source_ids.dtype, device=source_ids.device).repeat((batch_size, 1)) # shape = [batch_size, 1].

        source_mask = self.create_basic_mask(num_effective_source_tokens, max_src_len) # shape = [batch_size, max_src_len].
        target_mask = torch.ones((batch_size, max_source_tokens_cur_batch+1), dtype=source_mask.dtype, device=source_mask.device) # shape = [batch_size, max_source_tokens_cur_batch].

        source_self_attention_mask, target_self_attention_mask, cross_attention_mask = \
            self.create_attention_mask(source_mask, target_mask)
        
        # get the encoder context and embedding output.
        encoder_inputs = {
            "input_ids": source_ids, 
            "attention_mask": source_self_attention_mask,
            "position_ids": None,
            "inputs_embeds": None,
            "return_emb": True,
        }
        if self.config.base_model_type == 'layoutlm':
            encoder_inputs["bbox"] = source_xys
        
        encoder_outputs = self.encoder(**encoder_inputs)
        encoder_context, encoder_embedding_output = encoder_outputs[0], encoder_outputs[-1]

        # prepare the source_mask_for_prection_scores.
        source_mask_for_prection_scores = source_mask.clone()
        # source_mask_for_prection_scores[:, 0] = 1
        
        
        prediction_scores_mask = ((1.0 - source_mask_for_prection_scores.unsqueeze(1)) * -1e5) # shape = [batch_size, 1, max_src_len].
        # autoregressively decoding the target seq.
        decoded_res = [] # store the decoded results. each item is a tensor of shape = [batch_size, 1], corresponding the the decoded token id at that time step.

        decoding_step = 0

        cur_input_ids = target_ids.clone()
        # print(f"cur_input_ids: {cur_input_ids}")
        if self.config.base_model_type == 'layoutlm':
            cur_input_xys = target_xys.clone()

        decoder_all_layers_historical_hidden_states = ()
        while decoding_step < max_source_tokens_cur_batch + 1:

            # prepare decoder inputs of current decoding step.
            cur_self_attention_mask = target_self_attention_mask[:, decoding_step:(decoding_step+1), :(decoding_step+1)] # shape = [batch_size, 1, decoding_step+1].
            cur_cross_attention_mask = cross_attention_mask[:, decoding_step:(decoding_step+1), :] # shape = [batch_size, 1, max_src_len].


            decoder_inputs = {
                "encoder_context": encoder_context,
                "decoder_all_layers_historical_hidden_states": decoder_all_layers_historical_hidden_states,
                "decoder_cur_time_input_ids": cur_input_ids,
                "attention_mask": cur_self_attention_mask,
                "cross_attention_mask": cur_cross_attention_mask,
                "position_ids": None,
                "inputs_embeds": None,
            }
            if self.config.base_model_type == 'layoutlm':
                decoder_inputs["bbox"] = cur_input_xys

            # forward.
            decoder_all_layers_accu_hidden_states = self.decoder(**decoder_inputs)[0]
            decoder_last_layer_hidden_states = decoder_all_layers_accu_hidden_states[-1] # shape = [batch_size, decoding_step+1, hidden_dim].
            # prediction.
            cur_decoder_hidden_states = decoder_last_layer_hidden_states[:, -1:, :] # shape = [batch_size, 1, hidden_dim].
            prediction_scores = self.prediction_head(cur_decoder_hidden_states, encoder_embedding_output) # shape = [batch_size, 1, max_src_len].
            # prediction_scores_masked = prediction_scores + prediction_scores_mask # shape = [batch_size, 1, max_src_len]. Leveraging the prior knowledge of num_effective_source_tokens.
            prediction_scores_masked = prediction_scores # Not leveraging the prior knowledge of num_effective_source_tokens.
            _, max_ids = torch.max(prediction_scores_masked, dim=-1) # shape = [batch_size, 1].
            # print(f"prediction_scores_masked: {prediction_scores_masked}")

            # get the decoded token id (and layout) from the source seq, store the decoded token id, accumulate the decoder input ids and input layouts, and update the cur_input_ids/cur_input_xys for next decoding step.
            cur_decoded_ids = torch.gather(source_ids, 1, max_ids) # shape = [batch_size, 1].
            decoded_res.append(cur_decoded_ids)
            # accu_decoder_input_ids = torch.cat((accu_decoder_input_ids, cur_decoded_ids), dim=1) 
            cur_input_ids = cur_decoded_ids
            decoder_all_layers_historical_hidden_states = decoder_all_layers_accu_hidden_states
            if  self.config.base_model_type == 'layoutlm':
                _, _, layout_dim = source_xys.shape
                max_ids_ = max_ids.unsqueeze(-1).expand(max_ids.size(0), max_ids.size(1), layout_dim) # shape = [batch_size, 1, 4].
                cur_decoded_xys = torch.gather(source_xys, 1, max_ids_) # shape = [batch_size, 1, 4].
                # accu_decoder_input_xys = torch.cat((accu_decoder_input_xys, cur_decoded_xys), dim=1)
                cur_input_xys = cur_decoded_xys

            decoding_step += 1

        return torch.cat(decoded_res, dim=1) # shape = [batch_size, max_source_tokens_cur_batch].


# ************************* Adding the sentence segmentation task. ********************************

class BertEncoderForSenseg(nn.Module):
    def __init__(self, config):
        super(BertEncoderForSenseg, self).__init__()
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.senseg_encoder_num_hidden_layers)])

    def forward(self, hidden_states, attention_mask):
        """
            hidden_states: shape = [batch_size, max_tgt_len, hidden_dim].
            attention_mask: shape = [batch_size, max_tgt_len, max_tgt_len].
        """

        # attention mask dimension extension.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        for i, layer_module in enumerate(self.layer):

            layer_outputs = layer_module(hidden_states, extended_attention_mask)
            hidden_states = layer_outputs[0]


        outputs = (hidden_states,)
        return outputs  # last-layer hidden states of shape = [batch_size, max_tgt_len, hidden_dim].


class LayoutlmSensegPredictionHead(nn.Module):
    def __init__(self, config):
        super(LayoutlmSensegPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.senseg_prediction = nn.Linear(config.hidden_size, len(config.senseg_task_ctg_to_id_map), bias=True)

    def forward(self, hidden_states):
        """
            hidden_states: shape = [batch_size, max_tgt_len, hidden_dim].
        """
        hidden_states = self.transform(hidden_states)
        hidden_states = self.senseg_prediction(hidden_states)
        # hidden_states = F.linear(hidden_states, weight=src_emb, bias=self.bias)
        return hidden_states # [batch_size, max_tgt_len, senseg_num_ctg].


class LayoutlmReorderingSenseg(BertPreTrainedForSeq2SeqModel):

    def __init__(self, config):
        super(LayoutlmReorderingSenseg, self).__init__(config)

        self.config = config

        # For word reordering task.
        if config.base_model_type == 'layoutlm':
            self.encoder = LayoutlmModel(config)
            self.decoder = LayoutlmModelForDecoder(config, self.encoder)
        else:
            self.encoder = BertModel(config)
            self.decoder = BertModelForDecoder(config, self.encoder)

        self.prediction_head = LayoutlmSPOnlyMLMHead(config, src_len=config.max_source_length)
        
        self.crit_mask_lm_smoothed = CrossEntropyLoss_(config.label_smoothing, config.max_source_length, ignore_index=None, reduction='none')

        # For sentence segmentation task.
        self.senseg_encoder = BertEncoderForSenseg(config)
        self.senseg_prediction_head = LayoutlmSensegPredictionHead(config)

        # self.crit_mask_senseg_smoothed = FocalLoss_(config.label_smoothing, len(config.senseg_task_ctg_to_id_map), ignore_index=None, reduction='none')

        self.crit_mask_senseg_smoothed = FocalLoss_(len(config.senseg_task_ctg_to_id_map), ignore_index=None, reduction='none')


        # Random initializing the weights, which may be overwritten by the pretrained model weights depend on the code logic.
        self.init_weights()


    @staticmethod
    def create_basic_mask(num_effective_tokens, max_seq_len):
        """
            num_effective_tokens: shape = [batch_size].
        """

        base_position_matrix = torch.arange(
            0, max_seq_len, dtype=num_effective_tokens.dtype, device=num_effective_tokens.device).view(1, -1) # [1, max_seq_len].
        mask = (base_position_matrix < (num_effective_tokens + 1).view(-1, 1)).type_as(num_effective_tokens) #  [batch_size, max_seq_len].
 
        return mask # shape = [batch_size, max_seq_len]

    @staticmethod
    def create_attention_mask(source_mask, target_mask, ):
        """
        inputs:
            source_mask: shape = [batch_size, max_src_len].
            target_mask: shape = [batch_size, max_tgt_len].

        returns:
            source_self_attn_mask: shape = [batch_size, max_src_len, max_src_len].
            target_self_attn_mask: shape = [batch_size, max_tgt_len, max_tgt_len].
            cross_attn_mask: shape = [batch_size, max_tgt_len, max_src_len].

        """
        
        # produce source_self_attn_mask.
        batch_size = source_mask.size(0)
        from_seq_len, to_seq_len = source_mask.size(1), source_mask.size(1)
        cls_token_attn_mask = torch.zeros((batch_size, from_seq_len, to_seq_len), dtype=source_mask.dtype, device=source_mask.device)
        cls_token_attn_mask[:, 0, :] = source_mask[:, :]
        effective_token_attn_mask = source_mask.unsqueeze(1).expand((batch_size, from_seq_len, to_seq_len))
        non_cls_pad_mask = source_mask.unsqueeze(2).expand((batch_size, from_seq_len, to_seq_len))
        effective_token_attn_mask = (effective_token_attn_mask & non_cls_pad_mask).type_as(source_mask)
        pad_token_attn_mask = torch.zeros((batch_size, from_seq_len, to_seq_len), dtype=source_mask.dtype, device=source_mask.device)
        ## produce the final source_self_attn_mask.
        source_self_attn_mask = (cls_token_attn_mask | effective_token_attn_mask | pad_token_attn_mask).type_as(source_mask)


        # produce target_self_attn_mask.
        batch_size = target_mask.size(0)
        from_seq_len, to_seq_len = target_mask.size(1), target_mask.size(1)
        sep_token_attn_mask = torch.zeros((batch_size, from_seq_len, to_seq_len), dtype=target_mask.dtype, device=target_mask.device)
        sep_token_attn_mask[:, 0, 0] = 1
        effective_token_attn_mask = torch.tril(torch.ones(from_seq_len, to_seq_len), diagonal=0).to(target_mask.dtype).to(target_mask.device).expand((batch_size, from_seq_len, to_seq_len))
        non_sep_pad_mask = target_mask.unsqueeze(2).expand((batch_size, from_seq_len, to_seq_len))
        effective_token_attn_mask = (effective_token_attn_mask & non_sep_pad_mask).type_as(target_mask)
        pad_token_attn_mask = torch.zeros((batch_size, from_seq_len, to_seq_len), dtype=target_mask.dtype, device=target_mask.device)
        ## produce the final target_self_attn_mask.
        target_self_attn_mask = (sep_token_attn_mask | effective_token_attn_mask | pad_token_attn_mask).type_as(target_mask)


        # produce cross_attn_mask.
        batch_size = target_mask.size(0)
        from_seq_len, to_seq_len = target_mask.size(1), source_mask.size(1)
        sep_token_attn_mask = torch.zeros((batch_size, from_seq_len, to_seq_len), dtype=source_mask.dtype, device=target_mask.device)
        sep_token_attn_mask[:, 0, :] = source_mask[:, :]
        effective_token_attn_mask = source_mask.unsqueeze(1).expand((batch_size, from_seq_len, to_seq_len))
        non_cls_pad_mask = target_mask.unsqueeze(2).expand((batch_size, from_seq_len, to_seq_len))
        effective_token_attn_mask = (effective_token_attn_mask & non_cls_pad_mask).type_as(source_mask)
        pad_token_attn_mask = torch.zeros((batch_size, from_seq_len, to_seq_len), dtype=source_mask.dtype, device=source_mask.device)
        ## produce the final cross_attn_mask.
        cross_attn_mask = (sep_token_attn_mask | effective_token_attn_mask | pad_token_attn_mask).type_as(source_mask)

        return source_self_attn_mask, target_self_attn_mask, cross_attn_mask

    @staticmethod
    def create_senseg_basic_mask(num_effective_tokens, max_seq_len):
        """
            num_effective_tokens: shape = [batch_size].
        """

        base_position_matrix = torch.arange(
            0, max_seq_len, dtype=num_effective_tokens.dtype, device=num_effective_tokens.device).view(1, -1) # [1, max_seq_len].
        mask = (base_position_matrix < num_effective_tokens.view(-1, 1)).type_as(num_effective_tokens) #  [batch_size, max_seq_len].
 
        return mask # shape = [batch_size, max_seq_len].

    @staticmethod
    def create_senseg_attention_mask(target_mask):
        """
        inputs:
            target_mask: shape = [batch_size, max_tgt_len].

        returns:
            target_self_attn_mask: shape = [batch_size, max_tgt_len, max_tgt_len].
        """

        # produce target_self_attn_mask.
        batch_size = target_mask.size(0)
        from_seq_len, to_seq_len = target_mask.size(1), target_mask.size(1)
        sep_token_attn_mask = torch.zeros((batch_size, from_seq_len, to_seq_len), dtype=target_mask.dtype, device=target_mask.device)
        sep_token_attn_mask[:, 0, :] = target_mask[:, :]
        effective_token_attn_mask = target_mask.unsqueeze(1).expand((batch_size, from_seq_len, to_seq_len))
        non_sep_pad_mask = target_mask.unsqueeze(2).expand((batch_size, from_seq_len, to_seq_len))
        effective_token_attn_mask = (effective_token_attn_mask & non_sep_pad_mask).type_as(target_mask)
        pad_token_attn_mask = torch.zeros((batch_size, from_seq_len, to_seq_len), dtype=target_mask.dtype, device=target_mask.device)
        ## produce the final source_self_attn_mask.
        target_self_attn_mask = (sep_token_attn_mask | effective_token_attn_mask | pad_token_attn_mask).type_as(target_mask)

        return target_self_attn_mask

    def forward(self, source_idxys, target_idxys, target_index, pseudo_target_idxys, num_effective_source_tokens, num_effective_target_tokens, senseg_target_labels):
        """
            source_idxys: shape = [batch_size, max_src_len, (4)].
            target_idxys: shape = [batch_size, max_tgt_len, (4)].
            target_index: shape = [batch_size, max_target_len].
            pseudo_target_idxys: shape = [batch_size, max_tgt_len, (4)].
            num_effective_source_tokens: shape = [batch_size].
            num_effective_target_tokens: shape = [batch_size].
            senseg_target_labels: shape = [batch_size, max_tgt_len].
        """

        # ************************* Word reordering task. *****************************

        max_src_len = source_idxys.size(1)
        max_tgt_len = target_idxys.size(1)
        max_pseudo_tgt_len = pseudo_target_idxys.size(1)
        assert max_tgt_len == max_pseudo_tgt_len
        assert max_src_len > 0 and max_tgt_len > 0

        if self.config.base_model_type == 'layoutlm':
            source_xys = source_idxys[:, :, 1:] # shape = [batch_size, max_src_len, 4].
            target_xys = target_idxys[:, :, 1:]
            pseudo_target_xys = pseudo_target_idxys[:, :, 1:]

            source_ids = source_idxys[:, :, 0] # shape = [batch_size, max_src_len].
            target_ids = target_idxys[:, :, 0]
            pseudo_target_ids = pseudo_target_idxys[:, :, 0]
        else:
            source_ids = source_idxys # shape = [batch_size, max_src_len].
            target_ids = target_idxys
            pseudo_target_ids = pseudo_target_idxys


        source_mask = self.create_basic_mask(num_effective_source_tokens, max_src_len) # shape = [batch_size, max_src_len], each element the value of 0/1, where 1 stands for "valid token" and 0 stands for "special token".
        target_mask = self.create_basic_mask(num_effective_target_tokens, max_tgt_len) # shape = [batch_size, max_tgt_len], each element the value of 0/1, where 1 stands for "valid token" and 0 stands for "special token".

        source_self_attention_mask, target_self_attention_mask, cross_attention_mask = \
            self.create_attention_mask(source_mask, target_mask)

        if self.config.base_model_type == 'layoutlm':
            # encoder
            encoder_outputs = self.encoder(input_ids=source_ids, bbox=source_xys, attention_mask=source_self_attention_mask, position_ids=None, inputs_embeds=None, return_emb=True)
            encoder_context, encoder_embedding_output = encoder_outputs[0], encoder_outputs[-1]
            # decoder
            decoder_outputs = self.decoder(encoder_context=encoder_context, decoder_input_ids=pseudo_target_ids, bbox=pseudo_target_xys, attention_mask=target_self_attention_mask, cross_attention_mask=cross_attention_mask, position_ids=None, inputs_embeds=None)
            decoder_hidden_states = decoder_outputs[0] # shape = [batch_size, max_tgt_len, hidden_dim].
        else:
            # encoder
            encoder_outputs = self.encoder(input_ids=source_ids,  attention_mask=source_self_attention_mask, position_ids=None, inputs_embeds=None, return_emb=True)
            encoder_context, encoder_embedding_output = encoder_outputs[0], encoder_outputs[-1]
            # decoder
            decoder_outputs = self.decoder(encoder_context=encoder_context, decoder_input_ids=pseudo_target_ids, attention_mask=target_self_attention_mask, cross_attention_mask=cross_attention_mask, position_ids=None, inputs_embeds=None)
            decoder_hidden_states = decoder_outputs[0] # shape = [batch_size, max_tgt_len, hidden_dim].
            
        # do the prediction and mask the prediction scores of the special tokens.
        prediction_scores = self.prediction_head(decoder_hidden_states, encoder_embedding_output) # shape: [batch_size, max_tgt_len, max_src_len].

        prediction_scores_masked = prediction_scores # Not leveraging the prior knowledge of num_effective_source_tokens.
        


        # loss calculation.
        loss_mask = target_mask[...]
        masked_loss = self.crit_mask_lm_smoothed(
                F.log_softmax(prediction_scores_masked.float(), dim=-1), target_index, num_effective_src_tokens=num_effective_source_tokens,
                loss_mask=loss_mask) # shape = [batch_size, max_tgt_len].
        

        # ************************* Sentence segmentation task. *****************************

        # get the senseg task self_attn_mask.
        senseg_target_mask = self.create_senseg_basic_mask(num_effective_target_tokens, max_tgt_len)
        senseg_attention_mask = self.create_senseg_attention_mask(senseg_target_mask)
        # print(senseg_attention_mask.shape)
        # print(decoder_hidden_states.shape)

        # forward.
        senseg_encoder_outputs = self.senseg_encoder(decoder_hidden_states, senseg_attention_mask)
        senseg_encoder_hidden_states = senseg_encoder_outputs[0] # shape = [batch_size, max_tgt_len, hidden_dim].

        senseg_prediction_scores = self.senseg_prediction_head(senseg_encoder_hidden_states) # shape = [batch_size, max_tgt_len, senseg_num_label].

        # loss calculation.
        senseg_loss_mask = senseg_target_mask.clone()
        senseg_masked_loss = self.crit_mask_senseg_smoothed(F.softmax(senseg_prediction_scores, dim=-1), senseg_target_labels, senseg_loss_mask)

        return {
            "reordering_task_loss": masked_loss,
            "senseg_task_loss": senseg_masked_loss,
        }

class LayoutlmReorderingSensegIncr(BertPreTrainedForSeq2SeqModel):

    def __init__(self, config):
        super(LayoutlmReorderingSensegIncr, self).__init__(config)

        self.config = config

        # For word reordering task.
        if config.base_model_type == 'layoutlm':
            self.encoder = LayoutlmModel(config)
            self.decoder = LayoutlmModelForDecoderIncr(config, self.encoder)
        else:
            self.encoder = BertModel(config)
            self.decoder = BertModelForDecoderIncr(config, self.encoder)

        self.prediction_head = LayoutlmSPOnlyMLMHead(config, src_len=config.max_source_length)
        
        self.crit_mask_lm_smoothed = CrossEntropyLoss_(config.label_smoothing, config.max_source_length, ignore_index=None, reduction='none')

        # For sentence segmentation task.
        self.senseg_encoder = BertEncoderForSenseg(config)
        self.senseg_prediction_head = LayoutlmSensegPredictionHead(config)

        # self.crit_mask_senseg_smoothed = FocalLoss_(config.label_smoothing, len(config.senseg_task_ctg_to_id_map), ignore_index=None, reduction='none')

        self.crit_mask_senseg_smoothed = FocalLoss_(len(config.senseg_task_ctg_to_id_map), ignore_index=None, reduction='none')


        # Random initializing the weights, which may be overwritten by the pretrained model weights depend on the code logic.
        self.init_weights()

    @staticmethod
    def create_basic_mask(num_effective_tokens, max_seq_len):
        """
            num_effective_tokens: shape = [batch_size].
        """

        base_position_matrix = torch.arange(
            0, max_seq_len, dtype=num_effective_tokens.dtype, device=num_effective_tokens.device).view(1, -1) # [1, max_seq_len].
        mask = (base_position_matrix < (num_effective_tokens + 1).view(-1, 1)).type_as(num_effective_tokens) #  [batch_size, max_seq_len].
 
        return mask # shape = [batch_size, max_seq_len]

    @staticmethod
    def create_attention_mask(source_mask, target_mask, ):
        """
        inputs:
            source_mask: shape = [batch_size, max_src_len].
            target_mask: shape = [batch_size, max_tgt_len].

        returns:
            source_self_attn_mask: shape = [batch_size, max_src_len, max_src_len].
            target_self_attn_mask: shape = [batch_size, max_tgt_len, max_tgt_len].
            cross_attn_mask: shape = [batch_size, max_tgt_len, max_src_len].

        """
        
        # produce source_self_attn_mask.
        batch_size = source_mask.size(0)
        from_seq_len, to_seq_len = source_mask.size(1), source_mask.size(1)
        cls_token_attn_mask = torch.zeros((batch_size, from_seq_len, to_seq_len), dtype=source_mask.dtype, device=source_mask.device)
        cls_token_attn_mask[:, 0, :] = source_mask[:, :]
        effective_token_attn_mask = source_mask.unsqueeze(1).expand((batch_size, from_seq_len, to_seq_len))
        non_cls_pad_mask = source_mask.unsqueeze(2).expand((batch_size, from_seq_len, to_seq_len))
        effective_token_attn_mask = (effective_token_attn_mask & non_cls_pad_mask).type_as(source_mask)
        pad_token_attn_mask = torch.zeros((batch_size, from_seq_len, to_seq_len), dtype=source_mask.dtype, device=source_mask.device)
        ## produce the final source_self_attn_mask.
        source_self_attn_mask = (cls_token_attn_mask | effective_token_attn_mask | pad_token_attn_mask).type_as(source_mask)


        # produce target_self_attn_mask.
        batch_size = target_mask.size(0)
        from_seq_len, to_seq_len = target_mask.size(1), target_mask.size(1)
        sep_token_attn_mask = torch.zeros((batch_size, from_seq_len, to_seq_len), dtype=target_mask.dtype, device=target_mask.device)
        sep_token_attn_mask[:, 0, 0] = 1
        effective_token_attn_mask = torch.tril(torch.ones(from_seq_len, to_seq_len), diagonal=0).to(target_mask.dtype).to(target_mask.device).expand((batch_size, from_seq_len, to_seq_len))
        non_sep_pad_mask = target_mask.unsqueeze(2).expand((batch_size, from_seq_len, to_seq_len))
        effective_token_attn_mask = (effective_token_attn_mask & non_sep_pad_mask).type_as(target_mask)
        pad_token_attn_mask = torch.zeros((batch_size, from_seq_len, to_seq_len), dtype=target_mask.dtype, device=target_mask.device)
        ## produce the final target_self_attn_mask.
        target_self_attn_mask = (sep_token_attn_mask | effective_token_attn_mask | pad_token_attn_mask).type_as(target_mask)


        # produce cross_attn_mask.
        batch_size = target_mask.size(0)
        from_seq_len, to_seq_len = target_mask.size(1), source_mask.size(1)
        sep_token_attn_mask = torch.zeros((batch_size, from_seq_len, to_seq_len), dtype=source_mask.dtype, device=target_mask.device)
        sep_token_attn_mask[:, 0, :] = source_mask[:, :]
        effective_token_attn_mask = source_mask.unsqueeze(1).expand((batch_size, from_seq_len, to_seq_len))
        non_cls_pad_mask = target_mask.unsqueeze(2).expand((batch_size, from_seq_len, to_seq_len))
        effective_token_attn_mask = (effective_token_attn_mask & non_cls_pad_mask).type_as(source_mask)
        pad_token_attn_mask = torch.zeros((batch_size, from_seq_len, to_seq_len), dtype=source_mask.dtype, device=source_mask.device)
        ## produce the final cross_attn_mask.
        cross_attn_mask = (sep_token_attn_mask | effective_token_attn_mask | pad_token_attn_mask).type_as(source_mask)

        return source_self_attn_mask, target_self_attn_mask, cross_attn_mask

    @staticmethod
    def create_senseg_basic_mask(num_effective_tokens, max_seq_len):
        """
            num_effective_tokens: shape = [batch_size].
        """

        base_position_matrix = torch.arange(
            0, max_seq_len, dtype=num_effective_tokens.dtype, device=num_effective_tokens.device).view(1, -1) # [1, max_seq_len].
        mask = (base_position_matrix < num_effective_tokens.view(-1, 1)).type_as(num_effective_tokens) #  [batch_size, max_seq_len].
 
        return mask # shape = [batch_size, max_seq_len].

    @staticmethod
    def create_senseg_attention_mask(target_mask):
        """
        inputs:
            target_mask: shape = [batch_size, max_tgt_len].

        returns:
            target_self_attn_mask: shape = [batch_size, max_tgt_len, max_tgt_len].
        """

        # produce target_self_attn_mask.
        batch_size = target_mask.size(0)
        from_seq_len, to_seq_len = target_mask.size(1), target_mask.size(1)
        sep_token_attn_mask = torch.zeros((batch_size, from_seq_len, to_seq_len), dtype=target_mask.dtype, device=target_mask.device)
        sep_token_attn_mask[:, 0, :] = target_mask[:, :]
        effective_token_attn_mask = target_mask.unsqueeze(1).expand((batch_size, from_seq_len, to_seq_len))
        non_sep_pad_mask = target_mask.unsqueeze(2).expand((batch_size, from_seq_len, to_seq_len))
        effective_token_attn_mask = (effective_token_attn_mask & non_sep_pad_mask).type_as(target_mask)
        pad_token_attn_mask = torch.zeros((batch_size, from_seq_len, to_seq_len), dtype=target_mask.dtype, device=target_mask.device)
        ## produce the final source_self_attn_mask.
        target_self_attn_mask = (sep_token_attn_mask | effective_token_attn_mask | pad_token_attn_mask).type_as(target_mask)

        return target_self_attn_mask


    def greedy_decode(self, source_idxys, num_effective_source_tokens, tokenizer):
        """
            source_idxys: shape = [batch_size, max_src_len, (5)].
            num_effective_source_tokens: shape = [batch_size].
        """

        # ************************* Word reordering task. *****************************

        batch_size = source_idxys.size(0)
        max_src_len = source_idxys.size(1)
        # assert max_src_len > 0

        max_source_tokens_cur_batch = int(torch.max(num_effective_source_tokens))

        if self.config.base_model_type == 'layoutlm':
            source_xys = source_idxys[:, :, 1:] # shape = [batch_size, max_src_len, 4].
            source_ids = source_idxys[:, :, 0] # shape = [batch_size, max_src_len].

            # construct the sep token_id and layout for the "start" of decoding.
            target_xys = torch.tensor(predefined_constant.SEP_TOKEN_LAYOUT, dtype=source_xys.dtype, device=source_xys.device).repeat((batch_size, 1, 1)) # shape = [batch_size, 1, 4].
            target_ids = torch.tensor(tokenizer.sep_token_id, dtype=source_ids.dtype, device=source_ids.device).repeat((batch_size, 1)) # shape = [batch_size, 1].
        else:
            source_ids = source_idxys # shape = [batch_size, max_src_len].

            target_ids = torch.tensor(tokenizer.sep_token_id, dtype=source_ids.dtype, device=source_ids.device).repeat((batch_size, 1)) # shape = [batch_size, 1].

        # 构建好basic_mask和attn_mask.
        source_mask = self.create_basic_mask(num_effective_source_tokens, max_src_len) # shape = [batch_size, max_src_len].
        target_mask = torch.ones((batch_size, max_source_tokens_cur_batch+1), dtype=source_mask.dtype, device=source_mask.device) # shape = [batch_size, max_source_tokens_cur_batch].

        source_self_attention_mask, target_self_attention_mask, cross_attention_mask = \
            self.create_attention_mask(source_mask, target_mask)
        
        # get the encoder context and embedding output.
        encoder_inputs = {
            "input_ids": source_ids, 
            "attention_mask": source_self_attention_mask,
            "position_ids": None,
            "inputs_embeds": None,
            "return_emb": True,
        }
        if self.config.base_model_type == 'layoutlm':
            encoder_inputs["bbox"] = source_xys
        
        encoder_outputs = self.encoder(**encoder_inputs)
        encoder_context, encoder_embedding_output = encoder_outputs[0], encoder_outputs[-1]

        # prepare the source_mask_for_prection_scores.
        source_mask_for_prection_scores = source_mask.clone()
        # source_mask_for_prection_scores[:, 0] = 1
        
        
        prediction_scores_mask = ((1.0 - source_mask_for_prection_scores.unsqueeze(1)) * -1e5) # shape = [batch_size, 1, max_src_len].
        # autoregressively decoding the target seq.
        decoded_res = [] # store the decoded results. each item is a tensor of shape = [batch_size, 1], corresponding the the decoded token id at that time step.

        decoding_step = 0

        cur_input_ids = target_ids.clone()
        # print(f"cur_input_ids: {cur_input_ids}")
        if self.config.base_model_type == 'layoutlm':
            cur_input_xys = target_xys.clone()

        decoder_all_layers_historical_hidden_states = ()
        while decoding_step < max_source_tokens_cur_batch + 1:

            # prepare decoder inputs of current decoding step.
            cur_self_attention_mask = target_self_attention_mask[:, decoding_step:(decoding_step+1), :(decoding_step+1)] # shape = [batch_size, 1, decoding_step+1].
            cur_cross_attention_mask = cross_attention_mask[:, decoding_step:(decoding_step+1), :] # shape = [batch_size, 1, max_src_len].


            decoder_inputs = {
                "encoder_context": encoder_context,
                "decoder_all_layers_historical_hidden_states": decoder_all_layers_historical_hidden_states,
                "decoder_cur_time_input_ids": cur_input_ids,
                "attention_mask": cur_self_attention_mask,
                "cross_attention_mask": cur_cross_attention_mask,
                "position_ids": None,
                "inputs_embeds": None,
            }
            if self.config.base_model_type == 'layoutlm':
                decoder_inputs["bbox"] = cur_input_xys

            # forward.
            decoder_all_layers_accu_hidden_states = self.decoder(**decoder_inputs)[0]
            decoder_last_layer_hidden_states = decoder_all_layers_accu_hidden_states[-1] # shape = [batch_size, decoding_step+1, hidden_dim].
            # prediction.
            cur_decoder_hidden_states = decoder_last_layer_hidden_states[:, -1:, :] # shape = [batch_size, 1, hidden_dim].
            prediction_scores = self.prediction_head(cur_decoder_hidden_states, encoder_embedding_output) # shape = [batch_size, 1, max_src_len].
            # prediction_scores_masked = prediction_scores + prediction_scores_mask # shape = [batch_size, 1, max_src_len]. Leveraging the prior knowledge of num_effective_source_tokens.
            prediction_scores_masked = prediction_scores # Not leveraging the prior knowledge of num_effective_source_tokens.
            _, max_ids = torch.max(prediction_scores_masked, dim=-1) # shape = [batch_size, 1].
            # print(f"prediction_scores_masked: {prediction_scores_masked}")

            # get the decoded token id (and layout) from the source seq, store the decoded token id, accumulate the decoder input ids and input layouts, and update the cur_input_ids/cur_input_xys for next decoding step.
            cur_decoded_ids = torch.gather(source_ids, 1, max_ids) # shape = [batch_size, 1].
            decoded_res.append(cur_decoded_ids)
            # accu_decoder_input_ids = torch.cat((accu_decoder_input_ids, cur_decoded_ids), dim=1) 
            cur_input_ids = cur_decoded_ids
            decoder_all_layers_historical_hidden_states = decoder_all_layers_accu_hidden_states
            if  self.config.base_model_type == 'layoutlm':
                _, _, layout_dim = source_xys.shape
                max_ids_ = max_ids.unsqueeze(-1).expand(max_ids.size(0), max_ids.size(1), layout_dim) # shape = [batch_size, 1, 4].
                cur_decoded_xys = torch.gather(source_xys, 1, max_ids_) # shape = [batch_size, 1, 4].
                # accu_decoder_input_xys = torch.cat((accu_decoder_input_xys, cur_decoded_xys), dim=1)
                cur_input_xys = cur_decoded_xys

            decoding_step += 1

        reordering_task_decoding_res = torch.cat(decoded_res, dim=1) # shape = [batch_size, max_source_tokens_cur_batch + 1].


        # ************************* Sentence segmentation task. *****************************

        # get the word reordering task's decoder hidden_states.
        decoder_last_layer_accu_hidden_states = decoder_all_layers_accu_hidden_states[-1] # shape = [batch_size, max_source_tokens_cur_batch+1, hidden_dim].

        # get the senseg task self_attn_mask.
        senseg_target_mask = self.create_senseg_basic_mask(num_effective_source_tokens, max_source_tokens_cur_batch+1)
        senseg_attention_mask = self.create_senseg_attention_mask(senseg_target_mask)

        # forward.
        senseg_encoder_outputs = self.senseg_encoder(decoder_last_layer_accu_hidden_states, senseg_attention_mask)
        senseg_encoder_hidden_states = senseg_encoder_outputs[0] # shape = [batch_size, max_source_tokens_cur_batch+1, hidden_dim].

        senseg_prediction_scores = self.senseg_prediction_head(senseg_encoder_hidden_states) # shape = [batch_size, max_source_tokens_cur_batch+1, senseg_num_label].
        _, senseg_max_ids = torch.max(senseg_prediction_scores, dim=-1) # shape = [batch_size, max_source_tokens_cur_batch+1].

        senseg_task_prediction_res = senseg_max_ids

        return {
            "reordering_task_res": reordering_task_decoding_res,
            "senseg_task_res": senseg_task_prediction_res,
        }
    


 
# *********************** Adding the translation task. *******************************************

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return x

class BertEmbeddingsForTrans(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super(BertEmbeddingsForTrans, self).__init__()
        self.word_embeddings = nn.Embedding(config.tgt_vocab_size, config.hidden_size)
        # self.position_embeddings = nn.Embedding(config.tgt_max_position_embeddings, config.hidden_size)
        self.position_encodings = PositionalEncoding(config.hidden_size, config.tgt_max_position_embeddings)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, position_ids=None, inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids) # input_embeds shape = [batch_size, seq_len, hidden_dim].
        # position_embeddings = self.position_embeddings(position_ids)
        

        # embeddings = inputs_embeds + position_embeddings
        
        # add position encoding.
        embeddings = self.position_encodings(inputs_embeds.permute(1, 0, 2)) # # shape = [seq_len, batch_size, hidden_dim].
        embeddings = embeddings.permute(1, 0, 2) # shape = [batch_size, seq_len, hidden_dim].

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings # shape = [batch_size, max_src_len, embedding_dim].
    
class BertDecoderForTrans(nn.Module):
    def __init__(self, config):
        super(BertDecoderForTrans, self).__init__()
        self.output_attentions = config.output_attentions
        self.layer = nn.ModuleList([BertLayerForDecoder(config) for _ in range(config.trans_decoder_num_hidden_layers)])

    def forward(self, encoder_context, decoder_hidden_states, cross_attention_mask, attention_mask):

        all_decoder_self_attentions = ()
        all_decoder_cross_attentions = ()

        for i, layer_module in enumerate(self.layer):

            layer_outputs = layer_module(encoder_context, decoder_hidden_states, cross_attention_mask, attention_mask)
            if self.output_attentions:
                decoder_hidden_states, decoder_self_attentions, decoder_cross_attentions = layer_outputs
            else:
                decoder_hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_decoder_self_attentions = all_decoder_self_attentions + (decoder_self_attentions, )
                all_decoder_cross_attentions = all_decoder_cross_attentions + (decoder_cross_attentions, )

        outputs = (decoder_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_decoder_self_attentions,) + (all_decoder_cross_attentions,)
        return outputs  # last-layer hidden state, (all_decoder_self_attentions), (all_decoder_cross_attentions)

class BertDecoderForTransIncr(nn.Module):
    def __init__(self, config):
        super(BertDecoderForTransIncr, self).__init__()
        self.layer = nn.ModuleList([BertLayerForDecoderIncr(config) for _ in range(config.trans_decoder_num_hidden_layers)])

    def forward(self, encoder_context, decoder_all_layers_historical_hidden_states, decoder_cur_time_hidden_states, cross_attention_mask, attention_mask):
        """
            encoder_context: shape = [batch_size, max_src_len, hidden_dim].
            decoder_all_layers_historical_hidden_states: a tuple containing: first decoder embedding output, then layer_0 - layer_(n-1) historical hidden_states, each element is of the shape = [batch_size, historical_decoding_steps, hidden_dim]. or () for the first decoding step.
            decoder_cur_time_hidden_states: shape = [batch_size, 1, hidden_dim].
            cross_attention_mask: shape = [batch_size, 1, max_src_len].
            attention_mask: shape = [batch_size, 1, accu_decoding_steps].
        """

        decoder_all_layers_cur_time_hidden_states = (decoder_cur_time_hidden_states, )

        if decoder_all_layers_historical_hidden_states:
            decoder_all_layers_accu_hidden_states = (torch.cat((decoder_all_layers_historical_hidden_states[0], decoder_cur_time_hidden_states), dim=1), )
        else:
            decoder_all_layers_accu_hidden_states = (decoder_cur_time_hidden_states, )

        for i, layer_module in enumerate(self.layer):
            if decoder_all_layers_historical_hidden_states:
                layer_decoder_historical_hidden_states = decoder_all_layers_historical_hidden_states[i]
                layer_decoder_accu_hidden_states = torch.cat(
                    (layer_decoder_historical_hidden_states, decoder_cur_time_hidden_states), dim=1
                )
            else:
                layer_decoder_accu_hidden_states = decoder_cur_time_hidden_states
            
            
            layer_outputs = layer_module(
                    encoder_context,
                    layer_decoder_accu_hidden_states,
                    decoder_cur_time_hidden_states,
                    cross_attention_mask,
                    attention_mask,
                )
            decoder_cur_time_hidden_states = layer_outputs[0]

            decoder_all_layers_cur_time_hidden_states += (decoder_cur_time_hidden_states, )


        if decoder_all_layers_historical_hidden_states:
            decoder_all_layers_accu_hidden_states = ()
            for historical_hidden_states, cur_time_hidden_states in zip(decoder_all_layers_historical_hidden_states, decoder_all_layers_cur_time_hidden_states):
                decoder_all_layers_accu_hidden_states += (torch.cat((historical_hidden_states, cur_time_hidden_states), dim=1), )
        else:
            decoder_all_layers_accu_hidden_states = decoder_all_layers_cur_time_hidden_states
        
        return (decoder_all_layers_accu_hidden_states, )
        # first the decoder_embedding_output, then decoder layer_0 - layer_(n-1)'s hidden_states, accumulated up to the current decoding step.
        # Each element is of the shape = [batch_size, accu_decoding_steps, hidden_dim].

class BertModelForDecoderForTrans(BertPreTrainedForSeq2SeqModel):

    def __init__(self, config):
        super(BertModelForDecoderForTrans, self).__init__(config)
        self.config = config

        self.embeddings = BertEmbeddingsForTrans(config)
        # self.embeddings = BertEmbeddings(config)
        self.decoder = BertDecoderForTrans(config)

    def forward(self, encoder_context, decoder_input_ids, attention_mask, cross_attention_mask, 
                position_ids=None, inputs_embeds=None):
        if decoder_input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif decoder_input_ids is not None:
            input_shape = decoder_input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = decoder_input_ids.device if decoder_input_ids is not None else inputs_embeds.device

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]

        # Provided a padding mask of dimensions [batch_size, seq_length]
        # - if the model is a decoder, apply a causal mask in addition to the padding mask
        # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # for cross_attention_mask.
        if cross_attention_mask.dim() == 3:
            extended_cross_attention_mask = cross_attention_mask[:, None, :, :]
        if cross_attention_mask.dim() == 2:
            extended_cross_attention_mask = cross_attention_mask[:, None, None, :]
        extended_cross_attention_mask = extended_cross_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_cross_attention_mask = (1.0 - extended_cross_attention_mask) * -10000.0


        embedding_output = self.embeddings(
            input_ids=decoder_input_ids, position_ids=position_ids, inputs_embeds=inputs_embeds)
        decoder_outputs = self.decoder(
            encoder_context, embedding_output, extended_cross_attention_mask, extended_attention_mask)
        sequence_output = decoder_outputs[0]

        outputs = (sequence_output, ) + decoder_outputs[1:]  # add hidden_states and attentions if they are here. last-layer hidden state, (all_decoder_self_attentions), (all_decoder_cross_attentions)

        return outputs  # last-layer hidden_state, (all_decoder_self_attentions), (all_decoder_cross_attentions).
    
    def prepare_inputs_for_generation(self, input_ids, **kwargs):

        kwargs["decoder_input_ids"] = input_ids

        return kwargs



class TransPredictionHead(nn.Module):
    def __init__(self, config, embedding_layer):
        super(TransPredictionHead, self).__init__()

        self.transform = BertPredictionHeadTransform(config)

        # embedding_layer_weight_data = embedding_layer.weight.data # shape = [vocab_size, hidden_dim].
        # Linear_layer_weight_data = embedding_layer_weight_data.permute(1, 0) # shape = [hidden_dim, vocab_size].
        # Linear_layer_weight_data = embedding_layer_weight_data # shape = [vocab, hidden_dim].

        self.prediction_head = nn.Linear(config.hidden_size, config.tgt_vocab_size, bias=True)
        # self.prediction_head.weight = nn.Parameter(Linear_layer_weight_data)
        self.prediction_head.weight = embedding_layer.weight

    def forward(self, hidden_states):
        """
            hidden_states: shape = [batch_size, max_tgt_len, hidden_dim].
        """

        hidden_states = self.transform(hidden_states) # shape = [batch_size, max_tgt_len, hidden_dim].
        prediction_scores = self.prediction_head(hidden_states) # shape = [batch_size, max_tgt_len, vocab_size].

        return prediction_scores # shape = [batch_size, max_tgt_len, vocab_size].


class LayoutlmReorderingSensegTrans(BertPreTrainedForSeq2SeqModel):

    def __init__(self, config):
        super(LayoutlmReorderingSensegTrans, self).__init__(config)

        self.config = config

        # For word reordering task.
        if config.base_model_type == 'layoutlm':
            self.encoder = LayoutlmModel(config)
            self.decoder = LayoutlmModelForDecoder(config, self.encoder)
        else:
            self.encoder = BertModel(config)
            self.decoder = BertModelForDecoder(config, self.encoder)

        self.prediction_head = LayoutlmSPOnlyMLMHead(config, src_len=config.max_source_length)
        
        self.crit_mask_lm_smoothed = CrossEntropyLoss_(config.label_smoothing, config.max_source_length, ignore_index=None, reduction='none')

        # For sentence segmentation task.
        self.senseg_encoder = BertEncoderForSenseg(config)
        self.senseg_prediction_head = LayoutlmSensegPredictionHead(config)

        # self.crit_mask_senseg_smoothed = FocalLoss_(config.label_smoothing, len(config.senseg_task_ctg_to_id_map), ignore_index=None, reduction='none')

        self.crit_mask_senseg_smoothed = FocalLoss_(len(config.senseg_task_ctg_to_id_map), ignore_index=None, reduction='none')


        # For translation task.
        self.trans_decoder = BertModelForDecoderForTrans(config)
        self.trans_prediction_head = TransPredictionHead(config, self.trans_decoder.embeddings.word_embeddings)

        self.crit_mask_trans_smoothed = CrossEntropyLossForTrans_(config.label_smoothing, config.tgt_vocab_size, ignore_index=None, reduction='none')

        self.trans_decoder_max_fwd_tokens = self.config.trans_decoder_max_fwd_tokens


        # Random initializing the weights, which may be overwritten by the pretrained model weights depend on the code logic.
        self.init_weights()


    @staticmethod
    def create_basic_mask(num_effective_tokens, max_seq_len):
        """
            num_effective_tokens: shape = [batch_size].
        """

        base_position_matrix = torch.arange(
            0, max_seq_len, dtype=num_effective_tokens.dtype, device=num_effective_tokens.device).view(1, -1) # [1, max_seq_len].
        mask = (base_position_matrix < (num_effective_tokens + 1).view(-1, 1)).type_as(num_effective_tokens) #  [batch_size, max_seq_len].
 
        return mask # shape = [batch_size, max_seq_len]

    @staticmethod
    def create_attention_mask(source_mask, target_mask, ):
        """
        inputs:
            source_mask: shape = [batch_size, max_src_len].
            target_mask: shape = [batch_size, max_tgt_len].

        returns:
            source_self_attn_mask: shape = [batch_size, max_src_len, max_src_len].
            target_self_attn_mask: shape = [batch_size, max_tgt_len, max_tgt_len].
            cross_attn_mask: shape = [batch_size, max_tgt_len, max_src_len].

        """
        
        # produce source_self_attn_mask.
        batch_size = source_mask.size(0)
        from_seq_len, to_seq_len = source_mask.size(1), source_mask.size(1)
        cls_token_attn_mask = torch.zeros((batch_size, from_seq_len, to_seq_len), dtype=source_mask.dtype, device=source_mask.device)
        cls_token_attn_mask[:, 0, :] = source_mask[:, :]
        effective_token_attn_mask = source_mask.unsqueeze(1).expand((batch_size, from_seq_len, to_seq_len))
        non_cls_pad_mask = source_mask.unsqueeze(2).expand((batch_size, from_seq_len, to_seq_len))
        effective_token_attn_mask = (effective_token_attn_mask & non_cls_pad_mask).type_as(source_mask)
        # pad_token_attn_mask = torch.zeros((batch_size, from_seq_len, to_seq_len), dtype=source_mask.dtype, device=source_mask.device)
        ## produce the final source_self_attn_mask.
        source_self_attn_mask = (cls_token_attn_mask | effective_token_attn_mask).type_as(source_mask)


        # produce target_self_attn_mask.
        batch_size = target_mask.size(0)
        from_seq_len, to_seq_len = target_mask.size(1), target_mask.size(1)
        ## produce sep_token_attn_mask. sep_token能attend到sep_token.
        sep_token_attn_mask = torch.zeros((batch_size, from_seq_len, to_seq_len), dtype=target_mask.dtype, device=target_mask.device)
        sep_token_attn_mask[:, 0, 0] = 1
        effective_token_attn_mask = torch.tril(torch.ones(from_seq_len, to_seq_len), diagonal=0).to(target_mask.dtype).to(target_mask.device).expand((batch_size, from_seq_len, to_seq_len))
        non_sep_pad_mask = target_mask.unsqueeze(2).expand((batch_size, from_seq_len, to_seq_len))
        effective_token_attn_mask = (effective_token_attn_mask & non_sep_pad_mask).type_as(target_mask)
        # pad_token_attn_mask = torch.zeros((batch_size, from_seq_len, to_seq_len), dtype=target_mask.dtype, device=target_mask.device)
        ## produce the final target_self_attn_mask.
        target_self_attn_mask = (sep_token_attn_mask | effective_token_attn_mask).type_as(target_mask)


        # produce cross_attn_mask.
        batch_size = target_mask.size(0)
        from_seq_len, to_seq_len = target_mask.size(1), source_mask.size(1)
        sep_token_attn_mask = torch.zeros((batch_size, from_seq_len, to_seq_len), dtype=source_mask.dtype, device=target_mask.device)
        sep_token_attn_mask[:, 0, :] = source_mask[:, :]
        effective_token_attn_mask = source_mask.unsqueeze(1).expand((batch_size, from_seq_len, to_seq_len))
        non_cls_pad_mask = target_mask.unsqueeze(2).expand((batch_size, from_seq_len, to_seq_len))
        effective_token_attn_mask = (effective_token_attn_mask & non_cls_pad_mask).type_as(source_mask)
        # pad_token_attn_mask = torch.zeros((batch_size, from_seq_len, to_seq_len), dtype=source_mask.dtype, device=source_mask.device)
        ## produce the final cross_attn_mask.
        cross_attn_mask = (sep_token_attn_mask | effective_token_attn_mask).type_as(source_mask)

        return source_self_attn_mask, target_self_attn_mask, cross_attn_mask

    @staticmethod
    def create_senseg_basic_mask(num_effective_tokens, max_seq_len):
        """
            num_effective_tokens: shape = [batch_size].
        """

        base_position_matrix = torch.arange(
            0, max_seq_len, dtype=num_effective_tokens.dtype, device=num_effective_tokens.device).view(1, -1) # [1, max_seq_len].
        mask = (base_position_matrix < num_effective_tokens.view(-1, 1)).type_as(num_effective_tokens) #  [batch_size, max_seq_len].
 
        return mask # shape = [batch_size, max_seq_len].

    @staticmethod
    def create_senseg_attention_mask(target_mask):
        """
        inputs:
            target_mask: shape = [batch_size, max_tgt_len].

        returns:
            target_self_attn_mask: shape = [batch_size, max_tgt_len, max_tgt_len].
        """

        # produce target_self_attn_mask.
        batch_size = target_mask.size(0)
        from_seq_len, to_seq_len = target_mask.size(1), target_mask.size(1)
        sep_token_attn_mask = torch.zeros((batch_size, from_seq_len, to_seq_len), dtype=target_mask.dtype, device=target_mask.device)
        sep_token_attn_mask[:, 0, :] = target_mask[:, :]
        effective_token_attn_mask = target_mask.unsqueeze(1).expand((batch_size, from_seq_len, to_seq_len))
        non_sep_pad_mask = target_mask.unsqueeze(2).expand((batch_size, from_seq_len, to_seq_len))
        effective_token_attn_mask = (effective_token_attn_mask & non_sep_pad_mask).type_as(target_mask)
        # pad_token_attn_mask = torch.zeros((batch_size, from_seq_len, to_seq_len), dtype=target_mask.dtype, device=target_mask.device)
        ## produce the final source_self_attn_mask.
        target_self_attn_mask = (sep_token_attn_mask | effective_token_attn_mask).type_as(target_mask)

        return target_self_attn_mask


    @staticmethod
    def create_trans_basic_src_mask(num_effective_tokens, src_sen_max_len):
        """
            num_effective_tokens: shape = [batch_size_sen_level].
        """

        base_position_matrix = torch.arange(
            0, src_sen_max_len, dtype=num_effective_tokens.dtype, device=num_effective_tokens.device).view(1, -1) # [1, src_sen_max_len].
        mask = (base_position_matrix < num_effective_tokens.view(-1, 1)).type_as(num_effective_tokens) #  [batch_size, src_sen_max_len].
 
        return mask # shape = [batch_size_sen_level, src_sen_max_len].

    @staticmethod
    def create_trans_basic_tgt_mask(num_effective_tokens, tgt_sen_max_len):
        """
            num_effective_tokens: shape = [batch_size_sen_level].
        """

        base_position_matrix = torch.arange(
            0, tgt_sen_max_len, dtype=num_effective_tokens.dtype, device=num_effective_tokens.device).view(1, -1) # [1, tgt_sen_max_len].
        mask = (base_position_matrix < (num_effective_tokens + 1).view(-1, 1)).type_as(num_effective_tokens) #  [batch_size_sen_level, tgt_sen_max_len].
 
        return mask # shape = [batch_size_sen_level, tgt_sen_max_len]

    @staticmethod
    def create_trans_attention_mask(source_mask, target_mask, ):
        """
        inputs:
            source_mask: shape = [batch_size_sen_level, src_sen_max_len].
            target_mask: shape = [batch_size_sen_level, tgt_sen_max_len].

        returns:
            target_self_attn_mask: shape = [batch_size_sen_level, tgt_sen_max_len, tgt_sen_max_len].
            cross_attn_mask: shape = [batch_size_sen_level, tgt_sen_max_len, src_sen_max_len].

        """

        # produce target_self_attn_mask.
        batch_size = target_mask.size(0)
        from_seq_len, to_seq_len = target_mask.size(1), target_mask.size(1)
        sep_token_attn_mask = torch.zeros((batch_size, from_seq_len, to_seq_len), dtype=target_mask.dtype, device=target_mask.device)
        sep_token_attn_mask[:, 0, 0] = 1
        effective_token_attn_mask = torch.tril(torch.ones(from_seq_len, to_seq_len), diagonal=0).to(target_mask.dtype).to(target_mask.device).expand((batch_size, from_seq_len, to_seq_len))
        non_sep_pad_mask = target_mask.unsqueeze(2).expand((batch_size, from_seq_len, to_seq_len))
        effective_token_attn_mask = (effective_token_attn_mask & non_sep_pad_mask).type_as(target_mask)
        # pad_token_attn_mask = torch.zeros((batch_size, from_seq_len, to_seq_len), dtype=target_mask.dtype, device=target_mask.device)
        ## produce the final target_self_attn_mask.
        target_self_attn_mask = (sep_token_attn_mask | effective_token_attn_mask).type_as(target_mask)


        # produce cross_attn_mask.
        batch_size = target_mask.size(0)
        from_seq_len, to_seq_len = target_mask.size(1), source_mask.size(1)
        sep_token_attn_mask = torch.zeros((batch_size, from_seq_len, to_seq_len), dtype=source_mask.dtype, device=target_mask.device)
        sep_token_attn_mask[:, 0, :] = source_mask[:, :]
        effective_token_attn_mask = source_mask.unsqueeze(1).expand((batch_size, from_seq_len, to_seq_len))
        non_cls_pad_mask = target_mask.unsqueeze(2).expand((batch_size, from_seq_len, to_seq_len))
        effective_token_attn_mask = (effective_token_attn_mask & non_cls_pad_mask).type_as(source_mask)
        # pad_token_attn_mask = torch.zeros((batch_size, from_seq_len, to_seq_len), dtype=source_mask.dtype, device=source_mask.device)
        ## produce the final cross_attn_mask.
        cross_attn_mask = (sep_token_attn_mask | effective_token_attn_mask).type_as(source_mask)

        return target_self_attn_mask, cross_attn_mask  


    def create_src_sen_context_from_decoder_hidden_states(self, senseg_target_labels, num_effective_target_tokens, decoder_hidden_states, num_effective_src_sen_tokens):
        """
            senseg_target_labels: shape = [batch_size, max_tgt_len].
            num_effective_target_tokens: shape = [batch_size].
            decoder_hidden_states: shape = [batch_size, max_tgt_len, hidden_dim].
            num_effective_src_sen_tokens: shape = [batch_size_sen_level].
        """

        # batch_size_sen_level = senseg_target_labels.numel() - senseg_target_labels.sum().item()
        batch_size_sen_level = num_effective_src_sen_tokens.shape[0]

        # # assert num_effective_src_sen_tokens_caled == num_effective_src_sen_tokens.tolist()
        src_sen_max_len_cur_batch = max(num_effective_src_sen_tokens)


        # pad_embedding = src_embedding_layer(torch.tensor(src_pad_token_id, dtype=senseg_target_labels.dtype, device=senseg_target_labels.device))
        # src_sen_context = pad_embedding.repeat((batch_size_sen_level, src_sen_max_len_cur_batch, 1)) # shape = [batch_size_sen_level, src_sen_max_len_cur_batch, hidden_dim].

        # src_sen_context_idx = torch.ones((batch_size_sen_level, src_sen_max_len_cur_batch), dtype=senseg_target_labels.dtype, device=senseg_target_labels.device)
        src_sen_context_list = [None] * batch_size_sen_level   # shape = [batch_size_sen_level, src_sen_max_len_cur_batch, hidden_dim].

        num_sen = 0
        for num_instance, instance_labels in enumerate(senseg_target_labels):
            num_effective_target_token = num_effective_target_tokens[num_instance].item()
            bos_token_index = torch.nonzero(instance_labels[:num_effective_target_token] == 0, as_tuple=True)[0]
            shifted_bos_token_index = torch.cat((bos_token_index[1:], torch.tensor([num_effective_target_token], dtype=bos_token_index.dtype, device=bos_token_index.device)), dim=-1)
            # print(bos_token_index)
            # print(shifted_bos_token_index)

            for bos_idx, eos_idx in zip(bos_token_index, shifted_bos_token_index):
                # print(bos_idx, eos_idx)
                if eos_idx - bos_idx > src_sen_max_len_cur_batch:
                    margin = (eos_idx - bos_idx) - src_sen_max_len_cur_batch
                    eos_idx = eos_idx - margin
                src_sen_context_list[num_sen] = decoder_hidden_states[num_instance, bos_idx: eos_idx]

                num_sen += 1

        return src_sen_context_list, src_sen_max_len_cur_batch

    def forward_reordering_task(self, source_idxys, target_idxys, target_index, pseudo_target_idxys, num_effective_source_tokens, num_effective_target_tokens):
        """
            source_idxys: shape = [batch_size, max_src_len, (4)].
            target_idxys: shape = [batch_size, max_tgt_len, (4)].
            target_index: shape = [batch_size, max_target_len].
            pseudo_target_idxys: shape = [batch_size, max_tgt_len, (4)].
            num_effective_source_tokens: shape = [batch_size].
            num_effective_target_tokens: shape = [batch_size].
        """

        max_src_len = source_idxys.size(1)
        max_tgt_len = target_idxys.size(1)
        max_pseudo_tgt_len = pseudo_target_idxys.size(1)
        # assert max_tgt_len == max_pseudo_tgt_len
        # assert max_src_len > 0 and max_tgt_len > 0

        if self.config.base_model_type == 'layoutlm':
            source_xys = source_idxys[:, :, 1:] # shape = [batch_size, max_src_len, 4].
            target_xys = target_idxys[:, :, 1:]
            pseudo_target_xys = pseudo_target_idxys[:, :, 1:]

            source_ids = source_idxys[:, :, 0] # shape = [batch_size, max_src_len].
            target_ids = target_idxys[:, :, 0]
            pseudo_target_ids = pseudo_target_idxys[:, :, 0]
        else:
            source_ids = source_idxys # shape = [batch_size, max_src_len].
            target_ids = target_idxys
            pseudo_target_ids = pseudo_target_idxys


        source_mask = self.create_basic_mask(num_effective_source_tokens, max_src_len) # shape = [batch_size, max_src_len], each element the value of 0/1, where 1 stands for "valid token" and 0 stands for "special token".
        target_mask = self.create_basic_mask(num_effective_target_tokens, max_tgt_len) # shape = [batch_size, max_tgt_len], each element the value of 0/1, where 1 stands for "valid token" and 0 stands for "special token".

        source_self_attention_mask, target_self_attention_mask, cross_attention_mask = \
            self.create_attention_mask(source_mask, target_mask)

        if self.config.base_model_type == 'layoutlm':
            # encoder
            encoder_outputs = self.encoder(input_ids=source_ids, bbox=source_xys, attention_mask=source_self_attention_mask, position_ids=None, inputs_embeds=None, return_emb=True)
            encoder_context, encoder_embedding_output = encoder_outputs[0], encoder_outputs[-1]
            # decoder
            decoder_outputs = self.decoder(encoder_context=encoder_context, decoder_input_ids=pseudo_target_ids, bbox=pseudo_target_xys, attention_mask=target_self_attention_mask, cross_attention_mask=cross_attention_mask, position_ids=None, inputs_embeds=None)
            decoder_hidden_states = decoder_outputs[0] # shape = [batch_size, max_tgt_len, hidden_dim].
        else:
            # encoder
            encoder_outputs = self.encoder(input_ids=source_ids,  attention_mask=source_self_attention_mask, position_ids=None, inputs_embeds=None, return_emb=True)
            encoder_context, encoder_embedding_output = encoder_outputs[0], encoder_outputs[-1]
            # decoder
            decoder_outputs = self.decoder(encoder_context=encoder_context, decoder_input_ids=pseudo_target_ids, attention_mask=target_self_attention_mask, cross_attention_mask=cross_attention_mask, position_ids=None, inputs_embeds=None)
            decoder_hidden_states = decoder_outputs[0] # shape = [batch_size, max_tgt_len, hidden_dim].
            
        # do the prediction and mask the prediction scores of the special tokens.
        prediction_scores = self.prediction_head(decoder_hidden_states, encoder_embedding_output) # shape: [batch_size, max_tgt_len, max_src_len].

        prediction_scores_masked = prediction_scores # Not leveraging the prior knowledge of num_effective_source_tokens.
        

        # loss calculation.
        loss_mask = target_mask[...]
        masked_loss = self.crit_mask_lm_smoothed(
                F.log_softmax(prediction_scores_masked.float(), dim=-1), target_index, num_effective_src_tokens=num_effective_source_tokens,
                loss_mask=loss_mask) # shape = [batch_size, max_tgt_len].

        return {
            "reordering_task_loss": masked_loss,
            "decoder_hidden_states": decoder_hidden_states,
            "max_tgt_len": max_tgt_len
        }

    def forward_senseg_task(self, senseg_target_labels, num_effective_target_tokens, decoder_hidden_states, max_tgt_len):
        """
            num_effective_target_tokens: shape = [batch_size].
            senseg_target_labels: shape = [batch_size, max_tgt_len].
            decoder_hidden_states: shape = [batch_size, max_tgt_len, hidden_dim].
        """

        # get the senseg task self_attn_mask.
        senseg_target_mask = self.create_senseg_basic_mask(num_effective_target_tokens, max_tgt_len)
        senseg_attention_mask = self.create_senseg_attention_mask(senseg_target_mask)
        # print(senseg_attention_mask.shape)
        # print(decoder_hidden_states.shape)

        # forward.
        senseg_encoder_outputs = self.senseg_encoder(decoder_hidden_states, senseg_attention_mask)
        senseg_encoder_hidden_states = senseg_encoder_outputs[0] # shape = [batch_size, max_tgt_len, hidden_dim].

        senseg_prediction_scores = self.senseg_prediction_head(senseg_encoder_hidden_states) # shape = [batch_size, max_tgt_len, senseg_num_label].

        # loss calculation.
        senseg_loss_mask = senseg_target_mask.clone()
        senseg_masked_loss = self.crit_mask_senseg_smoothed(F.softmax(senseg_prediction_scores, dim=-1), senseg_target_labels, senseg_loss_mask)

        return {
            "senseg_task_loss": senseg_masked_loss,
        }

    def prepare_chunks_for_trans_task(self, senseg_target_labels, num_effective_src_sen_tokens, num_effective_tgt_sen_tokens, num_effective_target_tokens, tokenizer, decoder_hidden_states, trans_decoder_input_ids, trans_decoder_labels):
        senseg_target_labels[:, 0] = 0
        src_sen_context_list, src_sen_max_len_cur_batch = self.create_src_sen_context_from_decoder_hidden_states(
            senseg_target_labels=senseg_target_labels,
            num_effective_target_tokens=num_effective_target_tokens,
            decoder_hidden_states=decoder_hidden_states,
            num_effective_src_sen_tokens=num_effective_src_sen_tokens,
        ) 

        src_pad_token_id=tokenizer.pad_token_id
        src_embedding_layer=self.encoder.embeddings.word_embeddings


        # sort the variables according to the effective seq len.
        _, ascending_idx = torch.sort(num_effective_src_sen_tokens)
        chunked_ascending_idx = []

        ascending_idx_cur_chunk = []
        accu_tokens_cur_chunk = 0
        i = 0
        while i <= ascending_idx.shape[0] - 1:
            idx = ascending_idx[i].item()
            num_src_tokens = num_effective_src_sen_tokens[idx]
            num_decoder_inputs_tokens = num_effective_tgt_sen_tokens[idx] + 1 # + 1 for [SEP] token.
            num_decoder_labels = num_effective_tgt_sen_tokens[idx] + 1 # + 1 for [EOS] token.

            tokens_cur_ins = num_src_tokens + num_decoder_inputs_tokens + num_decoder_labels
            accu_tokens_cur_chunk += tokens_cur_ins

            if accu_tokens_cur_chunk < self.trans_decoder_max_fwd_tokens:
                ascending_idx_cur_chunk.append(idx)
                i += 1
            else:
                chunked_ascending_idx.append(copy.deepcopy(ascending_idx_cur_chunk))
                ascending_idx_cur_chunk = []
                accu_tokens_cur_chunk = 0

        # do not forget the last chunk.
        chunked_ascending_idx.append(copy.deepcopy(ascending_idx_cur_chunk))

        assert sum([len(item) for item in chunked_ascending_idx]) == ascending_idx.shape[0]
        batch_size_sen_level = num_effective_src_sen_tokens.shape[0]
        assert batch_size_sen_level == ascending_idx.shape[0]


        # get the variable chunks.
        src_sen_context_chunks = []
        trans_decoder_input_ids_chunks = []
        trans_decoder_labels_chunks = []
        trans_basic_tgt_mask_chunks = []
        trans_target_self_attn_mask_chunks = []
        trans_cross_attn_mask_chunks = []
        for ascending_idx in chunked_ascending_idx:
            chunk_size = len(ascending_idx)

            # get src_context_cur_chunk.
            num_effective_src_sen_tokens_cur_chunk = num_effective_src_sen_tokens[ascending_idx].tolist()
            # num_effective_src_sen_tokens_cur_chunk = [num_effective_src_sen_tokens[i] for i in ascending_idx]
            max_src_tokens_num = max(num_effective_src_sen_tokens_cur_chunk)
            padding_ids_tsr = torch.ones((chunk_size, max_src_tokens_num), dtype=senseg_target_labels.dtype, device=senseg_target_labels.device) * src_pad_token_id
            src_context_cur_chunk = src_embedding_layer(padding_ids_tsr) # [chunk_size/num_sen_cur_chunk, max_src_tokens_num_cur_chunk, hidden_dim].

            for i in range(len(ascending_idx)):
                idx = ascending_idx[i]
                src_context_cur_chunk[i, 0:num_effective_src_sen_tokens[idx]] = src_sen_context_list[idx][:]
            src_sen_context_chunks.append(src_context_cur_chunk)

            # for idx, src_context_tsr in zip(ascending_idx, src_context_cur_chunk):
            #     src_context_tsr[0:num_effective_src_sen_tokens[idx]] = src_sen_context_list[idx]
            # src_sen_context_chunks.append(src_context_cur_chunk)

            # get trans_decoder_input_ids_chunk.
            num_effective_tgt_sen_tokens_cur_chunk = num_effective_tgt_sen_tokens[ascending_idx].tolist()
            # num_effective_tgt_sen_tokens_cur_chunk = [num_effective_tgt_sen_tokens[i] for i in ascending_idx]
            max_trans_decoder_input_ids_num = max(num_effective_tgt_sen_tokens_cur_chunk) + 1 # + 1 for [SEP] token.
            trans_decoder_input_ids_cur_chunk = trans_decoder_input_ids[torch.tensor(ascending_idx, dtype=torch.long, device=trans_decoder_input_ids.device), :]
            trans_decoder_input_ids_cur_chunk = trans_decoder_input_ids_cur_chunk[:, :max_trans_decoder_input_ids_num] # [chunk_size/num_sen_cur_chunk, max_trans_decoder_input_ids_num_cur_chunk].
            trans_decoder_input_ids_chunks.append(trans_decoder_input_ids_cur_chunk)

            # get trans_decoder_labels_chunk.
            max_trans_decoder_labels_num = max_trans_decoder_input_ids_num
            trans_decoder_labels_cur_chunk = trans_decoder_labels[torch.tensor(ascending_idx, dtype=torch.long, device=trans_decoder_input_ids.device), :]
            trans_decoder_labels_cur_chunk = trans_decoder_labels_cur_chunk[:, :max_trans_decoder_labels_num] # [chunk_size/num_sen_cur_chunk, max_trans_decoder_labels_num_cur_chunk].
            trans_decoder_labels_chunks.append(trans_decoder_labels_cur_chunk)

            # get masks.
            trans_basic_src_mask_cur_chunk = self.create_trans_basic_src_mask(torch.tensor(num_effective_src_sen_tokens_cur_chunk, dtype=num_effective_src_sen_tokens.dtype, device=num_effective_src_sen_tokens.device), max_src_tokens_num)
            trans_basic_tgt_mask_cur_chunk = self.create_trans_basic_tgt_mask(torch.tensor(num_effective_tgt_sen_tokens_cur_chunk, dtype=num_effective_src_sen_tokens.dtype, device=num_effective_src_sen_tokens.device), max_trans_decoder_input_ids_num)
            trans_target_self_attn_mask_cur_chunk, trans_cross_attn_mask_cur_chunk = self.create_trans_attention_mask(trans_basic_src_mask_cur_chunk, trans_basic_tgt_mask_cur_chunk)

            trans_basic_tgt_mask_chunks.append(trans_basic_tgt_mask_cur_chunk)
            trans_target_self_attn_mask_chunks.append(trans_target_self_attn_mask_cur_chunk)
            trans_cross_attn_mask_chunks.append(trans_cross_attn_mask_cur_chunk)

        return {
            "batch_size_sen_level": batch_size_sen_level,
            "num_chunks": len(chunked_ascending_idx),
            "src_sen_context_chunks": src_sen_context_chunks,
            "trans_decoder_input_ids_chunks": trans_decoder_input_ids_chunks,
            "trans_target_self_attn_mask_chunks": trans_target_self_attn_mask_chunks,
            "trans_cross_attn_mask_chunks": trans_cross_attn_mask_chunks,
            "trans_basic_tgt_mask_chunks": trans_basic_tgt_mask_chunks,
            "trans_decoder_labels_chunks": trans_decoder_labels_chunks,
        }


    def forward_trans_task_one_chunk(self, batch_size_sen_level, src_sen_context_chunk, trans_decoder_input_ids_chunk, trans_target_self_attn_mask_chunk, trans_cross_attn_mask_chunk, trans_basic_tgt_mask_chunk, trans_decoder_labels_chunk):
        trans_decoder_outputs = self.trans_decoder(
                encoder_context=src_sen_context_chunk,
                decoder_input_ids=trans_decoder_input_ids_chunk,
                attention_mask=trans_target_self_attn_mask_chunk,
                cross_attention_mask=trans_cross_attn_mask_chunk,
                position_ids=None,
            )
        trans_decoder_hidden_states_chunk = trans_decoder_outputs[0]
        trans_prediction_scores_chunk = self.trans_prediction_head(trans_decoder_hidden_states_chunk) # shape = [chunk_size, tgt_sen_max_len_cur_batch, vocab_size].

        # loss calculation.
        trans_loss_mask_chunk = trans_basic_tgt_mask_chunk.clone()
        trans_masked_loss_chunk = self.crit_mask_trans_smoothed(F.log_softmax(trans_prediction_scores_chunk.float(), dim=-1), trans_decoder_labels_chunk, loss_mask=trans_loss_mask_chunk)

        norm_ratio = (src_sen_context_chunk.shape[0]) / batch_size_sen_level
        normed_trans_masked_loss_chunk = trans_masked_loss_chunk * norm_ratio

        return {
            "trans_task_loss_one_chunk": normed_trans_masked_loss_chunk,
        }



class LayoutlmReorderingSensegTransIncr(BertPreTrainedForSeq2SeqModel):

    def __init__(self, config):
        super(LayoutlmReorderingSensegTransIncr, self).__init__(config)

        self.config = config

        # For word reordering task.
        if config.base_model_type == 'layoutlm':
            self.encoder = LayoutlmModel(config)
            self.decoder = LayoutlmModelForDecoderIncr(config, self.encoder)
        else:
            self.encoder = BertModel(config)
            self.decoder = BertModelForDecoderIncr(config, self.encoder)

        self.prediction_head = LayoutlmSPOnlyMLMHead(config, src_len=config.max_source_length)

        self.crit_mask_lm_smoothed = CrossEntropyLoss_(config.label_smoothing, config.max_source_length, ignore_index=None, reduction='none')
        
        # For sentence segmentation task.
        self.senseg_encoder = BertEncoderForSenseg(config)
        self.senseg_prediction_head = LayoutlmSensegPredictionHead(config)

        self.crit_mask_senseg_smoothed = FocalLoss_(len(config.senseg_task_ctg_to_id_map), ignore_index=None, reduction='none')


        # For translation task.
        self.trans_decoder = BertModelForDecoderForTrans(config)
        self.trans_prediction_head = TransPredictionHead(config, self.trans_decoder.embeddings.word_embeddings)

        self.crit_mask_trans_smoothed = CrossEntropyLossForTrans_(config.label_smoothing, config.tgt_vocab_size, ignore_index=None, reduction='none')

        self.trans_decoder_max_fwd_tokens = self.config.trans_decoder_max_fwd_tokens


        # Random initializing the weights, which may be overwritten by the pretrained model weights depend on the code logic.
        self.init_weights()


    @staticmethod
    def create_basic_mask(num_effective_tokens, max_seq_len):
        """
            num_effective_tokens: shape = [batch_size].
        """

        base_position_matrix = torch.arange(
            0, max_seq_len, dtype=num_effective_tokens.dtype, device=num_effective_tokens.device).view(1, -1) # [1, max_seq_len].
        mask = (base_position_matrix < (num_effective_tokens + 1).view(-1, 1)).type_as(num_effective_tokens) #  [batch_size, max_seq_len].
 
        return mask # shape = [batch_size, max_seq_len]

    @staticmethod
    def create_attention_mask(source_mask, target_mask, ):
        """
        inputs:
            source_mask: shape = [batch_size, max_src_len].
            target_mask: shape = [batch_size, max_tgt_len].

        returns:
            source_self_attn_mask: shape = [batch_size, max_src_len, max_src_len].
            target_self_attn_mask: shape = [batch_size, max_tgt_len, max_tgt_len].
            cross_attn_mask: shape = [batch_size, max_tgt_len, max_src_len].

        """
        
        # produce source_self_attn_mask.
        batch_size = source_mask.size(0)
        from_seq_len, to_seq_len = source_mask.size(1), source_mask.size(1)
        cls_token_attn_mask = torch.zeros((batch_size, from_seq_len, to_seq_len), dtype=source_mask.dtype, device=source_mask.device)
        cls_token_attn_mask[:, 0, :] = source_mask[:, :]
        effective_token_attn_mask = source_mask.unsqueeze(1).expand((batch_size, from_seq_len, to_seq_len))
        non_cls_pad_mask = source_mask.unsqueeze(2).expand((batch_size, from_seq_len, to_seq_len))
        effective_token_attn_mask = (effective_token_attn_mask & non_cls_pad_mask).type_as(source_mask)
        # pad_token_attn_mask = torch.zeros((batch_size, from_seq_len, to_seq_len), dtype=source_mask.dtype, device=source_mask.device)
        ## produce the final source_self_attn_mask.
        source_self_attn_mask = (cls_token_attn_mask | effective_token_attn_mask).type_as(source_mask)


        # produce target_self_attn_mask.
        batch_size = target_mask.size(0)
        from_seq_len, to_seq_len = target_mask.size(1), target_mask.size(1)
        ## produce sep_token_attn_mask. sep_token能attend到sep_token.
        sep_token_attn_mask = torch.zeros((batch_size, from_seq_len, to_seq_len), dtype=target_mask.dtype, device=target_mask.device)
        sep_token_attn_mask[:, 0, 0] = 1
        effective_token_attn_mask = torch.tril(torch.ones(from_seq_len, to_seq_len), diagonal=0).to(target_mask.dtype).to(target_mask.device).expand((batch_size, from_seq_len, to_seq_len))
        non_sep_pad_mask = target_mask.unsqueeze(2).expand((batch_size, from_seq_len, to_seq_len))
        effective_token_attn_mask = (effective_token_attn_mask & non_sep_pad_mask).type_as(target_mask)
        # pad_token_attn_mask = torch.zeros((batch_size, from_seq_len, to_seq_len), dtype=target_mask.dtype, device=target_mask.device)
        ## produce the final target_self_attn_mask.
        target_self_attn_mask = (sep_token_attn_mask | effective_token_attn_mask).type_as(target_mask)


        # produce cross_attn_mask.
        batch_size = target_mask.size(0)
        from_seq_len, to_seq_len = target_mask.size(1), source_mask.size(1)
        sep_token_attn_mask = torch.zeros((batch_size, from_seq_len, to_seq_len), dtype=source_mask.dtype, device=target_mask.device)
        sep_token_attn_mask[:, 0, :] = source_mask[:, :]
        effective_token_attn_mask = source_mask.unsqueeze(1).expand((batch_size, from_seq_len, to_seq_len))
        non_cls_pad_mask = target_mask.unsqueeze(2).expand((batch_size, from_seq_len, to_seq_len))
        effective_token_attn_mask = (effective_token_attn_mask & non_cls_pad_mask).type_as(source_mask)
        # pad_token_attn_mask = torch.zeros((batch_size, from_seq_len, to_seq_len), dtype=source_mask.dtype, device=source_mask.device)
        ## produce the final cross_attn_mask.
        cross_attn_mask = (sep_token_attn_mask | effective_token_attn_mask).type_as(source_mask)

        return source_self_attn_mask, target_self_attn_mask, cross_attn_mask

    @staticmethod
    def create_senseg_basic_mask(num_effective_tokens, max_seq_len):
        """
            num_effective_tokens: shape = [batch_size].
        """

        base_position_matrix = torch.arange(
            0, max_seq_len, dtype=num_effective_tokens.dtype, device=num_effective_tokens.device).view(1, -1) # [1, max_seq_len].
        mask = (base_position_matrix < num_effective_tokens.view(-1, 1)).type_as(num_effective_tokens) #  [batch_size, max_seq_len].
 
        return mask # shape = [batch_size, max_seq_len].

    @staticmethod
    def create_senseg_attention_mask(target_mask):
        """
        inputs:
            target_mask: shape = [batch_size, max_tgt_len].

        returns:
            target_self_attn_mask: shape = [batch_size, max_tgt_len, max_tgt_len].
        """

        # produce target_self_attn_mask.
        batch_size = target_mask.size(0)
        from_seq_len, to_seq_len = target_mask.size(1), target_mask.size(1)
        sep_token_attn_mask = torch.zeros((batch_size, from_seq_len, to_seq_len), dtype=target_mask.dtype, device=target_mask.device)
        sep_token_attn_mask[:, 0, :] = target_mask[:, :]
        effective_token_attn_mask = target_mask.unsqueeze(1).expand((batch_size, from_seq_len, to_seq_len))
        non_sep_pad_mask = target_mask.unsqueeze(2).expand((batch_size, from_seq_len, to_seq_len))
        effective_token_attn_mask = (effective_token_attn_mask & non_sep_pad_mask).type_as(target_mask)
        # pad_token_attn_mask = torch.zeros((batch_size, from_seq_len, to_seq_len), dtype=target_mask.dtype, device=target_mask.device)
        ## produce the final source_self_attn_mask.
        target_self_attn_mask = (sep_token_attn_mask | effective_token_attn_mask).type_as(target_mask)

        return target_self_attn_mask


    @staticmethod
    def create_trans_basic_src_mask(num_effective_tokens, src_sen_max_len):
        """
            num_effective_tokens: shape = [batch_size_sen_level].
        """

        base_position_matrix = torch.arange(
            0, src_sen_max_len, dtype=num_effective_tokens.dtype, device=num_effective_tokens.device).view(1, -1) # [1, src_sen_max_len].
        mask = (base_position_matrix < num_effective_tokens.view(-1, 1)).type_as(num_effective_tokens) #  [batch_size, src_sen_max_len].
 
        return mask # shape = [batch_size_sen_level, src_sen_max_len].

    @staticmethod
    def create_trans_basic_tgt_mask(num_effective_tokens, tgt_sen_max_len):
        """
            num_effective_tokens: shape = [batch_size_sen_level].
        """

        base_position_matrix = torch.arange(
            0, tgt_sen_max_len, dtype=num_effective_tokens.dtype, device=num_effective_tokens.device).view(1, -1) # [1, tgt_sen_max_len].
        mask = (base_position_matrix < (num_effective_tokens + 1).view(-1, 1)).type_as(num_effective_tokens) #  [batch_size_sen_level, tgt_sen_max_len].
 
        return mask # shape = [batch_size_sen_level, tgt_sen_max_len]

    @staticmethod
    def create_trans_attention_mask(source_mask, target_mask, ):
        """
        inputs:
            source_mask: shape = [batch_size_sen_level, src_sen_max_len].
            target_mask: shape = [batch_size_sen_level, tgt_sen_max_len].

        returns:
            target_self_attn_mask: shape = [batch_size_sen_level, tgt_sen_max_len, tgt_sen_max_len].
            cross_attn_mask: shape = [batch_size_sen_level, tgt_sen_max_len, src_sen_max_len].

        """

        # produce target_self_attn_mask.
        batch_size = target_mask.size(0)
        from_seq_len, to_seq_len = target_mask.size(1), target_mask.size(1)
        ## produce sep_token_attn_mask. sep_token能attend到sep_token.
        sep_token_attn_mask = torch.zeros((batch_size, from_seq_len, to_seq_len), dtype=target_mask.dtype, device=target_mask.device)
        sep_token_attn_mask[:, 0, 0] = 1
        effective_token_attn_mask = torch.tril(torch.ones(from_seq_len, to_seq_len), diagonal=0).to(target_mask.dtype).to(target_mask.device).expand((batch_size, from_seq_len, to_seq_len))
        non_sep_pad_mask = target_mask.unsqueeze(2).expand((batch_size, from_seq_len, to_seq_len))
        effective_token_attn_mask = (effective_token_attn_mask & non_sep_pad_mask).type_as(target_mask)
        # pad_token_attn_mask = torch.zeros((batch_size, from_seq_len, to_seq_len), dtype=target_mask.dtype, device=target_mask.device)
        ## produce the final target_self_attn_mask.
        target_self_attn_mask = (sep_token_attn_mask | effective_token_attn_mask).type_as(target_mask)


        # produce cross_attn_mask.
        batch_size = target_mask.size(0)
        from_seq_len, to_seq_len = target_mask.size(1), source_mask.size(1)
        sep_token_attn_mask = torch.zeros((batch_size, from_seq_len, to_seq_len), dtype=source_mask.dtype, device=target_mask.device)
        sep_token_attn_mask[:, 0, :] = source_mask[:, :]
        effective_token_attn_mask = source_mask.unsqueeze(1).expand((batch_size, from_seq_len, to_seq_len))
        non_cls_pad_mask = target_mask.unsqueeze(2).expand((batch_size, from_seq_len, to_seq_len))
        effective_token_attn_mask = (effective_token_attn_mask & non_cls_pad_mask).type_as(source_mask)
        # pad_token_attn_mask = torch.zeros((batch_size, from_seq_len, to_seq_len), dtype=source_mask.dtype, device=source_mask.device)
        ## produce the final cross_attn_mask.
        cross_attn_mask = (sep_token_attn_mask | effective_token_attn_mask).type_as(source_mask)

        return target_self_attn_mask, cross_attn_mask  


    def create_src_sen_context_from_decoder_hidden_states(self, senseg_predicted_res, num_effective_target_tokens, decoder_hidden_states, src_embedding_layer, src_pad_token_id):
        """
            senseg_predicted_res: shape = [batch_size, max_tgt_len].
            num_effective_target_tokens: shape = [batch_size].
            decoder_hidden_states: shape = [batch_size, max_tgt_len, hidden_dim].
        """

        batch_size_sen_level = 0
        num_src_sen_tokens = []
        num_sens_each_ins = []
        for i in range(senseg_predicted_res.shape[0]):
            num_effective_target_token = num_effective_target_tokens[i]
            senseg_res = senseg_predicted_res[i, :num_effective_target_token]

            bos_token_index = torch.nonzero(senseg_res == 0, as_tuple=True)[0]
            shifted_bos_token_index = torch.cat((bos_token_index[1:], torch.tensor([num_effective_target_token], dtype=bos_token_index.dtype, device=bos_token_index.device)), dim=-1)

            num_src_sen_tokens_cur_instance = (shifted_bos_token_index - bos_token_index).tolist()
            num_src_sen_tokens += num_src_sen_tokens_cur_instance

            num_sens_each_ins.append(bos_token_index.shape[0])
        
        batch_size_sen_level = sum(num_sens_each_ins)
        
        max_num_src_sen_tokens = max(num_src_sen_tokens)

        pad_embedding = src_embedding_layer(torch.tensor(src_pad_token_id, dtype=senseg_predicted_res.dtype, device=senseg_predicted_res.device))
        src_sen_context = pad_embedding.repeat((batch_size_sen_level, max_num_src_sen_tokens, 1)) # shape = [batch_size_sen_level, max_num_src_sen_tokens, hidden_dim].

    
        # src_sen_context_idx = torch.ones((batch_size_sen_level, src_sen_max_len_cur_batch), dtype=senseg_target_labels.dtype, device=senseg_target_labels.device)
        src_sen_context_list = [None] * batch_size_sen_level   # shape = [batch_size_sen_level, src_sen_max_len_cur_batch, hidden_dim].

        num_sen = 0
        for num_instance, instance_labels in enumerate(senseg_predicted_res):
            num_effective_target_token = num_effective_target_tokens[num_instance].item()
            bos_token_index = torch.nonzero(instance_labels[:num_effective_target_token] == 0, as_tuple=True)[0]
            shifted_bos_token_index = torch.cat((bos_token_index[1:], torch.tensor([num_effective_target_token], dtype=bos_token_index.dtype, device=bos_token_index.device)), dim=-1)
            # print(bos_token_index)
            # print(shifted_bos_token_index)

            for bos_idx, eos_idx in zip(bos_token_index, shifted_bos_token_index):
                # print(bos_idx, eos_idx)
                if eos_idx - bos_idx > max_num_src_sen_tokens:
                    margin = (eos_idx - bos_idx) - max_num_src_sen_tokens
                    eos_idx = eos_idx - margin
                src_sen_context[num_sen, 0:(eos_idx-bos_idx)] = decoder_hidden_states[num_instance, bos_idx: eos_idx]

                num_sen += 1

        num_effective_src_sen_tokens = torch.tensor(num_src_sen_tokens, dtype=num_effective_target_tokens.dtype, device=num_effective_target_tokens.device)

        return src_sen_context, num_effective_src_sen_tokens, num_sens_each_ins


    @staticmethod
    def _expand_inputs_for_generation(
        expand_size: int = 1,
        input_ids: torch.LongTensor = None,
        **model_kwargs,
    ) -> torch.LongTensor:
        """Expands tensors from [batch_size, ...] to [batch_size * expand_size, ...]"""

        def _expand_dict_for_generation(dict_to_expand):
            for key in dict_to_expand:
                if dict_to_expand[key] is not None and isinstance(dict_to_expand[key], torch.Tensor):
                    dict_to_expand[key] = dict_to_expand[key].repeat_interleave(expand_size, dim=0)
            return dict_to_expand

        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)

        model_kwargs = _expand_dict_for_generation(model_kwargs)

        # if is_encoder_decoder:
        #     if model_kwargs.get("encoder_outputs") is None:
        #         raise ValueError("If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined.")
        #     model_kwargs["encoder_outputs"] = _expand_dict_for_generation(model_kwargs["encoder_outputs"])

        return input_ids, model_kwargs

    def beam_search_for_trans_task(self, trans_decoder, trans_decoder_prediction_head, encoder_context, max_src_sen_len_cur_batch, num_effective_src_sen_tokens, tgt_tokenizer, num_beams, length_penalty, early_stopping, max_generation_len_ratio, max_generation_abs_len, num_return_sequences):
        """
            encoder_context: shape = [batch_size_sen_level, max_src_len_cur_batch].
            num_effective_src_sen_tokens: shape = [batch_size_sen_level].
        """

        batch_size_sen_level = encoder_context.shape[0]

        # prepare max generation_length.
        max_generation_length = min(int(max_generation_len_ratio * max_src_sen_len_cur_batch), max_generation_abs_len)

        # prepare decoder input ids, attention masks.
        decoder_input_ids = torch.ones((batch_size_sen_level, 1), dtype=num_effective_src_sen_tokens.dtype, device=num_effective_src_sen_tokens.device) * tgt_tokenizer.pad_token_id
        trans_basic_src_mask = self.create_trans_basic_src_mask(num_effective_src_sen_tokens, max_src_sen_len_cur_batch)
        trans_basic_tgt_mask = torch.ones((batch_size_sen_level, max_generation_length), dtype=trans_basic_src_mask.dtype, device=trans_basic_src_mask.device)
        trans_target_self_attn_mask, trans_cross_attn_mask = self.create_trans_attention_mask(trans_basic_src_mask, trans_basic_tgt_mask)

        model_kwargs = {
            "encoder_context": encoder_context,
            "attention_mask": trans_target_self_attn_mask, 
            "cross_attention_mask": trans_cross_attn_mask, 
        }
        
        # prepare logits precessors.
        logits_processor = LogitsProcessorList(
            [
                MinLengthLogitsProcessor(1, eos_token_id=tgt_tokenizer.cls_token_id),
            ]
        )

        # prepare stopping criteria.
        criteria = StoppingCriteriaList()
        criteria.append(MaxLengthCriteria(max_length=max_generation_length))

        # prepare beam search scorer.
        beam_scorer = BeamSearchScorer(
                        batch_size=batch_size_sen_level,
                        num_beams=num_beams,
                        device=decoder_input_ids.device,
                        length_penalty=length_penalty,
                        do_early_stopping=early_stopping,
                        num_beam_hyps_to_keep=num_return_sequences,
                        max_length=max_generation_length,
                    )

        # interleave input_ids with `num_beams` additional sequences per batch.
        decoder_input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids=decoder_input_ids,
                expand_size=num_beams,
                **model_kwargs,
            )

        # run beam search.
        return beam_search(
            model=trans_decoder,
            model_prediction_head=trans_decoder_prediction_head,
            input_ids=decoder_input_ids,
            beam_scorer=beam_scorer,
            logits_processor=logits_processor,
            stopping_criteria=criteria,
            pad_token_id=tgt_tokenizer.pad_token_id,
            eos_token_id=tgt_tokenizer.cls_token_id,
            **model_kwargs,
        ) # torch.Longtensor


    def greedy_decode_beam_search(self, source_idxys, num_effective_source_tokens, tokenizer, tgt_tokenizer, num_beams, length_penalty, early_stopping, max_generation_len_ratio, max_generation_abs_len, num_return_sequences=1):
        """
            source_idxys: shape = [batch_size, max_src_len, (5)].
            num_effective_source_tokens: shape = [batch_size].
        """

        # ************************* Word reordering task. *****************************

        batch_size = source_idxys.size(0)
        max_src_len = source_idxys.size(1)
        # assert max_src_len > 0

        max_source_tokens_cur_batch = int(torch.max(num_effective_source_tokens))

        if self.config.base_model_type == 'layoutlm':
            source_xys = source_idxys[:, :, 1:] # shape = [batch_size, max_src_len, 4].
            source_ids = source_idxys[:, :, 0] # shape = [batch_size, max_src_len].

            # construct the sep token_id and layout for the "start" of decoding.
            target_xys = torch.tensor(predefined_constant.SEP_TOKEN_LAYOUT, dtype=source_xys.dtype, device=source_xys.device).repeat((batch_size, 1, 1)) # shape = [batch_size, 1, 4].
            target_ids = torch.tensor(tokenizer.sep_token_id, dtype=source_ids.dtype, device=source_ids.device).repeat((batch_size, 1)) # shape = [batch_size, 1].
        else:
            source_ids = source_idxys # shape = [batch_size, max_src_len].

            target_ids = torch.tensor(tokenizer.sep_token_id, dtype=source_ids.dtype, device=source_ids.device).repeat((batch_size, 1)) # shape = [batch_size, 1].

        source_mask = self.create_basic_mask(num_effective_source_tokens, max_src_len) # shape = [batch_size, max_src_len].
        target_mask = torch.ones((batch_size, max_source_tokens_cur_batch+1), dtype=source_mask.dtype, device=source_mask.device) # shape = [batch_size, max_source_tokens_cur_batch].

        source_self_attention_mask, target_self_attention_mask, cross_attention_mask = \
            self.create_attention_mask(source_mask, target_mask)
        
        # get the encoder context and embedding output.
        encoder_inputs = {
            "input_ids": source_ids, 
            "attention_mask": source_self_attention_mask,
            "position_ids": None,
            "inputs_embeds": None,
            "return_emb": True,
        }
        if self.config.base_model_type == 'layoutlm':
            encoder_inputs["bbox"] = source_xys
        
        encoder_outputs = self.encoder(**encoder_inputs)
        encoder_context, encoder_embedding_output = encoder_outputs[0], encoder_outputs[-1]

        # prepare the source_mask_for_prection_scores.
        source_mask_for_prection_scores = source_mask.clone()
        # source_mask_for_prection_scores[:, 0] = 1
        
        
        prediction_scores_mask = ((1.0 - source_mask_for_prection_scores.unsqueeze(1)) * -1e5) # shape = [batch_size, 1, max_src_len].
        # autoregressively decoding the target seq.
        decoded_res = [] # store the decoded results. each item is a tensor of shape = [batch_size, 1], corresponding the the decoded token id at that time step.

        decoding_step = 0

        cur_input_ids = target_ids.clone()
        # print(f"cur_input_ids: {cur_input_ids}")
        if self.config.base_model_type == 'layoutlm':
            cur_input_xys = target_xys.clone()

        decoder_all_layers_historical_hidden_states = ()
        while decoding_step < max_source_tokens_cur_batch + 1:

            # prepare decoder inputs of current decoding step.
            cur_self_attention_mask = target_self_attention_mask[:, decoding_step:(decoding_step+1), :(decoding_step+1)] # shape = [batch_size, 1, decoding_step+1].
            cur_cross_attention_mask = cross_attention_mask[:, decoding_step:(decoding_step+1), :] # shape = [batch_size, 1, max_src_len].


            decoder_inputs = {
                "encoder_context": encoder_context,
                "decoder_all_layers_historical_hidden_states": decoder_all_layers_historical_hidden_states,
                "decoder_cur_time_input_ids": cur_input_ids,
                "attention_mask": cur_self_attention_mask,
                "cross_attention_mask": cur_cross_attention_mask,
                "position_ids": None,
                "inputs_embeds": None,
            }
            if self.config.base_model_type == 'layoutlm':
                decoder_inputs["bbox"] = cur_input_xys

            # forward.
            decoder_all_layers_accu_hidden_states = self.decoder(**decoder_inputs)[0]
            decoder_last_layer_hidden_states = decoder_all_layers_accu_hidden_states[-1] # shape = [batch_size, decoding_step+1, hidden_dim].
            # prediction.
            cur_decoder_hidden_states = decoder_last_layer_hidden_states[:, -1:, :] # shape = [batch_size, 1, hidden_dim].
            prediction_scores = self.prediction_head(cur_decoder_hidden_states, encoder_embedding_output) # shape = [batch_size, 1, max_src_len].
            # prediction_scores_masked = prediction_scores + prediction_scores_mask # shape = [batch_size, 1, max_src_len]. Leveraging the prior knowledge of num_effective_source_tokens.
            prediction_scores_masked = prediction_scores # Not leveraging the prior knowledge of num_effective_source_tokens.
            _, max_ids = torch.max(prediction_scores_masked, dim=-1) # shape = [batch_size, 1].
            # print(f"prediction_scores_masked: {prediction_scores_masked}")

            # get the decoded token id (and layout) from the source seq, store the decoded token id, accumulate the decoder input ids and input layouts, and update the cur_input_ids/cur_input_xys for next decoding step.
            cur_decoded_ids = torch.gather(source_ids, 1, max_ids) # shape = [batch_size, 1].
            decoded_res.append(cur_decoded_ids)
            # accu_decoder_input_ids = torch.cat((accu_decoder_input_ids, cur_decoded_ids), dim=1) 
            cur_input_ids = cur_decoded_ids
            decoder_all_layers_historical_hidden_states = decoder_all_layers_accu_hidden_states
            if  self.config.base_model_type == 'layoutlm':
                _, _, layout_dim = source_xys.shape
                max_ids_ = max_ids.unsqueeze(-1).expand(max_ids.size(0), max_ids.size(1), layout_dim) # shape = [batch_size, 1, 4].
                cur_decoded_xys = torch.gather(source_xys, 1, max_ids_) # shape = [batch_size, 1, 4].
                # accu_decoder_input_xys = torch.cat((accu_decoder_input_xys, cur_decoded_xys), dim=1)
                cur_input_xys = cur_decoded_xys

            decoding_step += 1

        reordering_task_decoding_res = torch.cat(decoded_res, dim=1) # shape = [batch_size, max_source_tokens_cur_batch + 1].


        # ************************* Sentence segmentation task. *****************************

        # get the word reordering task's decoder hidden_states.
        decoder_last_layer_accu_hidden_states = decoder_all_layers_accu_hidden_states[-1] # shape = [batch_size, max_source_tokens_cur_batch+1, hidden_dim].

        # get the senseg task self_attn_mask.
        senseg_target_mask = self.create_senseg_basic_mask(num_effective_source_tokens, max_source_tokens_cur_batch+1)
        senseg_attention_mask = self.create_senseg_attention_mask(senseg_target_mask)

        # forward.
        senseg_encoder_outputs = self.senseg_encoder(decoder_last_layer_accu_hidden_states, senseg_attention_mask)
        senseg_encoder_hidden_states = senseg_encoder_outputs[0] # shape = [batch_size, max_source_tokens_cur_batch+1, hidden_dim].

        senseg_prediction_scores = self.senseg_prediction_head(senseg_encoder_hidden_states) # shape = [batch_size, max_source_tokens_cur_batch+1, senseg_num_label].
        _, senseg_max_ids = torch.max(senseg_prediction_scores, dim=-1) # shape = [batch_size, max_source_tokens_cur_batch+1].

        senseg_task_prediction_res = senseg_max_ids

        # ************************* Translation task. *****************************
        src_sen_context, num_effective_src_sen_tokens, num_sens_each_instance = self.create_src_sen_context_from_decoder_hidden_states(
            senseg_predicted_res=senseg_task_prediction_res[:, :-1],
            num_effective_target_tokens=num_effective_source_tokens,
            decoder_hidden_states=decoder_last_layer_accu_hidden_states[:, :-1, :],
            src_embedding_layer=self.encoder.embeddings.word_embeddings,
            src_pad_token_id=tokenizer.pad_token_id,
        ) # src_sen_context, shape = [batch_size_sen_level, max_num_src_sen_tokens, hidden_dim].

        # chunk beam search to prevent OOM error.
        max_num_src_sen_tokens_cur_batch = src_sen_context.shape[1]
        batch_size_sen_level = src_sen_context.shape[0]
        num_chunks = math.ceil(batch_size_sen_level / (math.floor(self.trans_decoder_max_fwd_tokens / (max_num_src_sen_tokens_cur_batch * 3))))
        src_sen_context_chunk_list = torch.chunk(src_sen_context, num_chunks)
        num_effective_src_sen_tokens_chunk_list = torch.chunk(num_effective_src_sen_tokens, num_chunks)

        assert len(src_sen_context_chunk_list) == len(num_effective_src_sen_tokens_chunk_list)

        max_src_sen_len_cur_batch = max(num_effective_src_sen_tokens.tolist())
        beam_searched_tgt_ids_list = []
        for i in range(len(src_sen_context_chunk_list)):
            src_sen_context_chunk = src_sen_context_chunk_list[i]
            num_effective_src_sen_tokens_chunk = num_effective_src_sen_tokens_chunk_list[i]
            beam_searched_tgt_ids_chunk = self.beam_search_for_trans_task(
                trans_decoder=self.trans_decoder,
                trans_decoder_prediction_head=self.trans_prediction_head,
                encoder_context=src_sen_context_chunk,
                max_src_sen_len_cur_batch=max_src_sen_len_cur_batch,
                num_effective_src_sen_tokens=num_effective_src_sen_tokens_chunk,
                tgt_tokenizer=tgt_tokenizer,
                num_beams=num_beams,
                length_penalty=length_penalty,
                early_stopping=early_stopping,
                max_generation_len_ratio=max_generation_len_ratio,
                max_generation_abs_len=max_generation_abs_len,
                num_return_sequences=num_return_sequences,
            ) # shape = [chunk_size, max_generation_len].

            beam_searched_tgt_ids_list.append(beam_searched_tgt_ids_chunk)

        max_generated_len = max([tsr.shape[-1] for tsr in beam_searched_tgt_ids_list])
        beam_searched_tgt_ids_padded_list = []
        for tsr in beam_searched_tgt_ids_list:
            pad_tsr = torch.ones((tsr.shape[0], max_generated_len - tsr.shape[-1]), dtype=tsr.dtype, device=tsr.device) * tgt_tokenizer.pad_token_id
            beam_searched_tgt_ids_padded_list.append(torch.cat((tsr, pad_tsr), dim=-1))
        beam_searched_tgt_ids = torch.cat(beam_searched_tgt_ids_padded_list, dim=0) # shape = [batch_size_sen_level, max_generation_len].

        # group the trans res of sentences belonging to the same instance together.
        beam_searched_tgt_ids_tuple = torch.split(beam_searched_tgt_ids, num_sens_each_instance)

        return {
            "reordering_task_res": reordering_task_decoding_res, # shape = [batch_size, max_source_tokens_cur_batch + 1].
            "senseg_task_res": senseg_task_prediction_res, # shape = [batch_size, max_source_tokens_cur_batch+1].
            "trans_task_res": beam_searched_tgt_ids_tuple # a tuple of tensor of shape = [num_sen_this_instance, max_generation_len]. 
        }