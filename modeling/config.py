from .predefined_constant import LAYOUTLM_PRETRAINED_CONFIG_ARCHIVE_MAP

from transformers import BertConfig


import logging
logger = logging.getLogger(__name__)

class BertForSeq2SeqConfig(BertConfig):

    def __init__(self, **kwargs):
        super(BertForSeq2SeqConfig, self).__init__(**kwargs)

    @classmethod
    def from_exist_config(cls, config, max_source_length, max_target_length, label_smoothing, base_model_type, layoutlm_only_layout, senseg_encoder_num_hidden_layers, senseg_task_loss_relative_weight, senseg_task_ctg_to_id_map, senseg_task_id_to_ctg_map, src_sen_max_len, tgt_sen_max_len, tgt_vocab_size, tgt_max_position_embeddings, trans_decoder_num_hidden_layers, trans_task_relative_weight, trans_decoder_max_fwd_tokens, *args, **kwargs_):

        required_keys = [
            "vocab_size", "hidden_size", "num_hidden_layers", "num_attention_heads",
            "hidden_act", "intermediate_size", "hidden_dropout_prob", "attention_probs_dropout_prob",
            "max_position_embeddings",  "initializer_range", "layer_norm_eps"]
        
        kwargs = {}
        for key in required_keys:
            assert hasattr(config, key)
            kwargs[key] = getattr(config, key)

        # config is an instance of LayoutlmConfig and has the "max_2d_position_embeddings" super-parameter. In the meanwhile, passing the LayoutlmConfig instance means the generated config is for layout-aware model.
        if hasattr(config, 'max_2d_position_embeddings'):
            layoutlm_special_keys = ['max_2d_position_embeddings',]
            for key in layoutlm_special_keys:
                kwargs[key] = getattr(config, key)

        kwargs['max_source_length'] = max_source_length
        kwargs['max_target_length'] = max_target_length
        kwargs['label_smoothing'] = label_smoothing
        kwargs['base_model_type'] = base_model_type
        kwargs['layoutlm_only_layout'] = layoutlm_only_layout

        # for senseg task.
        kwargs['senseg_encoder_num_hidden_layers'] = senseg_encoder_num_hidden_layers
        kwargs['senseg_task_loss_relative_weight'] = senseg_task_loss_relative_weight
        kwargs['senseg_task_ctg_to_id_map'] = senseg_task_ctg_to_id_map
        kwargs['senseg_task_id_to_ctg_map'] = senseg_task_id_to_ctg_map

        # for trans task.
        kwargs['src_sen_max_len'] = src_sen_max_len
        kwargs['tgt_sen_max_len'] = tgt_sen_max_len
        kwargs['tgt_vocab_size'] = tgt_vocab_size
        kwargs['tgt_max_position_embeddings'] = tgt_max_position_embeddings
        kwargs['trans_decoder_num_hidden_layers'] = trans_decoder_num_hidden_layers
        kwargs['trans_task_relative_weight'] = trans_task_relative_weight
        kwargs['trans_decoder_max_fwd_tokens'] = trans_decoder_max_fwd_tokens

        if kwargs_:
            kwargs.update(kwargs_)

        return cls(**kwargs)
    

class LayoutlmConfig(BertConfig):
    pretrained_config_archive_map = LAYOUTLM_PRETRAINED_CONFIG_ARCHIVE_MAP

    def __init__(self, **kwargs):
        super().__init__(**kwargs)