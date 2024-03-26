import torch

from transformers.modeling_utils import cached_path, WEIGHTS_NAME, TF2_WEIGHTS_NAME, TF_WEIGHTS_NAME

import copy
import logging
logger = logging.getLogger(__name__)


def get_checkpoint_from_transformer_cache(
        archive_file, pretrained_model_name_or_path, pretrained_model_archive_map,
        cache_dir, force_download, proxies, resume_download,
):
    try:
        resolved_archive_file = cached_path(archive_file, cache_dir=cache_dir, force_download=force_download,
                                            proxies=proxies, resume_download=resume_download)
    except EnvironmentError:
        if pretrained_model_name_or_path in pretrained_model_archive_map:
            msg = "Couldn't reach server at '{}' to download pretrained weights.".format(
                archive_file)
        else:
            msg = "Model name '{}' was not found in model name list ({}). " \
                  "We assumed '{}' was a path or url to model weight files named one of {} but " \
                  "couldn't find any such file at this path or url.".format(
                pretrained_model_name_or_path,
                ', '.join(pretrained_model_archive_map.keys()),
                archive_file,
                [WEIGHTS_NAME, TF2_WEIGHTS_NAME, TF_WEIGHTS_NAME])
        raise EnvironmentError(msg)

    if resolved_archive_file == archive_file:
        logger.info("loading weights file {}".format(archive_file))
    else:
        logger.info("loading weights file {} from cache at {}".format(
            archive_file, resolved_archive_file))

    return torch.load(resolved_archive_file, map_location='cpu')

def bert_to_bertEncoderDecoder(state_dict):
    # for bertEncoderDecoder encoder state_dict.
    encoder_state_dict = {}
    for key in state_dict:
        value = state_dict[key]

        is_omit_key = False
        omit_keys = ["token_type_embeddings", "pooler", "predictions", "seq_relationship"]
        for omit_key in omit_keys:
            if omit_key in key:
                is_omit_key = True
        
        if is_omit_key:
            continue
        
        converted_key = key.replace("bert", "encoder").replace("gamma", "weight").replace("beta", "bias")

        encoder_state_dict[converted_key] = value
    
    # for bertEncoderDecoder decoder state_dict.
    decoder_state_dict = {}
    for key in state_dict:
        value = state_dict[key]

        is_omit_key = False
        omit_keys = ["token_type_embeddings", "pooler", "predictions", "seq_relationship"]
        for omit_key in omit_keys:
            if omit_key in key:
                is_omit_key = True
        
        if is_omit_key:
            continue

        converted_key = key.replace("bert", "decoder").replace("encoder", "decoder").replace("attention", "self_attention").replace("gamma", "weight").replace("beta", "bias")
        decoder_state_dict[converted_key] = value

    converted_encoder_decoder_state_dict = {}
    converted_encoder_decoder_state_dict.update(encoder_state_dict)
    converted_encoder_decoder_state_dict.update(decoder_state_dict)

    return converted_encoder_decoder_state_dict

def layoutlm_to_layoutlmEncoderDecoder(state_dict):
    # for bertEncoderDecoder encoder state_dict.
    encoder_state_dict = {}
    for key in state_dict:
        value = state_dict[key]

        is_omit_key = False
        omit_keys = ["token_type_embeddings", "pooler", "position_ids", "predictions"]
        for omit_key in omit_keys:
            if omit_key in key:
                is_omit_key = True
        
        if is_omit_key:
            continue

        converted_key = key.replace("layoutlm", "encoder")
        encoder_state_dict[converted_key] = value
    
    # for bertEncoderDecoder decoder state_dict.
    decoder_state_dict = {}
    for key in state_dict:
        value = state_dict[key]
        
        is_omit_key = False
        omit_keys = ["token_type_embeddings", "pooler", "position_ids", "predictions"]
        for omit_key in omit_keys:
            if omit_key in key:
                is_omit_key = True
        
        if is_omit_key:
            continue

        converted_key = key.replace("layoutlm", "decoder").replace("encoder", "decoder").replace("attention", "self_attention")
        decoder_state_dict[converted_key] = value

    converted_encoder_decoder_state_dict = {}
    converted_encoder_decoder_state_dict.update(encoder_state_dict)
    converted_encoder_decoder_state_dict.update(decoder_state_dict)

    return converted_encoder_decoder_state_dict


# ********************** For senseg task. ************************

def bert_to_bertReorderingSenseg(state_dict):
    # for bertReorderingSenseg encoder state_dict.
    encoder_state_dict = {}
    for key in state_dict:
        value = state_dict[key]

        is_omit_key = False
        omit_keys = ["token_type_embeddings", "pooler", "predictions", "seq_relationship"]
        for omit_key in omit_keys:
            if omit_key in key:
                is_omit_key = True
        
        if is_omit_key:
            continue
        
        converted_key = key.replace("bert", "encoder").replace("gamma", "weight").replace("beta", "bias")

        encoder_state_dict[converted_key] = value
    
    # for bertReorderingSenseg decoder state_dict.
    decoder_state_dict = {}
    for key in state_dict:
        value = state_dict[key]

        is_omit_key = False
        omit_keys = ["token_type_embeddings", "pooler", "predictions", "seq_relationship"]
        for omit_key in omit_keys:
            if omit_key in key:
                is_omit_key = True
        
        if is_omit_key:
            continue

        converted_key = key.replace("bert", "decoder").replace("encoder", "decoder").replace("attention", "self_attention").replace("gamma", "weight").replace("beta", "bias")
        decoder_state_dict[converted_key] = value

    # for bertReorderingSenseg senseg_encoder state_dict.
    state_dict_ = {
        k.replace("bert.encoder", "senseg_encoder"): v for (k, v) in state_dict.items()
    }
    state_dict_ = {
        k: v for (k, v) in state_dict_.items() \
            if k.startswith("senseg_encoder")
    }

    senseg_encoder_state_dict = {}
    for key in state_dict_:
        value = state_dict_[key]

        converted_key = key.replace("gamma", "weight").replace("beta", "bias")

        senseg_encoder_state_dict[converted_key] = value


    converted_encoder_decoder_state_dict = {}
    converted_encoder_decoder_state_dict.update(encoder_state_dict)
    converted_encoder_decoder_state_dict.update(decoder_state_dict)
    converted_encoder_decoder_state_dict.update(senseg_encoder_state_dict)

    return converted_encoder_decoder_state_dict

def layoutlm_to_layoutlmReorderingSenseg(state_dict):
    # for layoutlmReorderingSenseg encoder state_dict.
    encoder_state_dict = {}
    for key in state_dict:
        value = state_dict[key]

        is_omit_key = False
        omit_keys = ["token_type_embeddings", "pooler", "position_ids", "predictions"]
        for omit_key in omit_keys:
            if omit_key in key:
                is_omit_key = True
        
        if is_omit_key:
            continue

        converted_key = key.replace("layoutlm", "encoder")
        encoder_state_dict[converted_key] = value
    
    # for layoutlmReorderingSenseg decoder state_dict.
    decoder_state_dict = {}
    for key in state_dict:
        value = state_dict[key]
        
        is_omit_key = False
        omit_keys = ["token_type_embeddings", "pooler", "position_ids", "predictions"]
        for omit_key in omit_keys:
            if omit_key in key:
                is_omit_key = True
        
        if is_omit_key:
            continue

        converted_key = key.replace("layoutlm", "decoder").replace("encoder", "decoder").replace("attention", "self_attention")
        decoder_state_dict[converted_key] = value

    # for layoutlmReorderingSenseg senseg encoder state_dict.
    state_dict_ = {
        k.replace("layoutlm.encoder", "senseg_encoder"): v for (k, v) in state_dict.items()
    }
    state_dict_ = {
        k: v for (k, v) in state_dict_.items() \
            if k.startswith("senseg_encoder")
    }

    senseg_encoder_state_dict = {}
    for key in state_dict_:
        value = state_dict_[key]

        senseg_encoder_state_dict[key] = value    

    converted_encoder_decoder_state_dict = {}
    converted_encoder_decoder_state_dict.update(encoder_state_dict)
    converted_encoder_decoder_state_dict.update(decoder_state_dict)
    converted_encoder_decoder_state_dict.update(senseg_encoder_state_dict)

    return converted_encoder_decoder_state_dict

def bert_to_bertReorderingSensegTrans(state_dict):
    # for bertReorderingSensegTrans encoder state_dict.
    encoder_state_dict = {}
    for key in state_dict:
        value = state_dict[key]

        is_omit_key = False
        omit_keys = ["token_type_embeddings", "pooler", "predictions", "seq_relationship"]
        for omit_key in omit_keys:
            if omit_key in key:
                is_omit_key = True
        
        if is_omit_key:
            continue
        
        converted_key = key.replace("bert", "encoder").replace("gamma", "weight").replace("beta", "bias")

        encoder_state_dict[converted_key] = value
    
    # for bertReorderingSensegTrans decoder state_dict.
    decoder_state_dict = {}
    for key in state_dict:
        value = state_dict[key]

        is_omit_key = False
        omit_keys = ["token_type_embeddings", "pooler", "predictions", "seq_relationship"]
        for omit_key in omit_keys:
            if omit_key in key:
                is_omit_key = True
        
        if is_omit_key:
            continue

        converted_key = key.replace("bert", "decoder").replace("encoder", "decoder").replace("attention", "self_attention").replace("gamma", "weight").replace("beta", "bias")
        decoder_state_dict[converted_key] = value

    # for bertReorderingSensegTrans senseg_encoder state_dict.
    state_dict_ = {
        k.replace("bert.encoder", "senseg_encoder"): v for (k, v) in state_dict.items()
    }
    state_dict_ = {
        k: v for (k, v) in state_dict_.items() \
            if k.startswith("senseg_encoder")
    }

    senseg_encoder_state_dict = {}
    for key in state_dict_:
        value = state_dict_[key]

        converted_key = key.replace("gamma", "weight").replace("beta", "bias")

        senseg_encoder_state_dict[converted_key] = value

    # for bertReorderingSensegTrans trans_decoder state_dict.
    state_dict_ = {
        k.replace("bert.encoder", "trans_decoder.decoder").replace("attention", "self_attention"): v for (k, v) in state_dict.items()
    }
    state_dict_ = {
        k: v for (k, v) in state_dict_.items() \
            if k.startswith("trans_decoder.decoder")
    }

    trans_decoder_state_dict = {}
    for key in state_dict_:
        value = state_dict_[key]

        converted_key = key.replace("gamma", "weight").replace("beta", "bias")

        trans_decoder_state_dict[converted_key] = value


    converted_encoder_decoder_state_dict = {}
    converted_encoder_decoder_state_dict.update(encoder_state_dict)
    converted_encoder_decoder_state_dict.update(decoder_state_dict)
    converted_encoder_decoder_state_dict.update(senseg_encoder_state_dict)
    converted_encoder_decoder_state_dict.update(trans_decoder_state_dict)

    return converted_encoder_decoder_state_dict


def layoutlm_to_layoutlmReorderingSensegTrans(state_dict):
    # for layoutlmReorderingSensegTrans encoder state_dict.
    encoder_state_dict = {}
    for key in state_dict:
        value = state_dict[key]

        is_omit_key = False
        omit_keys = ["token_type_embeddings", "pooler", "position_ids", "predictions"]
        for omit_key in omit_keys:
            if omit_key in key:
                is_omit_key = True
        
        if is_omit_key:
            continue

        converted_key = key.replace("layoutlm", "encoder")
        encoder_state_dict[converted_key] = value
    
    # for layoutlmReorderingSensegTrans decoder state_dict.
    decoder_state_dict = {}
    for key in state_dict:
        value = state_dict[key]
        
        is_omit_key = False
        omit_keys = ["token_type_embeddings", "pooler", "position_ids", "predictions"]
        for omit_key in omit_keys:
            if omit_key in key:
                is_omit_key = True
        
        if is_omit_key:
            continue

        converted_key = key.replace("layoutlm", "decoder").replace("encoder", "decoder").replace("attention", "self_attention")
        decoder_state_dict[converted_key] = value

    # for layoutlmReorderingSensegTrans senseg encoder state_dict.
    state_dict_ = {
        k.replace("layoutlm.encoder", "senseg_encoder"): v for (k, v) in state_dict.items()
    }
    state_dict_ = {
        k: v for (k, v) in state_dict_.items() \
            if k.startswith("senseg_encoder")
    }

    senseg_encoder_state_dict = {}
    for key in state_dict_:
        value = state_dict_[key]

        senseg_encoder_state_dict[key] = value    

    # for layoutlmReorderingSensegTrans trans decoder state_dict.
    state_dict_ = {
        k.replace("layoutlm.encoder", "trans_decoder.decoder").replace("attention", "self_attention"): v for (k, v) in state_dict.items()
    }
    state_dict_ = {
        k: v for (k, v) in state_dict_.items() \
            if k.startswith("trans_decoder.decoder")
    }

    trans_decoder_state_dict = {}
    for key in state_dict_:
        value = state_dict_[key]

        trans_decoder_state_dict[key] = value

    converted_encoder_decoder_state_dict = {}
    converted_encoder_decoder_state_dict.update(encoder_state_dict)
    converted_encoder_decoder_state_dict.update(decoder_state_dict)
    converted_encoder_decoder_state_dict.update(senseg_encoder_state_dict)
    converted_encoder_decoder_state_dict.update(trans_decoder_state_dict)

    return converted_encoder_decoder_state_dict

# for word reordering task only.
# state_dict_convert = {
#     'bert': bert_to_bertEncoderDecoder,
#     'layoutlm': layoutlm_to_layoutlmEncoderDecoder,
# }


# for word reordering task + sentence segmentation task.
# state_dict_convert = {
#     'bert': bert_to_bertReorderingSenseg,
#     'layoutlm': layoutlm_to_layoutlmReorderingSenseg,
# }

# for word reordering task + sentence segmentation task + trans task.
state_dict_convert = {
    'bert': bert_to_bertReorderingSensegTrans,
    'layoutlm': layoutlm_to_layoutlmReorderingSensegTrans,
}

