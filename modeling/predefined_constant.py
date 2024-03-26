from transformers.modeling_bert import BERT_PRETRAINED_MODEL_ARCHIVE_MAP


CLS_TOKEN_INDEX = 0
PAD_TOKEN_INDEX = 0
# SEP_TOKEN_INDEX = 0
CLS_TOKEN_LAYOUT = [1000, 1000, 1000, 1000]
PAD_TOKEN_LAYOUT = [0, 0, 0, 0]
SEP_TOKEN_LAYOUT = [0, 0, 0, 0]
MASK_TOKEN_LAYOUT = [0, 0, 0, 0]


LAYOUTLM_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    'layoutlm-base-uncased': 'https://huggingface.co/microsoft/layoutlm-base-uncased/resolve/main/config.json',
    'layoutlm-large-uncased': 'https://huggingface.co/microsoft/layoutlm-large-uncased/resolve/main/config.json'
}

LAYOUTLM_PRETRAINED_MODEL_ARCHIVE_MAP = {
    'layoutlm-base-uncased': 'https://huggingface.co/microsoft/layoutlm-base-uncased/resolve/main/pytorch_model.bin',
    'layoutlm-large-uncased': 'https://huggingface.co/microsoft/layoutlm-large-uncased/resolve/main/pytorch_model.bin'
}

# For senseg task.
BOS_TAG = "B"
IOS_TAG = "I"

BOS_TAG_ID = 0
IOS_TAG_ID = 1

senseg_task_ctg_to_id_map = {
    BOS_TAG: BOS_TAG_ID,
    IOS_TAG: IOS_TAG_ID,
}

senseg_task_id_to_ctg_map = {
    v: k for (k, v) in senseg_task_ctg_to_id_map.items()
}

SENSEG_LABEL_PAD_ID = senseg_task_ctg_to_id_map[BOS_TAG]

RAW_DATASET_EOS_TOKEN = "[CLS]"

