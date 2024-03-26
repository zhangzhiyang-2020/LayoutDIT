import torch

import numpy as np

import math
import random
import os
import glob

from transformers import AutoTokenizer, BertTokenizer

Chinese_tokenizer_dir = "/path/to/Chinese_tokenizer_dir"
Chinese_tokenizer = BertTokenizer.from_pretrained(Chinese_tokenizer_dir)


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)

ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}


def set_seed(see_num, num_gpu):
    random.seed(see_num)
    np.random.seed(see_num)
    torch.manual_seed(see_num)
    if num_gpu > 0:
        torch.cuda.manual_seed_all(see_num)

def get_max_step_model_optim(output_dir):

    fn_model_list = glob.glob(os.path.join(output_dir, "ckpt-*"))
    fn_optim_list = glob.glob(os.path.join(output_dir, "optim.*.bin"))
    if (not fn_model_list) or (not fn_optim_list):
        return None
    os.path.basename(output_dir)
    both_set = set([int(os.path.basename(fn).split('-')[1]) for fn in fn_model_list]
                   ) & set([int(os.path.basename(fn).split('.')[1]) for fn in fn_optim_list])
    if both_set:
        max_step = max(both_set)
        max_step_model_dir = [model_dir for model_dir in fn_model_list if str(max_step) in model_dir][0]
        max_step_optim_path = [optim_path for optim_path in fn_optim_list if str(max_step) in optim_path][0]
        return max_step, max_step_model_dir, max_step_optim_path
    else:
        return None
    

def is_all_items_the_same_len(list_of_list):
    is_the_same_len = True
    basic_len = len(list_of_list[0])
    for list_ in list_of_list:
        if len(list_) != basic_len:
            is_the_same_len = False
            break

    return is_the_same_len


class BatchListToBatchTensorsCollator():
    def __init__(self, max_source_len, max_target_len):
        self.max_source_len = max_source_len,
        self.max_target_len = max_target_len

    def __call__(self, batch):
        max_source_len = self.max_source_len
        max_target_len = self.max_target_len

        batch_tensors = []
        for x in zip(*batch):
            
            first_item = x[0]
            if not isinstance(first_item, list): # num_effective_src_tokens, num_effective_tgt_tokens
                batched_tensor = torch.tensor(x, dtype=torch.long)
            else:
                first_item_ele = first_item[0]
                if not isinstance(first_item_ele, list): # (source_ids), (target_ids), (pseudo_target_ids), target_index, senseg_label_ids, num_effective_src_sen_tokens_list, num_effective_tgt_sen_tokens_list
                    if (len(first_item) != max_source_len) and (len(first_item) != max_target_len): # num_effective_src_sen_tokens_list, num_effective_tgt_sen_tokens_list
                        all_item_list = []
                        for item in x:
                            all_item_list += item
                        batched_tensor = torch.tensor(all_item_list, dtype=torch.long)
                    else: # (source_ids), (target_ids), (pseudo_target_ids), target_index, senseg_label_ids
                        batched_tensor = torch.tensor(x, dtype=torch.long)
                elif isinstance(first_item_ele, list) and len(first_item_ele) == 5: # (source_ids), (target_ids), (pseudo_target_ids)
                    batched_tensor = torch.tensor(x, dtype=torch.long)
                else: # target_sen_input_ids_list, target_sen_labels_list,  
                    all_item_list = []
                    for item in x:
                        all_item_list += item
                    batched_tensor = torch.tensor(all_item_list, dtype=torch.long)

            batch_tensors.append(batched_tensor)
        return batch_tensors



def wdp_detokenizer(tk_list):
    r_list = []
    for tk in tk_list:
        if tk.startswith('##') and len(r_list) > 0:
            r_list[-1] = r_list[-1] + tk[2:]
        else:
            r_list.append(tk)
    return r_list

# def spm_detokenizer(tk_list):
#     r_list = []
#     for i, tk in enumerate(tk_list):
#         if i == 0:
#             tk = tk.replace("▁", "")
#         else:
#             if tk == "▁":
#                 tk = " "
        
#         r_list.append(tk)

#     return r_list

def spm_detokenizer(tk_list):
    detok_seq = "".join(tk_list).replace("▁", " ")
    return detok_seq

def convert_src_layout_inputs_to_tokens(inputs, converter, max_src_length, layout_flag=True):
    ret = []
    if not layout_flag: 
        for line in inputs:
            ret.append(converter(line["source_ids"])[: max_src_length])
    else: 
        for line in inputs:
            raw_text_ids = [x[0] for x in line['source_ids']]
            raw_text = converter(raw_text_ids)
            new_line = [[t] + x[1:] for t, x in zip(raw_text, line['source_ids'])][: max_src_length]
            ret.append(new_line)
    return ret


def convert_tgt_layout_inputs_to_tokens(inputs, converter, max_tgt_length, layout_flag=True):
    ret = []
    if not layout_flag:
        for line in inputs:
            ret.append(converter(line["target_ids"])[: max_tgt_length])
    else: 
        for line in inputs:
            raw_text_ids = [x[0] for x in line['target_ids']]
            ret.append(converter(raw_text_ids)[: max_tgt_length])
    return ret

# For senseg task.
def senesg_convert_tgt_layout_inputs_to_tokens_wi_eos(inputs, converter, max_tgt_length, senseg_task_id_to_ctg_map, bos_tag, ios_tag, eos_token, layout_flag=True):
    ret = []
    for line in inputs:
        if not layout_flag:
            target_ids = line["target_ids"]
        else:
            target_ids = [x[0] for x in line["target_ids"]]

        senseg_label_ids = line["senseg_label_ids"]
        assert len(target_ids) == len(senseg_label_ids)

        target_ids, senseg_label_ids = target_ids[: max_tgt_length], senseg_label_ids[: max_tgt_length]
        target_tokens = converter(target_ids)
        senseg_labels = [senseg_task_id_to_ctg_map[id_] for id_ in senseg_label_ids]


        target_tokens_wi_eos = []
        i = 0
        while i < len(target_tokens) - 1:
            is_next_token_bos_tag = (senseg_labels[i+1] == bos_tag)
            if not is_next_token_bos_tag:
                target_tokens_wi_eos.append(target_tokens[i])
            else:
                target_tokens_wi_eos.append(target_tokens[i])
                target_tokens_wi_eos.append(eos_token)

            i += 1
        # Append the last token and eos token.
        target_tokens_wi_eos.append(target_tokens[i])
        target_tokens_wi_eos.append(eos_token)

        ret.append(target_tokens_wi_eos)
    
    return ret

# For trans task.
def trans_convert_src_tgt_inputs_to_tokens(inputs, src_converter, tgt_converter, src_sen_max_len, tgt_sen_max_len):
    src_list_sen_tokens_list, tgt_list_sen_tokens_list = [], []
    for line in inputs:
        src_sen_ids_list = line["source_sen_ids_list"]
        tgt_sen_ids_list = line["target_sen_ids_list"]

        assert len(src_sen_ids_list) == len(tgt_sen_ids_list)

        src_sen_tokens_list, tgt_sen_tokens_list = [], []

        for src_sen_ids, tgt_sen_ids in zip(src_sen_ids_list, tgt_sen_ids_list):
            src_sen_tokens_list.append(src_converter(src_sen_ids)[: src_sen_max_len])
            tgt_sen_tokens_list.append(tgt_converter(tgt_sen_ids)[: tgt_sen_max_len])
        
        src_list_sen_tokens_list.append(src_sen_tokens_list)
        tgt_list_sen_tokens_list.append(tgt_sen_tokens_list)

    return src_list_sen_tokens_list, tgt_list_sen_tokens_list
