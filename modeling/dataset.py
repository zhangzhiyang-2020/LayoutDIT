import torch
import os
import glob
import re
import json
import tqdm
import random

import logging
logger = logging.getLogger(__name__)

from .predefined_constant import BOS_TAG, IOS_TAG, RAW_DATASET_EOS_TOKEN, senseg_task_ctg_to_id_map

def load_and_cache_layoutlm_examples(
        example_path, tokenizer, tgt_tokenizer, local_rank, cached_features_file, max_src_length,
        layout_flag, shuffle=True,
        file_info_flag=True,
        ):
    # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    if local_rank not in [-1, 0]:
        torch.distributed.barrier()

    if cached_features_file is not None and os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else: 
        logger.info("Creating features from dataset at %s", example_path)

        examples = []

        if os.path.isdir(example_path):
            text_files = glob.glob(f'{example_path}/*text*.json')
            layout_files = [re.sub('text|txt', 'layout', x, 1) for x in text_files]
        else:
            text_files = [example_path]
            layout_files = [re.sub('text|txt', 'layout', example_path, 1)]
        for text_file, layout_file in zip(text_files, layout_files):
            with open(text_file, mode='r', encoding='utf-8') as text_reader, \
                    open(layout_file, mode='r', encoding='utf-8') as layout_reader:
                logger.info(f'Start loading {text_file}')
                for i, (text_line, layout_line) in enumerate(zip(text_reader, layout_reader)):
                    if (i + 1) % 10000 == 0:
                        logger.info(f'{i + 1} lines ...')
                    examples.append((json.loads(text_line), json.loads(layout_line)))

        features = []

        def tokenize_text_and_layout_src(_text, _layout, _layout_flag):
            ret = []
            index_split = {}
            words = _text.split()
            new_token_index = 1  # first ordinary index
            for i, (word, box) in enumerate(zip(words, _layout)):

                if (not box[2] >= box[0]) or (not box[3] >= box[1]):
                    continue

                tokens = tokenizer.tokenize(word)
                tokens = tokenizer.convert_tokens_to_ids(tokens)
                new_token_ids = []
                for token in tokens:
                    if _layout_flag: #
                        ret.append([token] + box) 
                    else:
                        ret.append(token)
                    new_token_ids.append(new_token_index)
                    new_token_index += 1
                index_split[i] = new_token_ids

            return ret, index_split


        def tokenize_text_and_layout_tgt_getSensegLabel(_text, _layout, _index, _index_split, _layout_flag, _text_sen_split, _layout_sen_split, senseg_task_ctg_to_id_map, bos_tag, ios_tag, eos_token, _tgt_text_sen_split):
            # To get ret and ret_index for word reordering task.
            ret = []
            ret_index = []
            words = _text.split()
            for word, box, i in zip(words, _layout, _index):

                if (not box[2] >= box[0]) or (not box[3] >= box[1]):
                    continue

                tokens = tokenizer.tokenize(word)
                tokens = tokenizer.convert_tokens_to_ids(tokens)
                for token, ii in zip(tokens, _index_split[i]):
                    if _layout_flag:
                        ret.append([token] + box)
                    else:
                        ret.append(token)
                    ii = min(ii, max_src_length - 1)
                    ret_index.append(ii)

            # To get senseg_labels for senseg task, and target_sen_ids_list for trans task.
            eos_token_len = len(eos_token)
            assert _tgt_text_sen_split.endswith(f" {eos_token}")
            _tgt_text_sen_split_list = _tgt_text_sen_split[:-(eos_token_len+1)].split(f"{ eos_token }")
            words = _text_sen_split.split()
            effective_words = []
            pre_word = eos_token
            for cur_word, box in zip(words, _layout_sen_split):
                if (not box[2] >= box[0]) or (not box[3] >= box[1]):
                    continue
                
                if pre_word == eos_token and cur_word != eos_token:
                    effective_words.append(cur_word)
                elif pre_word == eos_token and cur_word == eos_token:
                    effective_words.append("")
                    effective_words.append(cur_word)
                elif pre_word != eos_token and cur_word != eos_token:
                    effective_words.append(cur_word)
                elif pre_word != eos_token and cur_word == eos_token:
                    effective_words.append(cur_word)

                pre_word = cur_word
            
            _text_sen_split_ = " ".join(effective_words)
            assert _text_sen_split_.endswith(f" {eos_token}")
            _text_sen_split_list = _text_sen_split_[:-(eos_token_len+1)].split(f"{ eos_token }")

            # _tgt_text_sen_split_list = _tgt_text_sen_split.rstrip(f" {eos_token}").split(f"{ eos_token }") # bug
            # _text_sen_split_list = _text_sen_split.rstrip(f" {eos_token}").split(f"{ eos_token }") # bug
            assert len(_tgt_text_sen_split_list) == len(_text_sen_split_list)

            target_sen_ids_list = [] # list of list.
            source_sen_ids_list = []
            senseg_label_list = []
            words = _text_sen_split.split()
            tokens = []
            for word, box in zip(words, _layout_sen_split):
                if (not box[2] >= box[0]) or (not box[3] >= box[1]):
                    continue

                tokens_ = tokenizer.tokenize(word)
                tokens += tokens_
            
            is_last_token_EOS = True
            num_sen = 0
            
            for token in tokens:
                is_cur_token_EOS = (token == eos_token)
                if (is_last_token_EOS is True) and (is_cur_token_EOS is False):
                    senseg_label_list.append(bos_tag)
                    target_sen_ids_list.append(tgt_tokenizer.convert_tokens_to_ids(tgt_tokenizer.tokenize(_tgt_text_sen_split_list[num_sen])))
                    source_sen_ids_list.append(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(_text_sen_split_list[num_sen])))
                elif (is_last_token_EOS is False) and (is_cur_token_EOS is False):
                    senseg_label_list.append(ios_tag)
                elif (is_last_token_EOS is False) and (is_cur_token_EOS is True):
                    num_sen += 1
                elif (is_last_token_EOS is True) and (is_cur_token_EOS is True):
                    num_sen += 1
                
                is_last_token_EOS = is_cur_token_EOS

            # convert category label to numerical id.
            senseg_label_id_list = [int(senseg_task_ctg_to_id_map[label]) for label in senseg_label_list]

            source_sen_ids_list_total_length = sum([len(item) for item in source_sen_ids_list])
            assert source_sen_ids_list_total_length == len(senseg_label_id_list)

            # # sanity checking.
            # source_sen_lengths = [len(item) for item in source_sen_ids_list]
            # senseg_label_id_tsr = torch.tensor(senseg_label_id_list)
            # bos_token_index = torch.nonzero(senseg_label_id_tsr == 0, as_tuple=True)[0]
            # shifted_bos_token_index = torch.cat((bos_token_index[1:], torch.tensor([senseg_label_id_tsr.shape[0]], dtype=bos_token_index.dtype, device=bos_token_index.device)), dim=-1)
            # num_src_sen_tokens_caled = (shifted_bos_token_index - bos_token_index).tolist()
            # assert source_sen_lengths == num_src_sen_tokens_caled

            # assert len(ret) == len()
            assert len(ret) == len(senseg_label_id_list)

            return ret, ret_index, senseg_label_id_list, target_sen_ids_list, source_sen_ids_list
        


        for text, layout in tqdm.tqdm(examples):
            if 'bleu' in text:
                bleu = text['bleu']
            else:
                bleu = 0

            src_ids, src_index_split = tokenize_text_and_layout_src(text['src'], layout['src'],
                                                                    _layout_flag=layout_flag)
            
            
            # tgt_ids, tgt_index = tokenize_text_and_layout_tgt(text['tgt'], layout['tgt'], text['tgt_index'],
            #                                                   src_index_split, _layout_flag=layout_flag)

            tgt_ids, tgt_index, senseg_label_id_list, target_sen_ids_list, source_sen_ids_list = tokenize_text_and_layout_tgt_getSensegLabel(text['tgt'], layout['tgt'], text['tgt_index'], src_index_split, _layout_flag=layout_flag, _text_sen_split=text['tgt_sen_split'], _layout_sen_split=layout["tgt_sen_split"], senseg_task_ctg_to_id_map=senseg_task_ctg_to_id_map, bos_tag=BOS_TAG, ios_tag=IOS_TAG, eos_token=RAW_DATASET_EOS_TOKEN, _tgt_text_sen_split=text["tgt_sen_translation_split"])

            
            feature = {
                "source_ids": src_ids, 
                "target_ids": tgt_ids,
                "target_index": tgt_index, 
                "senseg_label_ids": senseg_label_id_list, 
                "target_sen_ids_list": target_sen_ids_list, 
                "source_sen_ids_list": source_sen_ids_list,
                "bleu": bleu,
            }

            

            if file_info_flag:
                file_info = {'filename': text['filename']}
                feature['file_info'] = file_info

            features.append(feature)

        if shuffle:
            random.shuffle(features)

        if local_rank in [-1, 0] and cached_features_file is not None:
            if not os.path.exists(os.path.dirname(cached_features_file)):
                os.makedirs(os.path.dirname(cached_features_file))
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    # Make sure only the first process in distributed training process the dataset, and the others will use the cache.
    if local_rank == 0:
        torch.distributed.barrier()

    return features


class Seq2seqDatasetForLayoutlm(torch.utils.data.Dataset):

    def __init__(self, features, max_source_len, max_target_len, vocab_size, cls_id, sep_id, pad_id, mask_id, random_prob, keep_prob, offset, num_training_instances, layout_flag, cls_index, pad_index, cls_layout, pad_layout, sep_layout, mask_layout, senseg_label_pad_id, src_sen_max_len, tgt_sen_max_len, target_sen_pad_id, target_sen_sep_id, target_sen_eos_id):

        self.features = features
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len
        self.vocab_size = vocab_size
        self.cls_id = cls_id
        self.sep_id = sep_id
        self.pad_id = pad_id
        self.mask_id = mask_id
        self.random_prob = random_prob
        self.keep_prob = keep_prob
        self.offset = offset
        if offset > 0:
            logger.info("  ****  Set offset %d in Seq2seqDatasetForLayoutlm ****  ", offset)
        self.num_training_instances = num_training_instances
        self.layout_flag = layout_flag
        
        self.cls_index = cls_index
        self.pad_index = pad_index
        self.cls_layout = cls_layout
        self.pad_layout = pad_layout
        self.sep_layout = sep_layout
        self.mask_layout = mask_layout

        self.senseg_label_pad_id = int(senseg_label_pad_id)

        self.src_sen_max_len = src_sen_max_len
        self.tgt_sen_max_len = tgt_sen_max_len
        self.target_sen_pad_id = target_sen_pad_id
        self.target_sen_sep_id = target_sen_sep_id
        self.target_sen_eos_id = target_sen_eos_id



    def __len__(self):
        
        return int(self.num_training_instances)
    
    def __trunk(self, ids, max_len):
     

        if len(ids) >= max_len:
            ids = ids[:max_len-1]
        return ids
    
    def __clip_index(self, ids, replace_value): 
       

        # replace_value = self.cls_index
        for i in range(len(ids)):
            if ids[i] >= self.max_source_len:
                ids[i] = replace_value
        return ids
    
    def __pad(self, ids, max_len, layout_flag, pad_value_id, pad_value_layout):
        
        if len(ids) < max_len - 1:
            if layout_flag:
                return ids + [[pad_value_id] + pad_value_layout] * (max_len - 1 - len(ids))
            else:
                return ids + [pad_value_id] * (max_len - 1 - len(ids))
        else:
            assert len(ids) == max_len - 1
            return ids
        
    def __prepend_or_append(self, ids, value_id, value_layout, prepend_or_append, layout_flag):
        
        if prepend_or_append == "prepend":
            if layout_flag:
                return [[value_id] + value_layout] + ids
            else:
                return [value_id] + ids
        elif prepend_or_append == "append":
            return ids + [value_id]
        
    def __getitem_bert__(self, idx):

        idx = (self.offset + idx) % len(self.features)
        feature = self.features[idx]

        # truncate.
        source_ids = self.__trunk(feature["source_ids"], self.max_source_len)
        target_ids = self.__trunk(feature["target_ids"], self.max_target_len)
        target_index = self.__trunk(feature["target_index"], self.max_target_len)
        senseg_label_ids = self.__trunk(feature["senseg_label_ids"], self.max_target_len) # for senseg task.

        # for trans task.
        num_ids = 0
        target_sen_ids_list = feature["target_sen_ids_list"]
        source_sen_ids_list = feature["source_sen_ids_list"]
        trunked_source_sen_ids_list = []
        trunked_target_sen_ids_list = []
        for i, source_sen_ids in enumerate(source_sen_ids_list):
            if num_ids < len(target_ids):
                trunked_source_sen_ids = []
                for source_sen_id in source_sen_ids:
                    if num_ids + 1 <= len(target_ids):
                        trunked_source_sen_ids.append(source_sen_id)
                        num_ids += 1
                trunked_source_sen_ids_list.append(trunked_source_sen_ids)
                trunked_target_sen_ids_list.append(target_sen_ids_list[i])
            else:
                break
        
        assert len(trunked_source_sen_ids_list) == len(trunked_target_sen_ids_list)
        assert len(target_ids) == sum([len(item) for item in trunked_source_sen_ids_list])

        target_sen_ids_list = [self.__trunk(target_sen_ids, self.tgt_sen_max_len) for target_sen_ids in trunked_target_sen_ids_list] # for trans task.
        source_sen_ids_list = [self.__trunk(source_sen_ids, self.src_sen_max_len+1) for source_sen_ids in trunked_source_sen_ids_list] # for trans task.


        
        # memorize the effecitve num_src_tgt_tokens.
        num_effective_src_tokens = len(source_ids)
        num_effective_tgt_tokens = len(target_ids)
        num_effective_tgt_sen_tokens_list = [len(target_sen_ids) for target_sen_ids in target_sen_ids_list] # for trans task.
        num_effective_src_sen_tokens_list = [len(source_sen_ids) for source_sen_ids in source_sen_ids_list] # for trans task.


        # create pseudo_target_ids.
        pseudo_target_ids = []
        for tk_id in target_ids:
            p = random.random()
            if p < self.keep_prob:
                pseudo_target_ids.append(tk_id)
            elif p < self.keep_prob + self.random_prob:
                pseudo_target_ids.append(random.randint(0, self.vocab_size - 1))
            else:
                pseudo_target_ids.append(self.mask_id)

        # replace invalid label in target_index.
        target_index = self.__clip_index(target_index, self.cls_index)

        # padding.
        source_ids = self.__pad(source_ids, self.max_source_len, layout_flag=False, pad_value_id=self.pad_id, pad_value_layout=None)
        target_ids = self.__pad(target_ids, self.max_target_len, layout_flag=False, pad_value_id=self.pad_id, pad_value_layout=None)
        pseudo_target_ids = self.__pad(pseudo_target_ids, self.max_target_len, layout_flag=False, pad_value_id=self.pad_id, pad_value_layout=None)
        target_index = self.__pad(target_index, self.max_target_len, layout_flag=False, pad_value_id=self.pad_index, pad_value_layout=None)
        senseg_label_ids = self.__pad(senseg_label_ids, self.max_target_len, layout_flag=False, pad_value_id=self.senseg_label_pad_id, pad_value_layout=None) # for senseg task.
        target_sen_ids_list = [self.__pad(target_sen_ids, self.tgt_sen_max_len, layout_flag=False, pad_value_id=self.target_sen_pad_id, pad_value_layout=None) for target_sen_ids in target_sen_ids_list] # for trans task.

        # prepend & append.
        source_ids = self.__prepend_or_append(source_ids, self.cls_id, value_layout=None, prepend_or_append="prepend", layout_flag=False)
        target_ids = self.__prepend_or_append(target_ids, self.sep_id, value_layout=None, prepend_or_append="prepend", layout_flag=False)
        pseudo_target_ids = self.__prepend_or_append(pseudo_target_ids, self.sep_id, value_layout=None, prepend_or_append="prepend", layout_flag=False)
        target_index = self.__prepend_or_append(target_index, self.pad_index, value_layout=None, prepend_or_append="append", layout_flag=False)
        senseg_label_ids = self.__prepend_or_append(senseg_label_ids, self.senseg_label_pad_id, value_layout=None, prepend_or_append="append", layout_flag=False) # for senseg task.
        target_sen_input_ids_list = [self.__prepend_or_append(target_sen_ids, self.target_sen_sep_id, value_layout=None, prepend_or_append="prepend", layout_flag=False) for target_sen_ids in target_sen_ids_list] # for trans task.
        target_sen_labels_list = [self.__prepend_or_append(target_sen_ids, self.target_sen_pad_id, value_layout=None, prepend_or_append="append", layout_flag=False) for target_sen_ids in target_sen_ids_list] # for trans task.


        # post-process the target_index.
        target_index[num_effective_tgt_tokens] = self.cls_index
        # post-process the target_sen_labels_list.
        target_sen_labels_list_ = []
        for target_sen_labels, num_effective_tgt_sen_tokens in zip(target_sen_labels_list, num_effective_tgt_sen_tokens_list):
            target_sen_labels[num_effective_tgt_sen_tokens] = self.target_sen_eos_id
            target_sen_labels_list_.append(target_sen_labels)
        
        target_sen_labels_list = target_sen_labels_list_

        # for trans task.
        num_sens = len(target_sen_labels_list)

        # sanity checking.
        senseg_label_id_tsr = torch.tensor(senseg_label_ids)
        bos_token_index = torch.nonzero(senseg_label_id_tsr[:num_effective_tgt_tokens] == 0, as_tuple=True)[0]
        shifted_bos_token_index = torch.cat((bos_token_index[1:], torch.tensor
        ([num_effective_tgt_tokens], dtype=bos_token_index.dtype, device=bos_token_index.device)), dim=-1)
        num_src_sen_tokens_caled = (shifted_bos_token_index - bos_token_index).tolist()
        num_src_sen_tokens_caled = [num if num <= self.src_sen_max_len else self.src_sen_max_len for num in num_src_sen_tokens_caled]

        assert num_src_sen_tokens_caled == num_effective_src_sen_tokens_list

        return source_ids, target_ids, pseudo_target_ids, target_index, senseg_label_ids, target_sen_input_ids_list, target_sen_labels_list, num_effective_src_tokens, num_effective_tgt_tokens, num_effective_src_sen_tokens_list, num_effective_tgt_sen_tokens_list

    def __getitem_layout__(self, idx):

        idx = (self.offset + idx) % len(self.features)
        feature = self.features[idx]

        # truncate.
        source_ids = self.__trunk(feature["source_ids"], self.max_source_len)
        target_ids = self.__trunk(feature["target_ids"], self.max_target_len)
        target_index = self.__trunk(feature["target_index"], self.max_target_len)
        senseg_label_ids = self.__trunk(feature["senseg_label_ids"], self.max_target_len) # for senseg task.

        # for trans task.
        num_ids = 0
        target_sen_ids_list = feature["target_sen_ids_list"]
        source_sen_ids_list = feature["source_sen_ids_list"]
        trunked_source_sen_ids_list = []
        trunked_target_sen_ids_list = []
        for i, source_sen_ids in enumerate(source_sen_ids_list):
            if num_ids < len(target_ids):
                trunked_source_sen_ids = []
                for source_sen_id in source_sen_ids:
                    if num_ids + 1 <= len(target_ids):
                        trunked_source_sen_ids.append(source_sen_id)
                        num_ids += 1
                trunked_source_sen_ids_list.append(trunked_source_sen_ids)
                trunked_target_sen_ids_list.append(target_sen_ids_list[i])
            else:
                break
        
        assert len(trunked_source_sen_ids_list) == len(trunked_target_sen_ids_list)
        assert len(target_ids) == sum([len(item) for item in trunked_source_sen_ids_list])


        target_sen_ids_list = [self.__trunk(target_sen_ids, self.tgt_sen_max_len) for target_sen_ids in trunked_target_sen_ids_list] # for trans task.
        source_sen_ids_list = [self.__trunk(source_sen_ids, self.src_sen_max_len+1) for source_sen_ids in trunked_source_sen_ids_list] # for trans task.

        # memorize the effecitve num_src_tgt_tokens
        num_effective_src_tokens = len(source_ids)
        num_effective_tgt_tokens = len(target_ids)
        num_effective_tgt_sen_tokens_list = [len(target_sen_ids) for target_sen_ids in target_sen_ids_list] # for trans task.
        num_effective_src_sen_tokens_list = [len(source_sen_ids) for source_sen_ids in source_sen_ids_list] # for trans task.

        # create pseudo_target_ids.
        pseudo_target_ids = []
        for tk_id in target_ids:
            p = random.random()
            if p < self.keep_prob:
                pseudo_target_ids.append(tk_id)
            elif p < self.keep_prob + self.random_prob:
                pseudo_target_ids.append([random.randint(0, self.vocab_size - 1)] + self.mask_layout)
            else:
                pseudo_target_ids.append([self.mask_id] + self.mask_layout)

        # replace invalid label in target_index.
        target_index = self.__clip_index(target_index, self.cls_index)

        # padding.
        source_ids = self.__pad(source_ids, self.max_source_len, layout_flag=True, pad_value_id=self.pad_id, pad_value_layout=self.pad_layout)
        target_ids = self.__pad(target_ids, self.max_target_len, layout_flag=True, pad_value_id=self.pad_id, pad_value_layout=self.pad_layout)
        pseudo_target_ids = self.__pad(pseudo_target_ids, self.max_target_len, layout_flag=True, pad_value_id=self.pad_id, pad_value_layout=self.pad_layout)
        target_index = self.__pad(target_index, self.max_target_len, layout_flag=False, pad_value_id=self.pad_index, pad_value_layout=None)
        senseg_label_ids = self.__pad(senseg_label_ids, self.max_target_len, layout_flag=False, pad_value_id=self.senseg_label_pad_id, pad_value_layout=None) # for senseg task.
        target_sen_ids_list = [self.__pad(target_sen_ids, self.tgt_sen_max_len, layout_flag=False, pad_value_id=self.target_sen_pad_id, pad_value_layout=None) for target_sen_ids in target_sen_ids_list] # for trans task.

        # prepend & append.
        source_ids = self.__prepend_or_append(source_ids, self.cls_id, value_layout=self.cls_layout, prepend_or_append="prepend", layout_flag=True)
        target_ids = self.__prepend_or_append(target_ids, self.sep_id, value_layout=self.sep_layout, prepend_or_append="prepend", layout_flag=True)
        pseudo_target_ids = self.__prepend_or_append(pseudo_target_ids, self.sep_id, value_layout=self.sep_layout, prepend_or_append="prepend", layout_flag=True)
        target_index = self.__prepend_or_append(target_index, self.pad_index, value_layout=None, prepend_or_append="append", layout_flag=False)
        senseg_label_ids = self.__prepend_or_append(senseg_label_ids, self.senseg_label_pad_id, value_layout=None, prepend_or_append="append", layout_flag=False) # for senseg task.
        target_sen_input_ids_list = [self.__prepend_or_append(target_sen_ids, self.target_sen_sep_id, value_layout=None, prepend_or_append="prepend", layout_flag=False) for target_sen_ids in target_sen_ids_list] # for trans task.
        target_sen_labels_list = [self.__prepend_or_append(target_sen_ids, self.target_sen_pad_id, value_layout=None, prepend_or_append="append", layout_flag=False) for target_sen_ids in target_sen_ids_list] # for trans task.

        # post-process the target_index.
        target_index[num_effective_tgt_tokens] = self.cls_index
        # post-process the target_sen_labels_list.
        target_sen_labels_list_ = []
        for target_sen_labels, num_effective_tgt_sen_tokens in zip(target_sen_labels_list, num_effective_tgt_sen_tokens_list):
            target_sen_labels[num_effective_tgt_sen_tokens] = self.target_sen_eos_id
            target_sen_labels_list_.append(target_sen_labels)
        
        target_sen_labels_list = target_sen_labels_list_

        # for trans task.
        num_sens = len(target_sen_labels_list)

        # sanity checking.
        senseg_label_id_tsr = torch.tensor(senseg_label_ids)
        bos_token_index = torch.nonzero(senseg_label_id_tsr[:num_effective_tgt_tokens] == 0, as_tuple=True)[0]
        # print(bos_token_index)
        shifted_bos_token_index = torch.cat((bos_token_index[1:], torch.tensor
        ([num_effective_tgt_tokens], dtype=bos_token_index.dtype, device=bos_token_index.device)), dim=-1)
        # print(shifted_bos_token_index)
        num_src_sen_tokens_caled = (shifted_bos_token_index - bos_token_index).tolist()
        num_src_sen_tokens_caled = [num if num <= self.src_sen_max_len else self.src_sen_max_len for num in num_src_sen_tokens_caled]
        
 
        assert num_src_sen_tokens_caled == num_effective_src_sen_tokens_list

        return source_ids, target_ids, pseudo_target_ids, target_index, senseg_label_ids, target_sen_input_ids_list, target_sen_labels_list, num_effective_src_tokens, num_effective_tgt_tokens, num_effective_src_sen_tokens_list, num_effective_tgt_sen_tokens_list
    
    def __getitem__(self, idx):
        if self.layout_flag:
            return self.__getitem_layout__(idx)
        else:
            return self.__getitem_bert__(idx)